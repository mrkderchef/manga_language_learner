from __future__ import annotations

import asyncio
import json
import threading

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import BASE_DIR
from routes.scanner import _run_ocr, _run_rabbithole_existing
from services.storage.image_service import ImageService
from services.storage.json_store import write_json_atomic

router = APIRouter(prefix="/api/learning", tags=["learning"])

LEARNING_PROGRESS_FILE = BASE_DIR / "backend" / "data" / "learning_progress.json"
LEARNING_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
_LEARNING_PROGRESS_LOCK = threading.Lock()


class LearningAnswerRequest(BaseModel):
    word: str
    knew: bool
    panel_complete: bool = False


def load_learning_progress() -> dict:
    default = {"words": {}, "panels_completed": []}
    if not LEARNING_PROGRESS_FILE.exists():
        return default
    try:
        payload = json.loads(LEARNING_PROGRESS_FILE.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return default
        payload.setdefault("words", {})
        payload.setdefault("panels_completed", [])
        return payload
    except (OSError, json.JSONDecodeError):
        return default


def save_learning_progress(progress: dict) -> None:
    write_json_atomic(LEARNING_PROGRESS_FILE, progress)


@router.get("/panels")
async def get_learning_panels():
    return ImageService.get_all_panels()


@router.get("/{filename}/vocab")
async def get_learning_panel_vocab(filename: str):
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")

    ocr_result = await asyncio.to_thread(_run_ocr, panel_path)
    if not ocr_result or not ocr_result.get("success"):
        raise HTTPException(status_code=503, detail="No OCR data available")

    rabbithole_result = await asyncio.to_thread(_run_rabbithole_existing, panel_path)
    if not rabbithole_result or not rabbithole_result.get("success"):
        raise HTTPException(status_code=503, detail="No Rabbithole data available")

    vocab = []
    for annotation in rabbithole_result.get("annotations", []):
        japanese = str(annotation.get("text") or "").strip()
        rabbithole = annotation.get("rabbithole") or {}
        if japanese:
            vocab.append({
                "japanese": japanese,
                "reading": rabbithole.get("reading_romanji") or rabbithole.get("reading_hiragana", ""),
                "reading_hiragana": rabbithole.get("reading_hiragana", ""),
                "tokens": rabbithole.get("segments", []),
                "meaning": annotation.get("translated") or rabbithole.get("glossary", ""),
            })

    return {"vocab": vocab, "panel": filename}


@router.post("/{filename}/answer")
async def submit_learning_answer(filename: str, request: LearningAnswerRequest):
    if not ImageService.get_panel_by_filename(filename):
        raise HTTPException(status_code=404, detail="Panel not found")
    with _LEARNING_PROGRESS_LOCK:
        progress = load_learning_progress()
        word_stats = progress["words"].setdefault(request.word, {"seen": 0, "correct": 0})
        word_stats["seen"] += 1
        if request.knew:
            word_stats["correct"] += 1
        if request.panel_complete and filename not in progress["panels_completed"]:
            progress["panels_completed"].append(filename)
        save_learning_progress(progress)
    return {"success": True, "word": request.word, "stats": word_stats}


@router.get("/progress")
async def get_learning_progress():
    with _LEARNING_PROGRESS_LOCK:
        progress = load_learning_progress()
    words = progress["words"]
    return {
        "total_words_seen": len(words),
        "mastered": sum(1 for word in words.values() if word["correct"] >= 3),
        "panels_completed": progress["panels_completed"],
        "words": words,
    }
