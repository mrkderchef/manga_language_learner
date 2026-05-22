from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import BASE_DIR
from routes.scanner import _run_ocr, _run_rabbithole_existing
from services.rabbithole import nlp as rabbithole_service
from services.storage.image_service import ImageService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rabbithole", tags=["rabbithole"])

PROGRESS_FILE = BASE_DIR / "backend" / "data" / "progress.json"
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)


class AnswerRequest(BaseModel):
    word: str
    knew: bool


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"words": {}, "panels_completed": []}


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


@router.get("/panels")
async def get_rabbithole_panels():
    return ImageService.get_all_panels()


@router.get("/{filename}/vocab")
async def get_panel_vocab(filename: str):
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")

    ocr_result = _run_ocr(panel_path)
    if not ocr_result or not ocr_result.get("success"):
        raise HTTPException(status_code=503, detail="No OCR service available")

    rabbithole_result = _run_rabbithole_existing(panel_path)
    if not rabbithole_result or not rabbithole_result.get("success"):
        raise HTTPException(status_code=503, detail="No Rabbithole data available")

    vocab = []
    for ann in rabbithole_result.get("annotations", []):
        jp = str(ann.get("text") or "").strip()
        rabbit = ann.get("rabbithole") or {}
        if jp:
            vocab.append({
                "japanese": jp,
                "reading": rabbit.get("reading_romanji") or rabbit.get("reading_hiragana", ""),
                "reading_hiragana": rabbit.get("reading_hiragana", ""),
                "tokens": rabbit.get("segments", []),
                "meaning": ann.get("translated") or rabbit.get("glossary", ""),
            })

    return {"vocab": vocab, "panel": filename}


@router.post("/{filename}/answer")
async def submit_answer(filename: str, req: AnswerRequest):
    progress = load_progress()
    word_key = req.word

    if word_key not in progress["words"]:
        progress["words"][word_key] = {"seen": 0, "correct": 0}

    progress["words"][word_key]["seen"] += 1
    if req.knew:
        progress["words"][word_key]["correct"] += 1

    save_progress(progress)
    return {"success": True, "word": word_key, "stats": progress["words"][word_key]}


@router.get("/progress")
async def get_progress():
    progress = load_progress()
    total_words = len(progress["words"])
    mastered = sum(1 for word in progress["words"].values() if word["correct"] >= 3)

    return {
        "total_words_seen": total_words,
        "mastered": mastered,
        "panels_completed": progress["panels_completed"],
        "words": progress["words"],
    }


@router.get("/lookup")
async def lookup_text(text: str):
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
    return rabbithole_service.lookup_text(text)


@router.get("/kanji/{character}")
async def lookup_kanji(character: str):
    if not character:
        raise HTTPException(status_code=400, detail="No kanji provided")
    return rabbithole_service.lookup_kanji(character)


@router.get("/word")
async def lookup_word(text: str):
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
    return rabbithole_service.lookup_word(text)


@router.get("/reading/{reading}")
async def lookup_reading(reading: str):
    if not reading:
        raise HTTPException(status_code=400, detail="No reading provided")
    return rabbithole_service.lookup_reading(reading)
