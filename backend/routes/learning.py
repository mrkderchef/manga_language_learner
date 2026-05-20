from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.image_service import ImageService
from services import japanese_learning_service
from routes.scanner import _run_ocr
import logging
import json
from pathlib import Path
from config import BASE_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/learning", tags=["learning"])

# Simple file-based progress store
PROGRESS_FILE = BASE_DIR / "backend" / "data" / "progress.json"
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)


class AnswerRequest(BaseModel):
    word: str
    knew: bool


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"words": {}, "panels_completed": []}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


@router.get("/panels")
async def get_learning_panels():
    """Panels für den Lernmodus auflisten"""
    return ImageService.get_all_panels()


@router.get("/{filename}/vocab")
async def get_panel_vocab(filename: str):
    """
    Vokabeln aus einem Panel extrahieren.
    Führt OCR aus und zerlegt den Text in Lern-Einheiten.
    """
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")

    # Use shared cached OCR pipeline
    result = _run_ocr(panel_path)

    if not result or not result.get("success"):
        raise HTTPException(status_code=503, detail="No vision service available")

    annotations = result.get("annotations", [])
    vocab = []
    for ann in annotations:
        jp = ann.get("text", "").strip()
        if jp:
            vocab.append({
                "japanese": jp,
                "reading": ann.get("reading_romaji") or ann.get("reading_kana", ""),
                "reading_kana": ann.get("reading_kana", ""),
                "tokens": ann.get("tokens", []),
                "meaning": ann.get("translated", ""),
            })

    return {"vocab": vocab, "panel": filename}


@router.post("/{filename}/answer")
async def submit_answer(filename: str, req: AnswerRequest):
    """Antwort für ein Wort speichern (wusste/wusste nicht)"""
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
    """Gesamtfortschritt abrufen"""
    progress = load_progress()
    total_words = len(progress["words"])
    mastered = sum(1 for w in progress["words"].values() if w["correct"] >= 3)

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
    return japanese_learning_service.lookup_text(text)


@router.get("/kanji/{character}")
async def lookup_kanji(character: str):
    if not character:
        raise HTTPException(status_code=400, detail="No kanji provided")
    return japanese_learning_service.lookup_kanji(character)


@router.get("/word")
async def lookup_word(text: str):
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
    return japanese_learning_service.lookup_word(text)


@router.get("/reading/{reading}")
async def lookup_reading(reading: str):
    if not reading:
        raise HTTPException(status_code=400, detail="No reading provided")
    return japanese_learning_service.lookup_reading(reading)
