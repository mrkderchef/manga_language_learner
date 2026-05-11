from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.image_service import ImageService
from services.ocr_service import OCRService
from services.translation_service import TranslationService
import logging
import json
from pathlib import Path
from config import BASE_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/learning", tags=["learning"])

ocr_service = OCRService()
translation_service = TranslationService()

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

    # OCR ausführen
    ocr_result = ocr_service.extract_text_from_image(str(panel_path))
    if not ocr_result["success"]:
        raise HTTPException(status_code=500, detail="OCR failed")

    text = ocr_result.get("text", "")
    if not text:
        return {"vocab": [], "panel": filename}

    # Text in Wörter/Phrasen zerlegen (einfache Variante: Zeilen)
    # TODO: Morphologische Analyse (MeCab o.ä.) für bessere Segmentierung
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    vocab = []
    for line in lines:
        # Übersetze jede Zeile einzeln
        trans_result = translation_service.translate_text(line)
        meaning = trans_result.get("translated", "") if trans_result.get("success") else ""
        vocab.append({
            "japanese": line,
            "reading": "",  # TODO: Furigana/Reading via NLP
            "meaning": meaning,
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
