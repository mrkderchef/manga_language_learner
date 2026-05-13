from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from services.image_service import ImageService
from services.ollama_service import OllamaOCRService
from services import manga_ocr_service
from services.panel_renderer import render_translated_panel
import asyncio
import hashlib
import json
import logging
import traceback
from pathlib import Path
from config import BASE_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/scanner", tags=["scanner"])

# Primary: manga-ocr (comic-text-detector + manga_ocr model)
# Fallback: Ollama vision model
ollama_service = OllamaOCRService()

# OCR result cache directory
OCR_CACHE_DIR = BASE_DIR / "backend" / "data" / "ocr_cache"
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OCR_CACHE_VERSION = "ocr-preprocessing-matrix-v1"
MISSING_TRANSLATION_TEXT = "No translation available"


def _get_vision_service():
    """Return the first available vision service."""
    if ollama_service.is_available():
        return ollama_service
    return None


def _normalize_scan_result(result: dict | None) -> dict | None:
    if not result or not result.get("success"):
        return result

    normalized = dict(result)
    annotations = []
    for ann in result.get("annotations", []) or []:
        copy = dict(ann)
        translated = str(copy.get("translated") or "").strip()
        has_bbox = bool(copy.get("bbox"))
        if not has_bbox:
            copy["localization_missing"] = True
        elif translated in {"", "—", "..."}:
            copy["translated"] = MISSING_TRANSLATION_TEXT
            copy["translation_missing"] = True
        annotations.append(copy)
    normalized["annotations"] = annotations
    return normalized


def _cache_key(panel_path: Path) -> str:
    """Generate a cache key from file path + modification time."""
    stat = panel_path.stat()
    raw = f"{OCR_CACHE_VERSION}:{panel_path}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_cached_result(panel_path: Path):
    """Return cached OCR result if available and still valid."""
    key = _cache_key(panel_path)
    cache_file = OCR_CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            return _normalize_scan_result(json.loads(cache_file.read_text(encoding="utf-8")))
        except Exception:
            pass
    return None


def _save_to_cache(panel_path: Path, result: dict):
    """Persist a successful OCR result to disk cache."""
    key = _cache_key(panel_path)
    cache_file = OCR_CACHE_DIR / f"{key}.json"
    try:
        cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to write OCR cache: {e}")


def _run_ocr(panel_path: Path) -> dict:
    """Run OCR with caching. Primary: manga-ocr, fallback: Ollama."""
    cached = _get_cached_result(panel_path)
    if cached:
        logger.info(f"OCR cache hit for {panel_path.name}")
        return cached

    # Primary: manga-ocr (comic-text-detector + manga_ocr model)
    try:
        if manga_ocr_service.is_available():
            result = manga_ocr_service.extract_and_translate(str(panel_path))
            if result["success"]:
                result = _normalize_scan_result(result)
                _save_to_cache(panel_path, result)
                return result
            logger.warning(f"manga-ocr pipeline failed: {result.get('error')}")
    except Exception as e:
        logger.error(f"manga-ocr pipeline exception: {e}\n{traceback.format_exc()}")

    # Fallback: LLM-based vision service (Ollama)
    for svc in [ollama_service]:
        try:
            if svc.is_available():
                result = svc.extract_and_translate(str(panel_path))
                if result["success"]:
                    result = _normalize_scan_result(result)
                    _save_to_cache(panel_path, result)
                    return result
                logger.warning(f"{type(svc).__name__} failed: {result.get('error')}")
        except Exception as e:
            logger.error(f"{type(svc).__name__} exception: {e}\n{traceback.format_exc()}")

    return None


def _run_scan_translate(panel_path: Path) -> dict:
    """Run cached OCR/translation, then attach a generated translated panel."""
    result = _run_ocr(panel_path)
    if not result or not result.get("success"):
        return result

    enriched = dict(result)
    render_result = render_translated_panel(panel_path, enriched)
    enriched.update(render_result)
    return enriched


class TranslateRequest(BaseModel):
    text: str


@router.get("/panels")
async def list_panels():
    return ImageService.get_all_panels()


@router.post("/upload")
async def upload_panel(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = ImageService.save_uploaded_panel(content, file.filename)
        return result
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{filename}/ocr")
async def scan_panel(filename: str):
    """OCR - Text extrahieren mit Bounding Boxes"""
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")

    try:
        # Run blocking OCR in thread pool to avoid blocking the event loop
        result = await asyncio.get_event_loop().run_in_executor(None, _run_ocr, panel_path)
    except Exception as e:
        logger.error(f"OCR failed for {filename}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"OCR processing error: {e}")

    if result:
        return result
    raise HTTPException(status_code=503, detail="No vision service available (Ollama not running)")


@router.post("/{filename}/scan-translate")
async def scan_and_translate(filename: str):
    """OCR + Übersetzung in einem Schritt - Ollama Fallback"""
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")

    try:
        # Run blocking OCR in thread pool to avoid blocking the event loop
        result = await asyncio.get_event_loop().run_in_executor(None, _run_scan_translate, panel_path)
    except Exception as e:
        logger.error(f"OCR+translate failed for {filename}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"OCR processing error: {e}")

    if result:
        return result
    raise HTTPException(status_code=503, detail="No vision service available (Ollama not running)")


@router.post("/translate")
async def translate_text(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    svc = _get_vision_service()
    if not svc:
        raise HTTPException(status_code=503, detail="No vision service available")

    result = svc.translate_text(req.text)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Translation failed"))
    return result
