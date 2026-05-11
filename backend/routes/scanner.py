from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from services.image_service import ImageService
from services.gemini_service import GeminiOCRService
from services.ollama_service import OllamaOCRService
from services import manga_ocr_service
import hashlib
import json
import logging
from pathlib import Path
from config import BASE_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/scanner", tags=["scanner"])

# Primary: manga-ocr (comic-text-detector + manga_ocr model)
# Fallback: Gemini Vision, then Ollama local vision model
gemini_service = GeminiOCRService()
ollama_service = OllamaOCRService()

# OCR result cache directory
OCR_CACHE_DIR = BASE_DIR / "backend" / "data" / "ocr_cache"
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_vision_service():
    """Return the first available vision service."""
    if gemini_service.is_available():
        return gemini_service
    if ollama_service.is_available():
        return ollama_service
    return None


def _cache_key(panel_path: Path) -> str:
    """Generate a cache key from file path + modification time."""
    stat = panel_path.stat()
    raw = f"{panel_path}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_cached_result(panel_path: Path):
    """Return cached OCR result if available and still valid."""
    key = _cache_key(panel_path)
    cache_file = OCR_CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
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
    """Run OCR with caching. Primary: manga-ocr, fallback: Gemini/Ollama."""
    cached = _get_cached_result(panel_path)
    if cached:
        logger.info(f"OCR cache hit for {panel_path.name}")
        return cached

    # Primary: manga-ocr (comic-text-detector + manga_ocr model)
    if manga_ocr_service.is_available():
        result = manga_ocr_service.extract_and_translate(str(panel_path))
        if result["success"]:
            _save_to_cache(panel_path, result)
            return result
        logger.warning(f"manga-ocr pipeline failed: {result.get('error')}")

    # Fallback: LLM-based vision services
    for svc in [gemini_service, ollama_service]:
        if svc.is_available():
            result = svc.extract_and_translate(str(panel_path))
            if result["success"]:
                _save_to_cache(panel_path, result)
                return result
            logger.warning(f"{type(svc).__name__} failed: {result.get('error')}")

    return None


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

    result = _run_ocr(panel_path)
    if result:
        return result
    raise HTTPException(status_code=503, detail="No vision service available (Gemini quota exhausted, Ollama not running)")


@router.post("/{filename}/scan-translate")
async def scan_and_translate(filename: str):
    """OCR + Übersetzung in einem Schritt - Gemini oder Ollama Fallback"""
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")

    result = _run_ocr(panel_path)
    if result:
        return result
    raise HTTPException(status_code=503, detail="No vision service available (Gemini quota exhausted, Ollama not running)")


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
