from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from services.image_service import ImageService
from services.ocr_service import OCRService
from services.translation_service import TranslationService
from PIL import Image
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/scanner", tags=["scanner"])

ocr_service = OCRService()
translation_service = TranslationService()


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

    result = ocr_service.extract_text_from_image(str(panel_path))
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "OCR failed"))

    # Add image dimensions for frontend overlay positioning
    try:
        img = Image.open(str(panel_path))
        result["image_width"], result["image_height"] = img.size
    except Exception:
        pass

    return result


@router.post("/{filename}/scan-translate")
async def scan_and_translate(filename: str):
    """OCR + Übersetzung in einem Schritt - gibt Bounding Boxes mit Übersetzungen zurück"""
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")

    ocr_result = ocr_service.extract_text_from_image(str(panel_path))
    if not ocr_result["success"]:
        raise HTTPException(status_code=500, detail=ocr_result.get("error", "OCR failed"))

    # Get image dimensions
    try:
        img = Image.open(str(panel_path))
        img_w, img_h = img.size
    except Exception:
        img_w, img_h = 0, 0

    # Translate each annotation
    annotations = ocr_result.get("annotations", [])
    for ann in annotations:
        text = ann.get("text", "")
        if text.strip():
            trans = translation_service.translate_text(text)
            ann["translated"] = trans.get("translated", "") if trans.get("success") else ""
        else:
            ann["translated"] = ""

    return {
        "success": True,
        "text": ocr_result.get("text", ""),
        "annotations": annotations,
        "image_width": img_w,
        "image_height": img_h,
    }


@router.post("/translate")
async def translate_text(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    result = translation_service.translate_text(req.text)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Translation failed"))
    return result
