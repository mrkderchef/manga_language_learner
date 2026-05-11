from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
from services.image_service import ImageService
from services.ocr_service import OCRService
from services.translation_service import TranslationService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/panels", tags=["panels"])

# Initialize services
ocr_service = OCRService()
translation_service = TranslationService()


@router.get("/list")
async def list_panels():
    """Get all available manga panels"""
    result = ImageService.get_all_panels()
    return result


@router.post("/upload")
async def upload_panel(file: UploadFile = File(...)):
    """Upload a new manga panel"""
    try:
        content = await file.read()
        result = ImageService.save_uploaded_panel(content, file.filename)
        return result
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{filename}")
async def delete_panel(filename: str):
    """Delete an uploaded panel"""
    result = ImageService.delete_panel(filename)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post("/{filename}/ocr")
async def extract_text(filename: str):
    """
    Extract Japanese text from a panel using OCR
    """
    try:
        panel_path = ImageService.get_panel_by_filename(filename)
        
        if not panel_path:
            raise HTTPException(status_code=404, detail="Panel not found")
        
        # Extract text
        ocr_result = ocr_service.extract_text_from_image(str(panel_path))
        
        if not ocr_result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"OCR failed: {ocr_result.get('error', 'Unknown error')}"
            )
        
        return {
            "success": True,
            "filename": filename,
            "text": ocr_result.get("text", ""),
            "annotations": ocr_result.get("annotations", []),
            "method": "google_vision"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{filename}/extract-and-translate")
async def extract_and_translate(filename: str):
    """
    Extract text from panel and translate it to English
    """
    try:
        panel_path = ImageService.get_panel_by_filename(filename)
        
        if not panel_path:
            raise HTTPException(status_code=404, detail="Panel not found")
        
        # Extract text
        ocr_result = ocr_service.extract_text_from_image(str(panel_path))
        
        if not ocr_result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"OCR failed: {ocr_result.get('error', 'Unknown error')}"
            )
        
        # Translate annotations
        annotations = ocr_result.get("annotations", [])
        translated_annotations = translation_service.translate_annotations(annotations)
        
        # Also translate the full text
        full_text_translation = translation_service.translate_text(
            ocr_result.get("text", "")
        )
        
        return {
            "success": True,
            "filename": filename,
            "original_text": ocr_result.get("text", ""),
            "translated_text": full_text_translation.get("translated", ""),
            "annotations": translated_annotations,
            "ocr_method": "google_vision",
            "translation_method": "google_translate"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extraction and translation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/translate")
async def translate_text(text: str, target_language: str = "en"):
    """
    Translate text to target language
    """
    try:
        result = translation_service.translate_text(text, target_language=target_language)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def service_status():
    """Get status of OCR and Translation services"""
    return {
        "ocr": ocr_service.get_service_status(),
        "translation": translation_service.get_service_status()
    }
