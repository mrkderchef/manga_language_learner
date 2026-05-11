from typing import List, Dict
from google.cloud import vision
import os
from config import GOOGLE_APPLICATION_CREDENTIALS
import logging

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self):
        """Initialize Google Vision client"""
        self.use_google_vision = bool(GOOGLE_APPLICATION_CREDENTIALS)
        
        if self.use_google_vision:
            try:
                self.client = vision.ImageAnnotatorClient()
                logger.info("Google Vision OCR Service initialized")
            except Exception as e:
                logger.warning(f"Google Vision not available: {e}")
                self.use_google_vision = False
    
    def extract_text_from_image(self, image_path: str) -> Dict:
        """
        Extract Japanese text from image using Google Vision API
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with extracted text and metadata
        """
        if not self.use_google_vision:
            return self._fallback_ocr(image_path)
        
        try:
            with open(image_path, "rb") as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Perform text detection
            response = self.client.text_detection(image=image)
            texts = response.text_annotations
            
            if response.error.message:
                logger.error(f"OCR Error: {response.error.message}")
                return {
                    "success": False,
                    "error": response.error.message,
                    "text": "",
                    "annotations": []
                }
            
            # Extract full text and individual annotations
            full_text = texts[0].description if texts else ""
            
            # Extract individual words/characters with bounding boxes
            annotations = []
            for text in texts[1:]:  # Skip first which is the full text
                annotation = {
                    "text": text.description,
                    "confidence": text.confidence,
                    "vertices": [
                        {
                            "x": vertex.x,
                            "y": vertex.y
                        }
                        for vertex in text.bounding_poly.vertices
                    ]
                }
                annotations.append(annotation)
            
            return {
                "success": True,
                "text": full_text,
                "annotations": annotations,
                "raw_response": response
            }
        
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "annotations": []
            }
    
    def _fallback_ocr(self, image_path: str) -> Dict:
        """
        Fallback OCR using pytesseract (local)
        Requires tessdata for Japanese support
        """
        try:
            import pytesseract
            from PIL import Image
            
            img = Image.open(image_path)
            # Use Japanese language data if available
            text = pytesseract.image_to_string(img, lang='jpn')
            
            return {
                "success": True,
                "text": text,
                "annotations": [],
                "method": "pytesseract"
            }
        except Exception as e:
            logger.error(f"Fallback OCR failed: {str(e)}")
            return {
                "success": False,
                "error": f"OCR not available: {str(e)}",
                "text": "",
                "annotations": [],
                "method": "none"
            }
    
    def get_service_status(self) -> Dict:
        """Get status of OCR service"""
        return {
            "ocr_service": "google_vision" if self.use_google_vision else "not_available",
            "available": self.use_google_vision
        }
