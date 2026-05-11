from typing import List, Dict
import os
from config import GOOGLE_APPLICATION_CREDENTIALS
import logging

logger = logging.getLogger(__name__)

try:
    from google.cloud import vision
    HAS_GOOGLE_VISION = True
except ImportError:
    HAS_GOOGLE_VISION = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False


class OCRService:
    def __init__(self):
        self.use_google_vision = HAS_GOOGLE_VISION and bool(GOOGLE_APPLICATION_CREDENTIALS)
        self.easyocr_reader = None

        if self.use_google_vision:
            try:
                self.client = vision.ImageAnnotatorClient()
                logger.info("Google Vision OCR Service initialized")
            except Exception as e:
                logger.warning(f"Google Vision not available: {e}")
                self.use_google_vision = False

        if not self.use_google_vision and HAS_EASYOCR:
            try:
                self.easyocr_reader = easyocr.Reader(['ja', 'en'], gpu=False)
                logger.info("EasyOCR initialized (Japanese + English)")
            except Exception as e:
                logger.warning(f"EasyOCR init failed: {e}")

    def extract_text_from_image(self, image_path: str) -> Dict:
        if self.use_google_vision:
            return self._google_vision_ocr(image_path)
        if self.easyocr_reader:
            return self._easyocr(image_path)
        return {
            "success": False,
            "error": "No OCR engine available. Install easyocr or configure Google Vision.",
            "text": "",
            "annotations": []
        }

    def _google_vision_ocr(self, image_path: str) -> Dict:
        try:
            with open(image_path, "rb") as f:
                content = f.read()

            image = vision.Image(content=content)
            response = self.client.text_detection(image=image)
            texts = response.text_annotations

            if response.error.message:
                return {"success": False, "error": response.error.message, "text": "", "annotations": []}

            full_text = texts[0].description if texts else ""
            annotations = []
            for text in texts[1:]:
                annotations.append({
                    "text": text.description,
                    "bbox": [
                        [vertex.x, vertex.y]
                        for vertex in text.bounding_poly.vertices
                    ]
                })

            return {"success": True, "text": full_text, "annotations": annotations}
        except Exception as e:
            logger.error(f"Google Vision OCR failed: {e}")
            return {"success": False, "error": str(e), "text": "", "annotations": []}

    def _easyocr(self, image_path: str) -> Dict:
        try:
            results = self.easyocr_reader.readtext(image_path)

            full_text_parts = []
            annotations = []
            for bbox, text, confidence in results:
                full_text_parts.append(text)
                # bbox from easyocr: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                annotations.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": [[int(p[0]), int(p[1])] for p in bbox]
                })

            return {
                "success": True,
                "text": "\n".join(full_text_parts),
                "annotations": annotations,
                "method": "easyocr"
            }
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return {"success": False, "error": str(e), "text": "", "annotations": []}

    def get_service_status(self) -> Dict:
        if self.use_google_vision:
            engine = "google_vision"
        elif self.easyocr_reader:
            engine = "easyocr"
        else:
            engine = "none"
        return {"ocr_service": engine, "available": engine != "none"}
