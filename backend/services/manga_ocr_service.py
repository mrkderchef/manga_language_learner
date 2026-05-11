"""
Manga OCR pipeline: comic-text-detector (detection) + manga-ocr (recognition) + translation.

This replaces the LLM-based OCR approach with a proper deep-learning OCR model
(kha-white/manga-ocr-base) that is specifically trained on Japanese manga text.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from services.text_region_detector import detect_text_regions

logger = logging.getLogger(__name__)

# Lazy-loaded singleton for the manga-ocr model
_mocr = None


def _get_mocr():
    """Load the manga-ocr model (downloads on first use, ~400 MB)."""
    global _mocr
    if _mocr is None:
        from manga_ocr import MangaOcr
        logger.info("Loading manga-ocr model (first time may download ~400 MB)…")
        _mocr = MangaOcr()
        logger.info("manga-ocr model ready.")
    return _mocr


def _translate_texts(texts: list[str], target_lang: str = "en") -> list[str]:
    """Translate a batch of Japanese texts. Uses available translation backend."""
    if not texts:
        return []

    # Try Gemini for batch translation (fast, already configured)
    try:
        from services.gemini_service import GeminiOCRService
        from config import GEMINI_API_KEY
        if GEMINI_API_KEY:
            svc = GeminiOCRService()
            if svc.is_available():
                translations = []
                for t in texts:
                    result = svc.translate_text(t, target_lang)
                    if result.get("success"):
                        translations.append(result["translated"])
                    else:
                        translations.append("")
                if any(translations):
                    return translations
    except Exception as e:
        logger.warning(f"Gemini translation failed: {e}")

    # Try Google Translate
    try:
        from services.translation_service import TranslationService
        svc = TranslationService()
        if svc.use_google_translate:
            translations = []
            for t in texts:
                result = svc.translate_text(t, source_language="ja", target_language=target_lang)
                if result.get("success"):
                    translations.append(result["translated"])
                else:
                    translations.append("")
            return translations
    except Exception as e:
        logger.warning(f"Translation service failed: {e}")

    # Fallback: MyMemory free API
    try:
        import requests
        translations = []
        for t in texts:
            try:
                resp = requests.get(
                    "https://api.mymemory.translated.net/get",
                    params={"q": t, "langpair": f"ja|{target_lang}"},
                    timeout=10,
                )
                data = resp.json()
                if data.get("responseStatus") == 200:
                    translations.append(data["responseData"]["translatedText"])
                else:
                    translations.append("")
            except Exception:
                translations.append("")
        return translations
    except Exception as e:
        logger.warning(f"Fallback translation also failed: {e}")
        return [""] * len(texts)


def extract_and_translate(image_path: str, target_lang: str = "en") -> dict:
    """
    Full pipeline: detect text regions → OCR each region → translate.

    Returns the same response format as gemini_service/ollama_service so it's
    a drop-in replacement in the scanner route.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"success": False, "error": f"Cannot read image: {image_path}",
                "text": "", "annotations": []}

    im_h, im_w = img.shape[:2]

    # Step 1: Detect text regions with comic-text-detector
    regions = detect_text_regions(image_path)
    logger.info(f"manga-ocr pipeline: {len(regions)} text regions detected")

    if not regions:
        return {
            "success": True,
            "text": "",
            "annotations": [],
            "method": "manga-ocr",
            "image_width": im_w,
            "image_height": im_h,
        }

    # Step 2: Crop each region and run manga-ocr
    mocr = _get_mocr()
    pil_img = Image.open(image_path).convert("RGB")

    recognized_texts: list[str] = []
    valid_regions: list[dict] = []

    for region in regions:
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]

        # Add small padding for better OCR
        pad = max(4, int(min(w, h) * 0.05))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(im_w, x + w + pad)
        y2 = min(im_h, y + h + pad)

        crop = pil_img.crop((x1, y1, x2, y2))

        # Run manga-ocr on the crop
        text = mocr(crop).strip()

        if text:
            recognized_texts.append(text)
            valid_regions.append(region)
            logger.debug(f"  Region ({x},{y} {w}x{h}): '{text}'")

    logger.info(f"manga-ocr recognized text in {len(recognized_texts)}/{len(regions)} regions")

    # Step 3: Translate all texts
    translations = _translate_texts(recognized_texts, target_lang)

    # Step 4: Build response
    annotations = []
    for text, translation, region in zip(recognized_texts, translations, valid_regions):
        rx, ry = int(region["x"]), int(region["y"])
        rw, rh = int(region["width"]), int(region["height"])
        bbox = [
            [rx, ry],
            [rx + rw, ry],
            [rx + rw, ry + rh],
            [rx, ry + rh],
        ]
        annotations.append({
            "text": text,
            "translated": translation,
            "confidence": 0.95,
            "bbox": bbox,
            "char_count": len(text),
        })

    full_text = "\n".join(recognized_texts)

    return {
        "success": True,
        "text": full_text,
        "annotations": annotations,
        "method": "manga-ocr",
        "image_width": im_w,
        "image_height": im_h,
    }


def is_available() -> bool:
    """Check if manga-ocr can be loaded."""
    try:
        from manga_ocr import MangaOcr  # noqa: F401
        return True
    except ImportError:
        return False
