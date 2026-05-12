"""
Manga OCR pipeline: comic-text-detector (detection) + manga-ocr (recognition) + translation.

This replaces the LLM-based OCR approach with a proper deep-learning OCR model
(kha-white/manga-ocr-base) that is specifically trained on Japanese manga text.
"""

import logging
import re

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
    """Translate a batch of Japanese texts using Ollama with manga context."""
    if not texts:
        return []

    lang_names = {"en": "English", "de": "German", "fr": "French", "es": "Spanish"}
    lang_name = lang_names.get(target_lang, target_lang)

    # Primary: Ollama batch translation with context
    try:
        from services.ollama_service import OllamaOCRService
        svc = OllamaOCRService()
        if svc.is_available():
            translations = _ollama_batch_translate(svc, texts, lang_name)
            if translations and any(translations):
                return translations
    except Exception as e:
        logger.warning(f"Ollama batch translation failed: {e}")

    # Fallback: per-text Ollama translation
    try:
        from services.ollama_service import OllamaOCRService
        svc = OllamaOCRService()
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
        logger.warning(f"Ollama per-text translation failed: {e}")

    # Last resort: MyMemory free API
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


def _ollama_batch_translate(svc, texts: list[str], lang_name: str) -> list[str]:
    """Translate all texts in one Ollama call with manga dialogue context."""
    import json
    import re

    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt = f"""You are translating Japanese manga dialogue to {lang_name}.
These are speech bubbles from the same manga page, in reading order.
Translate naturally and contextually — use conversational tone appropriate for manga.
Do NOT translate literally. Capture the intent, emotion, and natural speech patterns.
Keep sound effects descriptive (e.g. ビリリ → *riiip*).
For short exclamations or reactions, keep them punchy.

Japanese texts:
{numbered}

Respond with ONLY a JSON array of {lang_name} translations in the same order.
Example: ["translation 1", "translation 2"]"""

    try:
        text_model = svc._find_text_model()
        response = svc._call_ollama(prompt, model=text_model)
        text = response.strip()
        # Extract JSON array from response
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            if isinstance(result, list):
                # Pad or truncate to match input length
                while len(result) < len(texts):
                    result.append("")
                return result[:len(texts)]
    except Exception as e:
        logger.warning(f"Ollama batch translate parse error: {e}")

    # Fallback: parse as numbered list
    try:
        translations = svc._parse_numbered_list(response)
        if translations:
            while len(translations) < len(texts):
                translations.append("")
            return translations[:len(texts)]
    except Exception:
        pass

    return None


_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def _cleanup_ocr_text(text: str) -> str:
    """Normalize manga-ocr output without changing the recognized wording."""
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("｜", "").replace("|", "")
    return text


def _ocr_score(text: str) -> int:
    """Score OCR candidates so preprocessing/rotation variants can compete."""
    if not text:
        return -100
    japanese_chars = len(_JAPANESE_RE.findall(text))
    bad_chars = text.count("�") + text.count("?")
    return japanese_chars * 3 + len(text) - bad_chars * 8


def _preprocess_crop_for_ocr(crop: Image.Image, upscale: int = 3) -> Image.Image:
    """
    Prepare a text crop for manga-ocr.

    Manga scans often have low contrast, noisy paper texture, and tiny kana.
    CLAHE + denoising + adaptive thresholding gives manga-ocr a cleaner,
    higher-resolution glyph image while keeping the crop as a normal RGB PIL
    image for the model API.
    """
    rgb = np.array(crop.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    scale = max(2, min(upscale, 4))
    if min(w, h) < 96:
        scale = 4
    bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    block_size = 31 if min(gray.shape[:2]) >= 120 else 21
    if block_size % 2 == 0:
        block_size += 1
    thresholded = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        9,
    )

    return Image.fromarray(cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB))


def _ocr_crop(mocr, crop: Image.Image, region: dict) -> tuple[str, dict]:
    """Run manga-ocr on preprocessed variants and keep the best result."""
    vertical = bool(region.get("vertical")) or crop.height > crop.width * 1.25
    variants: list[tuple[str, Image.Image]] = [
        ("preprocessed", _preprocess_crop_for_ocr(crop)),
    ]

    if vertical:
        # Try both directions: detector orientation and scan quirks can disagree.
        variants.append(("preprocessed_rot90_ccw", variants[0][1].rotate(90, expand=True)))
        variants.append(("preprocessed_rot90_cw", variants[0][1].rotate(-90, expand=True)))

    best_text = ""
    best_meta = {"variant": "", "vertical_candidate": vertical, "score": -100}

    for variant_name, variant_img in variants:
        try:
            text = _cleanup_ocr_text(mocr(variant_img))
        except Exception as e:
            logger.debug("manga-ocr failed for %s variant: %s", variant_name, e)
            continue

        score = _ocr_score(text)
        if score > best_meta["score"]:
            best_text = text
            best_meta = {
                "variant": variant_name,
                "vertical_candidate": vertical,
                "score": score,
            }

    # If aggressive preprocessing produced nothing useful, fall back to raw crop.
    if not best_text:
        try:
            best_text = _cleanup_ocr_text(mocr(crop))
            best_meta = {
                "variant": "raw_fallback",
                "vertical_candidate": vertical,
                "score": _ocr_score(best_text),
            }
        except Exception as e:
            logger.debug("manga-ocr raw fallback failed: %s", e)

    return best_text, best_meta


def _sort_regions_reading_order(regions: list[dict]) -> list[dict]:
    """
    Sort text blocks in manga reading order.

    comic-text-detector already does some sorting, but keeping it explicit here
    protects translation context when detector output order changes. Blocks are
    grouped into rough horizontal bands from top to bottom; inside each band,
    Japanese manga order is right to left.
    """
    if len(regions) <= 1:
        return regions

    heights = sorted(max(1, int(r.get("height", 1))) for r in regions)
    median_height = heights[len(heights) // 2]
    row_tolerance = max(24, int(median_height * 0.7))

    rows: list[list[dict]] = []
    for region in sorted(regions, key=lambda r: (r["y"], -r["x"])):
        center_y = region["y"] + region["height"] / 2
        placed = False
        for row in rows:
            row_center = sum(r["y"] + r["height"] / 2 for r in row) / len(row)
            if abs(center_y - row_center) <= row_tolerance:
                row.append(region)
                placed = True
                break
        if not placed:
            rows.append([region])

    ordered: list[dict] = []
    for row in rows:
        row.sort(key=lambda r: r["x"] + r["width"] / 2, reverse=True)
        ordered.extend(row)

    return ordered


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
    regions = _sort_regions_reading_order(detect_text_regions(image_path))
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

        # Run manga-ocr on preprocessed crop variants.
        text, ocr_meta = _ocr_crop(mocr, crop, region)

        if text:
            recognized_texts.append(text)
            region["ocr_meta"] = ocr_meta
            valid_regions.append(region)
            logger.debug(
                "  Region (%s,%s %sx%s, %s): '%s'",
                x, y, w, h, ocr_meta.get("variant"), text,
            )

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
            "vertical": bool(region.get("vertical")),
            "ocr_variant": region.get("ocr_meta", {}).get("variant", ""),
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
