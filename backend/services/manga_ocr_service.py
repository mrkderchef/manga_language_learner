"""
Manga OCR pipeline: comic-text-detector (detection) + manga-ocr (recognition) + translation.

This replaces the LLM-based OCR approach with a proper deep-learning OCR model
(kha-white/manga-ocr-base) that is specifically trained on Japanese manga text.
"""

import logging
import re
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import OCR_DEBUG_DIR
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
OCR_DEBUG_URL_PREFIX = "/ocr-debug"


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


def _ocr_debug_warnings(text: str, score: int, candidates: list[dict]) -> list[str]:
    """Return human-readable OCR quality hints for the debug panel."""
    warnings = []
    if not text:
        return ["empty_ocr"]

    japanese_chars = len(_JAPANESE_RE.findall(text))
    if japanese_chars == 0:
        warnings.append("no_japanese_chars")
    elif japanese_chars / max(1, len(text)) < 0.55:
        warnings.append("low_japanese_ratio")

    if len(text) <= 2:
        warnings.append("very_short_text")
    if "?" in text or "ï¿½" in text:
        warnings.append("replacement_or_question_marks")
    if re.search(r"(.)\1{5,}", text):
        warnings.append("repeated_character_run")

    sorted_scores = sorted((int(c.get("score", -100)) for c in candidates), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] <= 3:
        warnings.append("close_ocr_variant_scores")
    if score < 8:
        warnings.append("low_ocr_score")

    return warnings


def _ocr_confidence_from_debug(text: str, score: int, warnings: list[str]) -> tuple[float, str]:
    """Map heuristic OCR debug data to a practical UI confidence bucket."""
    if not text:
        return 0.0, "bad"

    confidence = 0.55 + min(max(score, 0), 60) / 150
    confidence -= min(len(warnings), 4) * 0.08
    if {"empty_ocr", "no_japanese_chars", "low_ocr_score"}.intersection(warnings):
        confidence -= 0.18

    confidence = max(0.05, min(0.98, confidence))
    if confidence >= 0.78 and not warnings:
        return confidence, "good"
    if confidence >= 0.52:
        return confidence, "warn"
    return confidence, "bad"


def _upscale_crop(crop: Image.Image, upscale: int = 3) -> np.ndarray:
    """Return an RGB crop as resized BGR OpenCV image."""
    rgb = np.array(crop.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    scale = max(2, min(upscale, 4))
    if min(w, h) < 96:
        scale = 4
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _bgr_to_pil_rgb(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _preprocess_crop_variants(crop: Image.Image, upscale: int = 3) -> list[tuple[str, Image.Image]]:
    """
    Prepare several text crop variants for manga-ocr.

    Manga scans often have low contrast, noisy paper texture, and tiny kana.
    Different pages react differently to preprocessing, so the OCR scorer gets
    raw, contrast-enhanced, and thresholded versions to compare.
    """
    bgr = _upscale_crop(crop, upscale)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_gray = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(contrast_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    block_size = 31 if min(denoised.shape[:2]) >= 120 else 21
    if block_size % 2 == 0:
        block_size += 1
    thresholded = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        9,
    )

    return [
        ("raw_upscaled", _bgr_to_pil_rgb(bgr)),
        ("contrast", Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB))),
        ("threshold", Image.fromarray(cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB))),
    ]


def _save_debug_preview(image: Image.Image, name: str) -> str:
    """Save a compact OCR debug preview image and return its static URL."""
    preview = image.convert("RGB")
    preview.thumbnail((520, 520), Image.LANCZOS)
    path = OCR_DEBUG_DIR / f"{name}.png"
    preview.save(path, optimize=True)
    return f"{OCR_DEBUG_URL_PREFIX}/{path.name}"


def _ocr_crop(mocr, crop: Image.Image, region: dict, debug_slug: str | None = None) -> tuple[str, dict]:
    """Run manga-ocr on preprocessed variants and keep the best result."""
    vertical = bool(region.get("vertical")) or crop.height > crop.width * 1.25
    base_variants = _preprocess_crop_variants(crop)
    variants: list[tuple[str, Image.Image]] = list(base_variants)

    if vertical:
        # Try both directions: detector orientation and scan quirks can disagree.
        for variant_name, variant_img in base_variants:
            variants.append((f"{variant_name}_rot90_ccw", variant_img.rotate(90, expand=True)))
            variants.append((f"{variant_name}_rot90_cw", variant_img.rotate(-90, expand=True)))

    best_text = ""
    candidates: list[dict] = []
    previews = {}
    if debug_slug:
        previews["crop"] = _save_debug_preview(crop, f"{debug_slug}_crop")
    best_meta = {
        "variant": "",
        "vertical_candidate": vertical,
        "score": -100,
        "candidates": candidates,
        "previews": previews,
    }

    for variant_name, variant_img in variants:
        preview_url = None
        if debug_slug:
            preview_url = _save_debug_preview(variant_img, f"{debug_slug}_{variant_name}")
        try:
            text = _cleanup_ocr_text(mocr(variant_img))
        except Exception as e:
            logger.debug("manga-ocr failed for %s variant: %s", variant_name, e)
            candidates.append({
                "variant": variant_name,
                "text": "",
                "score": -100,
                "error": str(e),
                "preview_url": preview_url,
            })
            continue

        score = _ocr_score(text)
        candidates.append({
            "variant": variant_name,
            "text": text,
            "score": score,
            "width": variant_img.width,
            "height": variant_img.height,
            "preview_url": preview_url,
        })
        if score > best_meta["score"]:
            best_text = text
            best_meta = {
                "variant": variant_name,
                "vertical_candidate": vertical,
                "score": score,
                "candidates": candidates,
                "previews": previews,
                "selected_preview_url": preview_url,
            }

    # If aggressive preprocessing produced nothing useful, fall back to raw crop.
    if not best_text:
        try:
            best_text = _cleanup_ocr_text(mocr(crop))
            fallback_score = _ocr_score(best_text)
            fallback_preview_url = previews.get("crop")
            candidates.append({
                "variant": "raw_fallback",
                "text": best_text,
                "score": fallback_score,
                "width": crop.width,
                "height": crop.height,
                "preview_url": fallback_preview_url,
            })
            best_meta = {
                "variant": "raw_fallback",
                "vertical_candidate": vertical,
                "score": fallback_score,
                "candidates": candidates,
                "previews": previews,
                "selected_preview_url": fallback_preview_url,
            }
        except Exception as e:
            logger.debug("manga-ocr raw fallback failed: %s", e)
            candidates.append({
                "variant": "raw_fallback",
                "text": "",
                "score": -100,
                "error": str(e),
                "preview_url": previews.get("crop"),
            })

    best_meta["warnings"] = _ocr_debug_warnings(best_text, int(best_meta.get("score", -100)), candidates)
    confidence, quality = _ocr_confidence_from_debug(
        best_text,
        int(best_meta.get("score", -100)),
        best_meta["warnings"],
    )
    best_meta["confidence"] = confidence
    best_meta["quality"] = quality
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
    scan_path = Path(image_path)
    scan_stat = scan_path.stat()
    scan_slug_raw = f"{scan_path}:{scan_stat.st_size}:{scan_stat.st_mtime}"
    scan_slug = hashlib.md5(scan_slug_raw.encode("utf-8")).hexdigest()[:12]

    recognized_texts: list[str] = []
    valid_regions: list[dict] = []

    for region_index, region in enumerate(regions, start=1):
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]

        # Add small padding for better OCR
        pad = max(4, int(min(w, h) * 0.05))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(im_w, x + w + pad)
        y2 = min(im_h, y + h + pad)

        crop = pil_img.crop((x1, y1, x2, y2))

        # Run manga-ocr on preprocessed crop variants.
        debug_slug = f"{scan_slug}_r{region_index:03d}"
        text, ocr_meta = _ocr_crop(mocr, crop, region, debug_slug)
        ocr_meta["crop_box"] = [int(x1), int(y1), int(x2), int(y2)]

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
    for idx, (text, translation, region) in enumerate(zip(recognized_texts, translations, valid_regions), start=1):
        rx, ry = int(region["x"]), int(region["y"])
        rw, rh = int(region["width"]), int(region["height"])
        bbox = [
            [rx, ry],
            [rx + rw, ry],
            [rx + rw, ry + rh],
            [rx, ry + rh],
        ]
        ocr_meta = region.get("ocr_meta", {})
        annotations.append({
            "id": f"ann_{idx:04d}",
            "text": text,
            "translated": translation,
            "confidence": float(ocr_meta.get("confidence", 0.0)),
            "bbox": bbox,
            "char_count": len(text),
            "vertical": bool(region.get("vertical")),
            "ocr_variant": ocr_meta.get("variant", ""),
            "reading_order": idx,
            "font_size": int(region.get("font_size") or 0),
            "angle": int(region.get("angle") or 0),
            "lines": region.get("lines", []),
            "ocr_debug": {
                "selected_variant": ocr_meta.get("variant", ""),
                "score": ocr_meta.get("score", -100),
                "quality": ocr_meta.get("quality", "bad"),
                "warnings": ocr_meta.get("warnings", []),
                "candidates": ocr_meta.get("candidates", []),
                "previews": ocr_meta.get("previews", {}),
                "selected_preview_url": ocr_meta.get("selected_preview_url"),
                "vertical_candidate": bool(ocr_meta.get("vertical_candidate")),
                "crop_box": ocr_meta.get("crop_box"),
                "detected_box": [rx, ry, rw, rh],
                "detector": {
                    "vertical": bool(region.get("vertical")),
                    "font_size": int(region.get("font_size") or 0),
                    "angle": int(region.get("angle") or 0),
                    "lines": len(region.get("lines", []) or []),
                },
            },
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
