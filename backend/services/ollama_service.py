"""
Ollama Vision-based manga OCR + Translation service.
Pipeline: Read text (vision) → Translate → Detect N regions (OpenCV) → Match
Ollama determines the text blocks, OpenCV finds where they are on the image.
"""
import base64
import json
import logging
import re
from typing import Dict, List, Optional

import requests
from PIL import Image

from config import TRANSLATION_TARGET_LANGUAGE, OLLAMA_BASE_URL, OLLAMA_MODEL
from services.text_region_detector import detect_text_regions

logger = logging.getLogger(__name__)

# --- Step 1: Read ALL text from the manga panel in one vision call ---
READ_ALL_PROMPT = """You are a Japanese manga text reader. Look at this manga panel carefully.

List ALL Japanese text you can see in speech bubbles, thought bubbles, narration boxes, and sound effects.
Read vertical Japanese text from top to bottom, right to left.

For each separate text block (speech bubble or text area), output it on its own line.
Number each text block.

Example output:
1. おはよう
2. なに？
3. ドキドキ

Return ONLY the numbered list of Japanese text. No English, no explanations."""

# --- Step 2: Translate all collected texts ---
TRANSLATE_PROMPT = """Translate each Japanese text to {target_lang}. Keep the same numbering.

{texts}

Return ONLY the translations with the same numbering. No Japanese text, no explanations.
Example:
1. Good morning
2. What?
3. *heartbeat*"""


def _lang_name(code: str) -> str:
    return {"en": "English", "de": "German", "fr": "French", "es": "Spanish"}.get(code, code)


class OllamaOCRService:
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
        self._available = None
        self._best_vision_model = None
        logger.info(f"Ollama OCR Service configured (model: {self.model}, url: {self.base_url})")

    def _find_best_vision_model(self) -> Optional[str]:
        """Find the best available vision-capable model."""
        if self._best_vision_model:
            return self._best_vision_model
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code != 200:
                return None
            models = r.json().get("models", [])
            # Preference order: larger models first, vision-capable
            vision_preferences = ["llava:7b", "minicpm-v", "llava:13b"]
            model_names = [m["name"] for m in models]
            for pref in vision_preferences:
                for name in model_names:
                    if pref in name:
                        self._best_vision_model = name
                        logger.info(f"Selected best vision model: {name}")
                        return name
            # Fallback to configured model
            base = self.model.split(":")[0]
            for name in model_names:
                if base in name:
                    self._best_vision_model = name
                    return name
        except Exception as e:
            logger.warning(f"Failed to find vision model: {e}")
        return None

    def is_available(self) -> bool:
        if self._available is True:
            return True
        model = self._find_best_vision_model()
        self._available = model is not None
        return self._available

    def _call_ollama(self, prompt: str, image_b64: Optional[str] = None,
                     model: Optional[str] = None) -> str:
        """Send a single prompt to Ollama, optionally with an image."""
        use_model = model or self._best_vision_model or self.model
        payload = {
            "model": use_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 4096},
        }
        if image_b64:
            payload["images"] = [image_b64]

        logger.info(f"  Calling Ollama model={use_model} (image={'yes' if image_b64 else 'no'})...")
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=None)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def _parse_numbered_list(self, text: str) -> List[str]:
        """Parse a numbered list response into individual items.
        Only accepts lines that start with a number (rejects preamble text)."""
        lines = text.strip().split("\n")
        items = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Only accept lines starting with a number (1. / 1) / 1:)
            m = re.match(r'^(\d+)[\.\)\:]\s*(.*)', line)
            if m and m.group(2):
                items.append(m.group(2))
        return items

    # ── Step 1: Read all text from image ─────────────────────────
    def _step_read_all(self, image_b64: str) -> List[str]:
        """Read all Japanese text from the manga panel in one call."""
        logger.info("Ollama Step 1/2: Reading all text from panel...")
        response = self._call_ollama(READ_ALL_PROMPT, image_b64)
        logger.info(f"  Read response ({len(response)} chars): {response[:500]}")
        texts = self._parse_numbered_list(response)
        logger.info(f"  Found {len(texts)} text blocks")
        return texts

    # ── Step 2: Translate all texts ──────────────────────────────
    def _step_translate(self, texts: List[str], target_lang: str) -> List[str]:
        """Translate all Japanese texts in one call (text-only, no image)."""
        if not texts:
            return []

        lang_name = _lang_name(target_lang)
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        prompt = TRANSLATE_PROMPT.format(target_lang=lang_name, texts=numbered)

        # Use a non-vision model if available (faster for text-only)
        text_model = self._find_text_model()

        logger.info(f"Ollama Step 2/2: Translating {len(texts)} texts to {lang_name}...")
        response = self._call_ollama(prompt, model=text_model)
        logger.info(f"  Translate response: {response[:500]}")

        translations = self._parse_numbered_list(response)
        # Ensure same length
        while len(translations) < len(texts):
            translations.append("—")
        return translations[:len(texts)]

    def _find_text_model(self) -> Optional[str]:
        """Find a text-only model for translation (faster than vision models)."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                # Prefer text-only models for translation
                for pref in ["qwen2.5", "llama3.1"]:
                    for name in models:
                        if pref in name:
                            return name
        except Exception:
            pass
        return None  # Will use vision model as fallback

    # ── Main pipeline ────────────────────────────────────────────
    def extract_and_translate(self, image_path: str, target_lang: str = None) -> Dict:
        if target_lang is None:
            target_lang = TRANSLATION_TARGET_LANGUAGE

        try:
            img = Image.open(image_path)
            img_w, img_h = img.size

            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")

            # Step 1: Ollama reads all text from the image
            texts = self._step_read_all(image_b64)
            if not texts:
                return {
                    "success": True,
                    "text": "",
                    "annotations": [],
                    "method": "ollama-vision",
                    "image_width": img_w,
                    "image_height": img_h,
                }

            # Step 2: Ollama translates
            translations = self._step_translate(texts, target_lang)

            # Step 3: OpenCV finds text regions – request exactly as many
            #         as Ollama found text blocks
            n_blocks = len(texts)
            regions = detect_text_regions(image_path, max_regions=n_blocks)
            logger.info(f"OpenCV returned {len(regions)} regions for {n_blocks} text blocks")

            # Step 4: Match by reading order (both sorted right→left, top→bottom)
            annotations = []
            for i, (text, translation) in enumerate(zip(texts, translations)):
                if not text:
                    continue
                region = regions[i] if i < len(regions) else None
                # Convert {x, y, width, height} to [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                # to match the frontend's expected format (Gemini-compatible)
                bbox = None
                if region:
                    rx, ry = region["x"], region["y"]
                    rw, rh = region["width"], region["height"]
                    bbox = [
                        [rx, ry],
                        [rx + rw, ry],
                        [rx + rw, ry + rh],
                        [rx, ry + rh],
                    ]
                annotations.append({
                    "text": text,
                    "translated": translation,
                    "confidence": 0.7,
                    "bbox": bbox,
                    "char_count": len(text),
                })

            full_text = "\n".join(a["text"] for a in annotations)
            return {
                "success": True,
                "text": full_text,
                "annotations": annotations,
                "method": "ollama-vision",
                "image_width": img_w,
                "image_height": img_h,
            }

        except Exception as e:
            logger.error(f"Ollama pipeline failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "text": "", "annotations": []}

    def translate_text(self, text: str, target_lang: str = None) -> Dict:
        if target_lang is None:
            target_lang = TRANSLATION_TARGET_LANGUAGE

        lang_name = _lang_name(target_lang)
        try:
            text_model = self._find_text_model()
            response = self._call_ollama(
                f"Translate the following Japanese text to {lang_name}. Return ONLY the translation, nothing else.\n\n{text}",
                model=text_model,
            )
            return {"success": True, "translated": response, "source": text}
        except Exception as e:
            logger.error(f"Ollama translation failed: {e}")
            return {"success": False, "error": str(e)}
