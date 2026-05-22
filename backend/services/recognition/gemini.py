"""
Gemini Vision-based manga OCR + Translation service.
Uses Google Gemini's vision capabilities to:
1. Identify speech bubbles in manga panels
2. Read the Japanese text
3. Translate to the target language
4. Return bounding box coordinates for each text region
"""
import base64
import importlib
import json
import logging
import os
import re
from typing import Any, Dict, List, cast

from PIL import Image

try:
	genai = cast(Any, importlib.import_module("google.genai"))
	types = cast(Any, importlib.import_module("google.genai.types"))
except Exception:
	genai = cast(Any, None)
	types = cast(Any, None)

from config import TRANSLATION_TARGET_LANGUAGE, GEMINI_API_KEY

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a manga text extraction and translation expert. 
Analyze the manga panel image and identify ALL speech bubbles and text regions.

For each text region, provide:
1. The Japanese text exactly as written
2. A natural translation to {target_lang}
3. The bounding box coordinates as [x1, y1, x2, y2] in pixels (top-left and bottom-right corners)

Rules:
- Only extract actual text in speech bubbles, thought bubbles, narration boxes, and sound effects
- Do NOT detect panel borders, character bodies, clothing, or background art as text regions
- Keep sound effects (onomatopoeia) as-is in Japanese, but provide a translation/description
- For ellipsis (…, ．．．) or single punctuation, still include them but mark the translation as "..." 
- Read vertical Japanese text correctly (top to bottom, right to left)
- Be precise with bounding boxes - they should tightly surround just the text area
- Coordinates are in pixels relative to the image dimensions

Respond with ONLY valid JSON in this exact format:
{{
  "texts": [
	{{
	  "japanese": "detected Japanese text",
	  "translation": "English translation",
	  "bbox": [x1, y1, x2, y2]
	}}
  ],
  "reading_order": "right-to-left, top-to-bottom"
}}"""


class GeminiOCRService:
	def __init__(self):
		self.client = None
		self.model_name = "gemini-2.0-flash"

		if GEMINI_API_KEY:
			try:
				self.client = genai.Client(api_key=GEMINI_API_KEY)
				logger.info(f"Gemini OCR Service initialized (model: {self.model_name})")
			except Exception as e:
				logger.error(f"Failed to initialize Gemini client: {e}")
		else:
			logger.warning("GEMINI_API_KEY not set - Gemini OCR unavailable")

	def is_available(self) -> bool:
		return self.client is not None

	def extract_and_translate(self, image_path: str, target_lang: str | None = None) -> Dict:
		"""
		Send manga panel to Gemini Vision for OCR + translation in one step.
		Returns annotations with bounding boxes, Japanese text, and translations.
		"""
		if not self.client:
			return {
				"success": False,
				"error": "Gemini API key not configured. Set GEMINI_API_KEY in .env",
				"text": "",
				"annotations": []
			}

		if target_lang is None:
			target_lang = TRANSLATION_TARGET_LANGUAGE

		response_text = ""
		try:
			# Load image and get dimensions
			img = Image.open(image_path)
			img_w, img_h = img.size

			# Read image as bytes
			with open(image_path, "rb") as f:
				image_bytes = f.read()

			# Determine mime type
			ext = os.path.splitext(image_path)[1].lower()
			mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
			mime_type = mime_map.get(ext, "image/jpeg")

			# Build the prompt with target language
			lang_names = {"en": "English", "de": "German", "fr": "French", "es": "Spanish"}
			lang_name = lang_names.get(target_lang, target_lang)
			prompt = SYSTEM_PROMPT.format(target_lang=lang_name)
			prompt += f"\n\nThe image dimensions are {img_w}x{img_h} pixels."

			# Call Gemini
			response = self.client.models.generate_content(
				model=self.model_name,
				contents=[
					types.Content(
						role="user",
						parts=[
							types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
							types.Part.from_text(text=prompt),
						],
					)
				],
				config=types.GenerateContentConfig(
					temperature=0.1,
					max_output_tokens=4096,
				),
			)

			# Parse response
			response_text = response.text.strip()
			logger.info(f"Gemini raw response length: {len(response_text)}")

			# Extract JSON from response (may be wrapped in ```json ... ```)
			json_match = re.search(r'```(?:json)?\s*(.*?)```', response_text, re.DOTALL)
			if json_match:
				json_str = json_match.group(1).strip()
			else:
				json_str = response_text

			data = json.loads(json_str)
			texts = data.get("texts", [])

			# Convert to our annotation format
			annotations = []
			for item in texts:
				jp_text = item.get("japanese", "").strip()
				translation = item.get("translation", "").strip()
				bbox_raw = item.get("bbox", [0, 0, 0, 0])

				if not jp_text:
					continue

				# Convert [x1, y1, x2, y2] to [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
				x1, y1, x2, y2 = bbox_raw
				bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

				annotations.append({
					"text": jp_text,
					"translated": translation,
					"confidence": 0.95,
					"bbox": bbox,
					"char_count": len(jp_text)
				})

			full_text = "\n".join(a["text"] for a in annotations)

			return {
				"success": True,
				"text": full_text,
				"annotations": annotations,
				"method": "gemini-vision",
				"image_width": img_w,
				"image_height": img_h,
			}

		except json.JSONDecodeError as e:
			logger.error(f"Failed to parse Gemini response as JSON: {e}\nResponse: {response_text[:500]}")
			return {
				"success": False,
				"error": f"Failed to parse Gemini response: {e}",
				"text": "",
				"annotations": []
			}
		except Exception as e:
			logger.error(f"Gemini OCR failed: {e}", exc_info=True)
			return {
				"success": False,
				"error": str(e),
				"text": "",
				"annotations": []
			}

	def get_service_status(self) -> Dict:
		return {
			"ocr_service": "gemini-vision" if self.client else "none",
			"model": self.model_name,
			"available": self.client is not None
		}

	def translate_text(self, text: str, target_lang: str | None = None) -> Dict:
		"""Translate Japanese text using Gemini."""
		if not self.client:
			return {"success": False, "error": "Gemini not configured"}

		if target_lang is None:
			target_lang = TRANSLATION_TARGET_LANGUAGE

		lang_names = {"en": "English", "de": "German", "fr": "French", "es": "Spanish"}
		lang_name = lang_names.get(target_lang, target_lang)

		try:
			response = self.client.models.generate_content(
				model=self.model_name,
				contents=f"Translate the following Japanese text to {lang_name}. Return ONLY the translation, nothing else.\n\n{text}",
				config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=1024),
			)
			return {"success": True, "translated": response.text.strip(), "source": text}
		except Exception as e:
			logger.error(f"Gemini translation failed: {e}")
			return {"success": False, "error": str(e)}

__all__ = ['GeminiOCRService']
