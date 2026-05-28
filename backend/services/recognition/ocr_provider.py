"""Unified OCR provider adapter.

The scanner now uses MangaOCR as the single OCR engine. The adapter remains so
callers have one stable entrypoint, while OCR-specific heuristics continue to
live in the MangaOCR pipeline itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import services.recognition.mangaocr as manga_ocr_service

def run_ocr(panel_path: Path, options: Dict | None = None, regions: List[Dict] | None = None) -> Dict:
	options = options or {}
	regions = regions or []
	if not manga_ocr_service.is_available():
		raise RuntimeError("MangaOCR is not installed")
	ocr_only_options = dict(options)
	ocr_only_options["ocr_engine"] = "mangaocr"
	result = manga_ocr_service.extract_ocr(
		str(panel_path),
		options=ocr_only_options,
		regions_override=regions,
	)

	if not result or not result.get("success"):
		raise RuntimeError((result or {}).get("error", "OCR engine failed"))

	# Return the raw engine result; callers may want to post-process it.
	return result

__all__ = ['run_ocr']
