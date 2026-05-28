"""
Manga OCR pipeline: comic-text-detector (detection) + manga-ocr (recognition).

This replaces the LLM-based OCR approach with a proper deep-learning OCR model
(kha-white/manga-ocr-base) that is specifically trained on Japanese manga text.
"""

import logging
import re
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import PANEL_DATA_DIR, ocr_panel_slug
from services.vision.bubble_allocator import candidate_box_from_region, estimate_allocation_space, reconcile_overlapping_bubble_spaces
from services.rabbithole import nlp as rabbithole_service
from services.detection.region_detector import detect_text_regions

logger = logging.getLogger(__name__)

MANGA_OCR_REPO_ID = "kha-white/manga-ocr-base"

# Lazy-loaded singleton for the manga-ocr model
_mocr = None


def _get_mocr():
	"""Load the manga-ocr model from the local Hugging Face snapshot."""
	global _mocr
	if _mocr is None:
		from huggingface_hub import snapshot_download
		from manga_ocr import MangaOcr
		try:
			model_path = snapshot_download(MANGA_OCR_REPO_ID, local_files_only=True)
		except Exception:
			logger.info("MangaOCR snapshot missing locally; downloading %s", MANGA_OCR_REPO_ID)
			model_path = snapshot_download(MANGA_OCR_REPO_ID)
		logger.info("Loading manga-ocr model from %s", model_path)
		_mocr = MangaOcr(model_path)
		logger.info("manga-ocr model ready.")
	return _mocr


_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
OCR_DEBUG_URL_PREFIX = "/api/media/ocr-debug"


def _safe_debug_component(value: str) -> str:
	value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
	return value.strip("._") or "debug"


def _cleanup_ocr_text(text: str) -> str:
	"""Normalize manga-ocr output without changing the recognized wording."""
	text = text.strip()
	text = re.sub(r"\s+", "", text)
	text = text.replace("｜", "").replace("|", "")
	return text


def _baseline_ocr_score(text: str) -> int:
	if not text:
		return -100
	japanese_chars = len(_JAPANESE_RE.findall(text))
	bad_chars = text.count("�") + text.count("?")
	return japanese_chars * 3 + len(text) - bad_chars * 8


def _variant_orientation(variant: str) -> str:
	return "horizontal" if "_rot90_" in variant else "vertical"


def _score_candidate(candidate: dict, candidates: list[dict], expected_orientation: str, options: dict | None = None) -> dict:
	"""Score an OCR candidate with visible component breakdown."""
	options = options or {}
	text = candidate.get("text") or ""
	if not text:
		return {
			"total": -100,
			"script_quality": -100,
			"layout_fit": 0,
			"variant_consensus": 0,
			"semantic_plausibility": 0,
			"orientation_bias": 0,
			"penalties": {"empty": -100},
		}

	length = len(text)
	japanese_chars = len(_JAPANESE_RE.findall(text))
	bad_chars = text.count("�") + text.count("?") + text.count("ï¿½")
	weird_chars = len(re.findall(r"[^\u3040-\u30ff\u3400-\u9fff。、！？…ー・\w]", text))
	jp_ratio = japanese_chars / max(1, length)

	script_quality = japanese_chars * 2.4 + min(length, 32) * 0.8 - bad_chars * 14 - weird_chars * 4

	orientation = _variant_orientation(candidate.get("variant", ""))
	layout_fit = 10 if orientation == expected_orientation else -12

	rotation_preference = str(options.get("vertical_preference", "normal"))
	orientation_bias = 0
	if expected_orientation == "vertical":
		orientation_bias = {"off": 0, "normal": 8, "strong": 14}.get(rotation_preference, 8)
		if orientation != "vertical":
			orientation_bias *= -1

	same_text = [
		c for c in candidates
		if c.get("text") == text and "_rot90_" not in c.get("variant", "")
	]
	variant_consensus = 0
	if "_rot90_" not in candidate.get("variant", ""):
		variant_consensus = min(len(same_text), 3) * 8
	elif len(same_text):
		variant_consensus = -8

	if str(options.get("semantic_rerank", "close")) == "off":
		semantic = {"score": 0, "enabled": False}
		semantic_plausibility = 0
	else:
		semantic = rabbithole_service.token_plausibility(text)
		semantic_plausibility = semantic["score"]

	penalties = {}
	if jp_ratio < 0.55:
		penalties["low_japanese_ratio"] = -12
	if re.search(r"(.)\1{5,}", text):
		penalties["repeated_character_run"] = -10
	if length > 48:
		penalties["overlong"] = -min(16, (length - 48) * 0.5)

	total = script_quality + layout_fit + variant_consensus + semantic_plausibility + orientation_bias + sum(penalties.values())
	return {
		"total": int(round(total)),
		"script_quality": round(script_quality, 2),
		"layout_fit": layout_fit,
		"variant_consensus": variant_consensus,
		"semantic_plausibility": semantic_plausibility,
		"semantic_details": semantic,
		"orientation_bias": orientation_bias,
		"penalties": penalties,
	}


def _ocr_score(text: str) -> int:
	"""Return the baseline text score used for OCR candidate diagnostics."""
	return _baseline_ocr_score(text)


def _ocr_debug_warnings(text: str, score: int, candidates: list[dict]) -> list[str]:
	"""Return human-readable OCR quality hints for the debug panel."""
	warnings = []
	if not text:
		return ["empty_ocr"]

	Japanese_chars = len(_JAPANESE_RE.findall(text))
	if Japanese_chars == 0:
		warnings.append("no_japanese_chars")
	elif Japanese_chars / max(1, len(text)) < 0.55:
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


def _region_box(region: dict) -> tuple[int, int, int, int]:
	x = int(region.get("x", 0))
	y = int(region.get("y", 0))
	width = max(1, int(region.get("width", 1)))
	height = max(1, int(region.get("height", 1)))
	return x, y, x + width, y + height


def _rect_area(rect: tuple[int, int, int, int]) -> int:
	return max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1])


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
	x1 = max(a[0], b[0])
	y1 = max(a[1], b[1])
	x2 = min(a[2], b[2])
	y2 = min(a[3], b[3])
	return max(0, x2 - x1) * max(0, y2 - y1)


def _xywh_box_to_rect(box: list[int] | tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
	if not isinstance(box, (list, tuple)) or len(box) < 4:
		return None
	try:
		x, y, width, height = [int(round(float(value))) for value in box[:4]]
	except Exception:
		return None
	if width <= 0 or height <= 0:
		return None
	return x, y, x + width, y + height


def _union_rect(rects: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
	if not rects:
		return None
	return (
		min(rect[0] for rect in rects),
		min(rect[1] for rect in rects),
		max(rect[2] for rect in rects),
		max(rect[3] for rect in rects),
	)


def _region_text_geometry_rect(region: dict, gray_img: np.ndarray | None = None) -> tuple[int, int, int, int]:
	line_box = candidate_box_from_region(region)
	refined_box = candidate_box_from_region(region, gray_img) if gray_img is not None else None
	rect = _union_rect([
		candidate
		for candidate in (
			_xywh_box_to_rect(line_box),
			_xywh_box_to_rect(refined_box),
		)
		if candidate is not None
	])
	return rect or _region_box(region)


def _entry_text_geometry_rect(entry: dict) -> tuple[int, int, int, int]:
	vision = ((entry.get("ocr_meta") or {}).get("vision") or {}) if isinstance(entry.get("ocr_meta"), dict) else {}
	rect = _union_rect([
		candidate
		for candidate in (
			_xywh_box_to_rect(vision.get("line_candidate_box")),
			_xywh_box_to_rect(vision.get("candidate_box")),
		)
		if candidate is not None
	])
	if rect is not None:
		return rect
	return entry.get("_text_geometry_rect") or _region_text_geometry_rect(entry["region"])


def _region_priority(region: dict, geometry_rect: tuple[int, int, int, int]) -> tuple[int, int, int]:
	return (
		max(1, _rect_area(geometry_rect)),
		len(region.get("lines") or []),
		max(1, _rect_area(_region_box(region))),
	)


def _overlaps_on_text_geometry(
	a: tuple[int, int, int, int],
	b: tuple[int, int, int, int],
	min_overlap_ratio: float = 0.5,
) -> bool:
	overlap = _rect_overlap(a, b)
	if overlap <= 0:
		return False
	smaller_area = max(1, min(_rect_area(a), _rect_area(b)))
	return (overlap / smaller_area) >= min_overlap_ratio


def _suppress_overlapping_detector_regions(regions: list[dict], gray_img: np.ndarray) -> list[dict]:
	"""Prefer the larger text-geometry region whenever detector regions overlap strongly."""
	if len(regions) <= 1:
		return regions

	prepared = []
	for region in regions:
		copy_region = dict(region)
		copy_region["_text_geometry_rect"] = _region_text_geometry_rect(copy_region, gray_img)
		prepared.append(copy_region)

	suppressed: set[int] = set()
	for index_a, region_a in enumerate(prepared):
		if index_a in suppressed:
			continue
		rect_a = region_a["_text_geometry_rect"]
		for index_b, region_b in enumerate(prepared[index_a + 1:], start=index_a + 1):
			if index_b in suppressed:
				continue
			rect_b = region_b["_text_geometry_rect"]
			if not _overlaps_on_text_geometry(rect_a, rect_b):
				continue

			priority_a = _region_priority(region_a, rect_a)
			priority_b = _region_priority(region_b, rect_b)
			if priority_a >= priority_b:
				winner_index, loser_index = index_a, index_b
			else:
				winner_index, loser_index = index_b, index_a

			winner = prepared[winner_index]
			loser = prepared[loser_index]
			winner.setdefault("_suppressed_region_ids", []).append(loser.get("region_id"))
			suppressed.add(loser_index)

	if not suppressed:
		return prepared
	return [region for index, region in enumerate(prepared) if index not in suppressed]


def _suppress_nested_regions(entries: list[dict]) -> list[dict]:
	"""Prefer the larger text-geometry owner when two OCR regions overlap strongly."""
	if len(entries) <= 1:
		return entries

	suppressed: set[int] = set()
	for index_a, entry_a in enumerate(entries):
		if index_a in suppressed:
			continue
		rect_a = _entry_text_geometry_rect(entry_a)
		for index_b, entry_b in enumerate(entries[index_a + 1:], start=index_a + 1):
			if index_b in suppressed:
				continue
			rect_b = _entry_text_geometry_rect(entry_b)
			if not _overlaps_on_text_geometry(rect_a, rect_b):
				continue

			priority_a = _region_priority(entry_a["region"], rect_a)
			priority_b = _region_priority(entry_b["region"], rect_b)
			if priority_a >= priority_b:
				winner_index, loser_index = index_a, index_b
			else:
				winner_index, loser_index = index_b, index_a

			winner = entries[winner_index]
			loser = entries[loser_index]
			winner_meta = winner.get("ocr_meta", {})
			winner_meta.setdefault("warnings", [])
			if "nested_child_box_suppressed" not in winner_meta["warnings"]:
				winner_meta["warnings"].append("nested_child_box_suppressed")
			loser.get("ocr_meta", {}).setdefault("suppressed_by_region", winner.get("region", {}).get("region_id"))
			suppressed.add(loser_index)

	if not suppressed:
		return entries
	return [entry for index, entry in enumerate(entries) if index not in suppressed]


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


def _bubble_options(options: dict | None = None) -> dict:
	options = options or {}
	return options.get("bubble") if isinstance(options.get("bubble"), dict) else options


def _estimate_bubble_geometry(gray_img: np.ndarray, region: dict, options: dict | None = None) -> dict:
	return estimate_allocation_space(gray_img, region, _bubble_options(options)).debug


def _estimate_bubble_allocation(gray_img: np.ndarray, region: dict, options: dict | None = None):
	return estimate_allocation_space(gray_img, region, _bubble_options(options))


def _preprocess_crop_variants(crop: Image.Image, options: dict | None = None) -> list[tuple[str, Image.Image]]:
	"""
	Prepare several text crop variants for manga-ocr.

	Manga scans often have low contrast, noisy paper texture, and tiny kana.
	Different pages react differently to preprocessing, so the OCR scorer gets
	raw, contrast-enhanced, and thresholded versions to compare.
	"""
	options = options or {}
	upscale = max(1, min(5, int(options.get("crop_upscale", 3) or 3)))
	preprocessing_set = str(options.get("preprocessing_set", "standard"))
	quality_mode = str(options.get("ocr_quality_mode", "balanced"))
	if quality_mode == "fast":
		preprocessing_set = "fast"
	elif quality_mode == "deep":
		preprocessing_set = "deep"
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

	variants = [("raw_upscaled", _bgr_to_pil_rgb(bgr))]
	if preprocessing_set != "fast":
		variants.append(("contrast", Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB))))
	if preprocessing_set in {"standard", "deep"}:
		variants.append(("threshold", Image.fromarray(cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB))))
	return variants


def _save_debug_preview(image: Image.Image, name: str) -> str:
	"""Save a compact OCR debug preview image and return its static URL."""
	preview = image.convert("RGB")
	preview.thumbnail((520, 520), Image.Resampling.LANCZOS)
	path = PANEL_DATA_DIR / f"{name}.png"
	path.parent.mkdir(parents=True, exist_ok=True)
	preview.save(path, optimize=True)
	relative = path.relative_to(PANEL_DATA_DIR).as_posix()
	return f"{OCR_DEBUG_URL_PREFIX}/{relative}"


def _ocr_crop(mocr, crop: Image.Image, region: dict, debug_slug: str | None = None, options: dict | None = None) -> tuple[str, dict]:
	"""Run manga-ocr on preprocessed variants and keep the best result."""
	options = options or {}
	forced_orientation = region.get("forced_orientation") or region.get("orientation_override")
	vertical = bool(region.get("vertical")) or crop.height > crop.width * 1.25
	if forced_orientation in {"vertical", "horizontal"}:
		vertical = forced_orientation == "vertical"
	expected_orientation = "vertical" if vertical else "horizontal"
	base_variants = _preprocess_crop_variants(crop, options)
	variants: list[tuple[str, Image.Image]] = list(base_variants)

	rotated_enabled = bool(options.get("enable_rotated_variants", True)) and str(options.get("ocr_quality_mode", "balanced")) != "fast"
	if vertical and rotated_enabled:
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
		"score_breakdown": {"total": -100},
		"candidates": candidates,
		"previews": previews,
		"recognized_orientation": expected_orientation,
		"orientation_source": "manual" if forced_orientation else "detector",
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

		score = _baseline_ocr_score(text)
		candidates.append({
			"variant": variant_name,
			"text": text,
			"score": score,
			"baseline_score": score,
			"legacy_score": score,
			"recognized_orientation": _variant_orientation(variant_name),
			"width": variant_img.width,
			"height": variant_img.height,
			"preview_url": preview_url,
		})

	for candidate in candidates:
		if candidate.get("error"):
			continue
		breakdown = _score_candidate(candidate, candidates, expected_orientation, options)
		candidate["score_breakdown"] = breakdown
		candidate["score"] = breakdown["total"]
		if candidate["score"] > best_meta["score"]:
			best_text = candidate.get("text", "")
			best_meta = {
				"variant": candidate.get("variant", ""),
				"vertical_candidate": vertical,
				"score": candidate["score"],
				"score_breakdown": breakdown,
				"candidates": candidates,
				"previews": previews,
				"selected_preview_url": candidate.get("preview_url"),
				"recognized_orientation": candidate.get("recognized_orientation", expected_orientation),
				"orientation_source": "manual" if forced_orientation else "auto_score",
			}

	# Guard against rotated hallucinations: rotated variants must clearly beat
	# the best unrotated candidate before they can override the detector layout.
	margin = int(options.get("rotation_win_margin", 15))
	if best_meta.get("variant") and "_rot90_" in best_meta["variant"]:
		unrotated = [
			c for c in candidates
			if c.get("text") and "_rot90_" not in c.get("variant", "") and not c.get("error")
		]
		if unrotated:
			best_unrotated = max(unrotated, key=lambda c: int(c.get("score", -100)))
			if int(best_unrotated.get("score", -100)) >= int(best_meta.get("score", -100)) - margin:
				best_text = best_unrotated.get("text", "")
				best_meta = {
					"variant": best_unrotated.get("variant", ""),
					"vertical_candidate": vertical,
					"score": int(best_unrotated.get("score", -100)),
					"score_breakdown": best_unrotated.get("score_breakdown", {}),
					"candidates": candidates,
					"previews": previews,
					"selected_preview_url": best_unrotated.get("preview_url"),
					"recognized_orientation": best_unrotated.get("recognized_orientation", expected_orientation),
					"orientation_source": "manual" if forced_orientation else "auto_score",
					"selection_note": f"unrotated preferred within rotation margin {margin}",
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
				"score_breakdown": {"total": fallback_score, "fallback": True},
				"candidates": candidates,
				"previews": previews,
				"selected_preview_url": fallback_preview_url,
				"recognized_orientation": expected_orientation,
				"orientation_source": "manual" if forced_orientation else "fallback",
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


def extract_ocr(image_path: str, options: dict | None = None, regions_override: list[dict] | None = None) -> dict:
	"""
	OCR pipeline: detect text regions, crop each region, and recognize Japanese text.

	Translation and Rabbithole analysis are intentionally handled by separate
	scanner stages.
	"""
	img = cv2.imread(image_path)
	if img is None:
		return {"success": False, "error": f"Cannot read image: {image_path}",
				"text": "", "annotations": []}

	im_h, im_w = img.shape[:2]

	options = options or {}

	# Step 1: Detect text regions with comic-text-detector, unless the caller
	# provides panel-state regions with manual overrides already applied.
	regions = _sort_regions_reading_order(regions_override or detect_text_regions(image_path, options=options))
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
	gray_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
	filtered_regions = _suppress_overlapping_detector_regions(regions, gray_img)
	if len(filtered_regions) != len(regions):
		logger.info(
			"manga-ocr geometry dedupe: kept %s/%s regions after overlap suppression",
			len(filtered_regions),
			len(regions),
		)
	regions = filtered_regions
	scan_path = Path(image_path)
	scan_stat = scan_path.stat()
	scan_slug_raw = f"{scan_path}:{scan_stat.st_size}:{scan_stat.st_mtime}"
	scan_slug = hashlib.md5(scan_slug_raw.encode("utf-8")).hexdigest()[:12]
	panel_debug_dir = f"{ocr_panel_slug(scan_path)}/ocr/debug"

	recognized_entries: list[dict] = []

	for region_index, region in enumerate(regions, start=1):
		x, y, w, h = region["x"], region["y"], region["width"], region["height"]

		# Add configurable padding for better OCR without changing the detected box.
		pad_ratio = float(options.get("crop_padding_ratio", 0.05) or 0.05)
		pad_min = int(options.get("crop_padding_min", 4) or 4)
		pad = max(pad_min, int(min(w, h) * pad_ratio))
		x1 = max(0, x - pad)
		y1 = max(0, y - pad)
		x2 = min(im_w, x + w + pad)
		y2 = min(im_h, y + h + pad)

		crop = pil_img.crop((x1, y1, x2, y2))

		# Run manga-ocr on preprocessed crop variants.
		region_id = region.get("region_id") or f"region_{region_index:03d}"
		geom_slug_raw = json.dumps({
			"x": int(region.get("x", 0)),
			"y": int(region.get("y", 0)),
			"width": int(region.get("width", 0)),
			"height": int(region.get("height", 0)),
			"forced_orientation": region.get("forced_orientation") or region.get("orientation_override") or "",
		}, sort_keys=True)
		geom_slug = hashlib.md5(geom_slug_raw.encode("utf-8")).hexdigest()[:10]
		debug_slug = f"{panel_debug_dir}/{scan_slug}_{_safe_debug_component(region_id)}_{geom_slug}"
		text, ocr_meta = _ocr_crop(mocr, crop, region, debug_slug, options)
		ocr_meta["crop_box"] = [int(x1), int(y1), int(x2), int(y2)]
		allocation = _estimate_bubble_allocation(gray_img, region, options)
		ocr_meta["vision"] = allocation.debug
		ocr_meta["_allocation_space"] = allocation

		if text:
			region["ocr_meta"] = ocr_meta
			recognized_entries.append({
				"text": text,
				"region": region,
				"ocr_meta": ocr_meta,
			})
			logger.debug(
				"  Region (%s,%s %sx%s, %s): '%s'",
				x, y, w, h, ocr_meta.get("variant"), text,
			)

	recognized_entries = _suppress_nested_regions(recognized_entries)
	if bool(_bubble_options(options).get("overlap_reconciliation", True)):
		reconcile_overlapping_bubble_spaces(recognized_entries, im_w, im_h)
	recognized_texts = [entry["text"] for entry in recognized_entries]
	valid_regions = [entry["region"] for entry in recognized_entries]

	logger.info(f"manga-ocr recognized text in {len(recognized_texts)}/{len(regions)} regions")

	# Step 3: Build OCR-only response
	annotations = []
	for idx, (text, region) in enumerate(zip(recognized_texts, valid_regions), start=1):
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
			"confidence": float(ocr_meta.get("confidence", 0.0)),
			"bbox": bbox,
			"char_count": len(text),
			"vertical": bool(region.get("vertical")),
			"ocr_variant": ocr_meta.get("variant", ""),
			"region_id": region.get("region_id") or f"region_{idx:04d}",
			"recognized_orientation": ocr_meta.get("recognized_orientation", "vertical" if region.get("vertical") else "horizontal"),
			"orientation_source": ocr_meta.get("orientation_source", "detector"),
			"reading_order": idx,
			"font_size": int(region.get("font_size") or 0),
			"angle": int(region.get("angle") or 0),
			"lines": region.get("lines", []),
			"ocr_debug": {
				"selected_variant": ocr_meta.get("variant", ""),
				"score": ocr_meta.get("score", -100),
				"quality": ocr_meta.get("quality", "bad"),
				"score_breakdown": ocr_meta.get("score_breakdown", {}),
				"selection_note": ocr_meta.get("selection_note"),
				"warnings": ocr_meta.get("warnings", []),
				"candidates": ocr_meta.get("candidates", []),
				"previews": ocr_meta.get("previews", {}),
				"selected_preview_url": ocr_meta.get("selected_preview_url"),
				"vertical_candidate": bool(ocr_meta.get("vertical_candidate")),
				"crop_box": ocr_meta.get("crop_box"),
				"detected_box": [rx, ry, rw, rh],
				"vision": ocr_meta.get("vision", {}),
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
		"ocr_engine_requested": options.get("ocr_engine", "mangaocr"),
		"ocr_engine_used": "mangaocr",
		"fallback_used": False,
		"fallback_reason": None,
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

__all__ = ['extract_ocr', 'is_available']
