"""Render translated text back into manga panels.

This is a small native renderer inspired by manga translation pipelines:
mask original text, clean it, fit translated text, and save a translated
image. It intentionally does not vendor code from external projects.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import RENDERED_PANELS_DIR

logger = logging.getLogger(__name__)

RENDERED_URL_PREFIX = "/rendered-panels"
RENDER_METHOD = "opencv-pillow-v4"


@dataclass
class LayoutSpace:
    ann: dict[str, Any]
    text: str
    min_rect: tuple[int, int, int, int]
    max_rect: tuple[int, int, int, int]
    mask: np.ndarray
    mask_origin: tuple[int, int]
    anchor: tuple[float, float]
    centroid: tuple[float, float]
    initial_size: int


@dataclass
class TextPlacement:
    text: str
    rect: tuple[int, int, int, int]
    font: ImageFont.ImageFont
    lines: list[str]
    spacing: int
    stroke_width: int
    angle: float
    fill: tuple[int, int, int, int]
    stroke: tuple[int, int, int, int]
    score: float
    overflow: bool
    alpha_rect: tuple[int, int, int, int] | None = None
    alpha_mask: np.ndarray | None = None


def render_translated_panel(panel_path: Path, scan_result: dict[str, Any]) -> dict[str, Any]:
    """Create or reuse a translated panel image for a scan result."""
    annotations = [
        ann for ann in scan_result.get("annotations", [])
        if ann.get("translated") and ann.get("bbox")
    ]
    warnings: list[str] = []

    if not annotations:
        return {
            "translated_image_url": None,
            "render_method": RENDER_METHOD,
            "render_warnings": ["No translated annotations to render."],
        }

    cache_name = _render_cache_name(panel_path, annotations)
    output_path = RENDERED_PANELS_DIR / cache_name
    if output_path.exists():
        return {
            "translated_image_url": f"{RENDERED_URL_PREFIX}/{output_path.name}",
            "render_method": RENDER_METHOD,
            "render_warnings": [],
        }

    bgr = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return {
            "translated_image_url": None,
            "render_method": RENDER_METHOD,
            "render_warnings": [f"Could not read panel image: {panel_path.name}"],
        }

    try:
        text_mask = _build_text_mask(bgr.shape[:2], annotations)
        cleaned = _clean_text_regions(bgr, text_mask)
        rendered_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
        rendered = Image.fromarray(rendered_rgb).convert("RGBA")

        font_path = _select_font_path(warnings)
        image_h, image_w = bgr.shape[:2]

        placements = _place_translations(rendered, bgr, annotations, font_path, warnings)
        for placement in placements:
            _paint_translation(rendered, placement, image_w, image_h)

        rendered.convert("RGB").save(output_path, quality=92, optimize=True)
        return {
            "translated_image_url": f"{RENDERED_URL_PREFIX}/{output_path.name}",
            "render_method": RENDER_METHOD,
            "render_warnings": warnings,
        }
    except Exception as exc:
        logger.error("Translated panel render failed: %s", exc, exc_info=True)
        return {
            "translated_image_url": None,
            "render_method": RENDER_METHOD,
            "render_warnings": [f"Translated image rendering failed: {exc}"],
        }


def _render_cache_name(panel_path: Path, annotations: list[dict[str, Any]]) -> str:
    stat = panel_path.stat()
    render_payload = {
        "path": str(panel_path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "annotations": [
            {
                "text": ann.get("text", ""),
                "translated": ann.get("translated", ""),
                "bbox": ann.get("bbox"),
                "lines": ann.get("lines", []),
                "font_size": ann.get("font_size", 0),
                "angle": ann.get("angle", 0),
                "vertical": ann.get("vertical", False),
            }
            for ann in annotations
        ],
        "renderer": RENDER_METHOD,
    }
    raw = json.dumps(render_payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return f"{panel_path.stem}_{digest}.png"


def _build_text_mask(shape: tuple[int, int], annotations: list[dict[str, Any]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for ann in annotations:
        filled = False
        for line in ann.get("lines") or []:
            pts = _safe_points(line, w, h)
            if pts is not None:
                cv2.fillConvexPoly(mask, pts, 255)
                filled = True

        if not filled:
            x1, y1, x2, y2 = _bbox_xyxy(ann.get("bbox"), w, h)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask[mask > 0] = 255
    return mask


def _clean_text_regions(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return bgr.copy()
    try:
        return cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    except Exception:
        cleaned = bgr.copy()
        cleaned[mask > 0] = np.array([255, 255, 255], dtype=np.uint8)
        return cleaned


def _place_translations(
    image: Image.Image,
    bgr: np.ndarray,
    annotations: list[dict[str, Any]],
    font_path: str | None,
    warnings: list[str],
) -> list[TextPlacement]:
    """Choose all text positions before drawing, so neighboring text can avoid collisions."""
    image_h, image_w = bgr.shape[:2]
    spaces: list[LayoutSpace] = []
    for ann in annotations:
        try:
            space = _build_layout_space(bgr, ann, image_w, image_h)
            if space.text:
                spaces.append(space)
        except Exception as exc:
            logger.warning("Failed to prepare annotation layout: %s", exc)
            warnings.append(f"Skipped one text region: {exc}")

    # Place the hardest regions first: long translations with little usable area.
    ordered = sorted(
        spaces,
        key=lambda space: (
            _rect_area(space.max_rect) / max(1, len(space.text)),
            _rect_area(space.max_rect),
        ),
    )

    selected: list[TextPlacement] = []
    for space in ordered:
        candidates = _generate_text_candidates(image, space, font_path, image_w, image_h)
        if not candidates:
            warnings.append(f"Could not place translation: {space.text[:32]}")
            continue

        selected_rects = [placement.rect for placement in selected]
        # Rectangle overlap is cheap and good enough to shortlist. The final
        # choice uses the rendered glyph mask so loose text boxes can overlap
        # when the actual letters do not.
        shortlisted = sorted(
            candidates,
            key=lambda candidate: candidate.score + _collision_penalty(candidate.rect, selected_rects) * 0.25,
        )[:10]
        best = min(
            shortlisted,
            key=lambda candidate: (
                candidate.score
                + _collision_penalty(candidate.rect, selected_rects) * 0.35
                + _glyph_collision_penalty(candidate, selected)
            ),
        )
        _ensure_alpha_mask(best)
        if best.overflow:
            warnings.append(f"Long translation was tightly fitted: {best.text[:32]}")
        selected.append(best)

    return selected


def _build_layout_space(
    bgr: np.ndarray,
    ann: dict[str, Any],
    image_w: int,
    image_h: int,
) -> LayoutSpace:
    text = str(ann.get("translated", "")).strip()
    bbox = _bbox_xyxy(ann.get("bbox"), image_w, image_h)
    vertical = bool(ann.get("vertical"))
    min_pad = max(3, int(min(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.08))
    min_rect = _pad_rect(bbox, image_w, image_h, min_pad)

    bubble = _detect_light_bubble_space(bgr, bbox, vertical)
    if bubble is None:
        max_rect = _expanded_rect(
            bbox,
            image_w,
            image_h,
            vertical,
            wide_factor=3.4 if vertical else 2.3,
            tall_factor=2.15,
        )
        mask = np.full((max_rect[3] - max_rect[1], max_rect[2] - max_rect[0]), 255, dtype=np.uint8)
        origin = (max_rect[0], max_rect[1])
    else:
        max_rect, mask, origin = bubble

    if not _rect_contains(max_rect, min_rect):
        max_rect = _union_rect(max_rect, min_rect, image_w, image_h)
        mask = np.full((max_rect[3] - max_rect[1], max_rect[2] - max_rect[0]), 255, dtype=np.uint8)
        origin = (max_rect[0], max_rect[1])

    centroid = _mask_centroid(mask, origin) or _rect_center(max_rect)
    anchor = _rect_center(bbox)
    initial_size = _layout_initial_font_size(ann, bbox)
    return LayoutSpace(
        ann=ann,
        text=text,
        min_rect=min_rect,
        max_rect=max_rect,
        mask=mask,
        mask_origin=origin,
        anchor=anchor,
        centroid=centroid,
        initial_size=initial_size,
    )


def _detect_light_bubble_space(
    bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    vertical: bool,
) -> tuple[tuple[int, int, int, int], np.ndarray, tuple[int, int]] | None:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    search = _expanded_rect(bbox, w, h, vertical, wide_factor=4.1, tall_factor=3.1)
    sx1, sy1, sx2, sy2 = search
    crop = bgr[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    light = cv2.inRange(gray, 184, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, kernel, iterations=2)
    light = cv2.morphologyEx(light, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    local_bbox = (x1 - sx1, y1 - sy1, x2 - sx1, y2 - sy1)
    min_area = max(80, (x2 - x1) * (y2 - y1) * 1.25)
    crop_area = crop.shape[0] * crop.shape[1]
    best_contour: np.ndarray | None = None
    best_score = 0.0

    for contour in contours:
        rx, ry, rw, rh = cv2.boundingRect(contour)
        rect = (rx, ry, rx + rw, ry + rh)
        rect_area = rw * rh
        if rect_area < min_area or rect_area > crop_area * 0.992:
            continue

        component = np.zeros(light.shape, dtype=np.uint8)
        cv2.drawContours(component, [contour], -1, 255, -1)
        lx1, ly1, lx2, ly2 = local_bbox
        lx1 = max(0, lx1)
        ly1 = max(0, ly1)
        lx2 = min(component.shape[1], lx2)
        ly2 = min(component.shape[0], ly2)
        if lx2 <= lx1 or ly2 <= ly1:
            continue
        overlap = int(cv2.countNonZero(component[ly1:ly2, lx1:lx2]))
        if overlap <= 0:
            overlap = _rect_overlap(rect, local_bbox)
        if overlap <= 0:
            continue

        component_area = max(1, int(cv2.contourArea(contour)))
        score = overlap * 4.0 + min(component_area, rect_area) * 0.04
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return None

    rx, ry, rw, rh = cv2.boundingRect(best_contour)
    full_mask = np.zeros(light.shape, dtype=np.uint8)
    cv2.drawContours(full_mask, [best_contour], -1, 255, -1)
    margin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inset_mask = cv2.erode(full_mask, margin_kernel, iterations=1)
    if cv2.countNonZero(inset_mask) > min_area * 0.6:
        full_mask = inset_mask

    local_mask = full_mask[ry:ry + rh, rx:rx + rw]
    max_rect = (sx1 + rx, sy1 + ry, sx1 + rx + rw, sy1 + ry + rh)
    return max_rect, local_mask, (max_rect[0], max_rect[1])


def _generate_text_candidates(
    image: Image.Image,
    space: LayoutSpace,
    font_path: str | None,
    image_w: int,
    image_h: int,
) -> list[TextPlacement]:
    max_x1, max_y1, max_x2, max_y2 = space.max_rect
    max_w = max(1, max_x2 - max_x1)
    max_h = max(1, max_y2 - max_y1)
    min_w = min(max_w, max(1, space.min_rect[2] - space.min_rect[0]))
    min_h = min(max_h, max(1, space.min_rect[3] - space.min_rect[1]))
    candidates: list[TextPlacement] = []

    width_ratios = [0.96, 0.84, 0.72, 0.60]
    height_ratios = [0.92, 0.78, 0.64]
    centers = _candidate_centers(space)
    angle = _render_angle(space.ann)

    for width_ratio in width_ratios:
        for height_ratio in height_ratios:
            fit_w = max(min_w, int(max_w * width_ratio))
            fit_h = max(min_h, int(max_h * height_ratio))
            font, lines, spacing, stroke_width, overflow = _fit_text(
                space.text,
                fit_w,
                fit_h,
                space.initial_size,
                font_path,
            )
            text_w, text_h = _measure_text_block(lines, font, spacing, stroke_width)
            pad = max(6, stroke_width * 4)
            rect_w = min(max_w, max(min_w, text_w + pad))
            rect_h = min(max_h, max(min_h, text_h + pad))

            for cx, cy in centers:
                rect = _rect_around_center(cx, cy, rect_w, rect_h, space.max_rect)
                coverage = _mask_coverage(space, rect)
                min_overlap = _rect_overlap(rect, space.min_rect) / max(1, _rect_area(space.min_rect))
                dist = _normalized_distance(_rect_center(rect), space.anchor, max_w, max_h)
                font_size = int(getattr(font, "size", 10))
                size_loss = max(0, space.initial_size - font_size) / max(1, space.initial_size)
                outside_penalty = (1.0 - coverage) * 3600.0
                anchor_penalty = dist * 420.0
                min_penalty = max(0.0, 0.28 - min_overlap) * 1300.0
                overflow_penalty = 850.0 if overflow else 0.0
                size_penalty = size_loss * 260.0
                area_penalty = _rect_area(rect) / max(1, image_w * image_h) * 35.0
                score = (
                    outside_penalty
                    + anchor_penalty
                    + min_penalty
                    + overflow_penalty
                    + size_penalty
                    + area_penalty
                )
                fill, stroke = _text_colors_for_region(image, rect)
                candidates.append(TextPlacement(
                    text=space.text,
                    rect=rect,
                    font=font,
                    lines=lines,
                    spacing=spacing,
                    stroke_width=stroke_width,
                    angle=angle,
                    fill=fill,
                    stroke=stroke,
                    score=score,
                    overflow=overflow,
                ))

    return candidates


def _candidate_centers(space: LayoutSpace) -> list[tuple[float, float]]:
    max_x1, max_y1, max_x2, max_y2 = space.max_rect
    max_w = max_x2 - max_x1
    max_h = max_y2 - max_y1
    bases = [
        space.anchor,
        space.centroid,
        _rect_center(space.min_rect),
        _rect_center(space.max_rect),
    ]
    offsets = [
        (0.0, 0.0),
        (-0.12, 0.0),
        (0.12, 0.0),
        (0.0, -0.12),
        (0.0, 0.12),
        (-0.20, -0.08),
        (0.20, -0.08),
        (-0.20, 0.08),
        (0.20, 0.08),
        (-0.28, 0.0),
        (0.28, 0.0),
    ]

    centers: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for base_x, base_y in bases:
        for ox, oy in offsets:
            cx = min(max_x2, max(max_x1, base_x + ox * max_w))
            cy = min(max_y2, max(max_y1, base_y + oy * max_h))
            key = (int(round(cx)), int(round(cy)))
            if key not in seen:
                seen.add(key)
                centers.append((cx, cy))
    return centers


def _paint_translation(
    image: Image.Image,
    placement: TextPlacement,
    image_w: int,
    image_h: int,
) -> None:
    x1, y1, x2, y2 = placement.rect
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    layer = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.multiline_text(
        (box_w / 2, box_h / 2),
        "\n".join(placement.lines),
        font=placement.font,
        fill=placement.fill,
        anchor="mm",
        align="center",
        spacing=placement.spacing,
        stroke_width=placement.stroke_width,
        stroke_fill=placement.stroke,
    )

    paste_x, paste_y = x1, y1
    if 3 < abs(placement.angle) < 32:
        layer = layer.rotate(-placement.angle, expand=True, resample=Image.Resampling.BICUBIC)
        cx, cy = _rect_center(placement.rect)
        paste_x = int(cx - layer.width / 2)
        paste_y = int(cy - layer.height / 2)

    _composite_layer(image, layer, paste_x, paste_y, image_w, image_h)


def _choose_writing_rect(bgr: np.ndarray, ann: dict[str, Any]) -> tuple[int, int, int, int]:
    h, w = bgr.shape[:2]
    bbox = _bbox_xyxy(ann.get("bbox"), w, h)
    bubble_rect = _detect_light_bubble_rect(bgr, bbox, bool(ann.get("vertical")))
    if bubble_rect is not None:
        return _inset_rect(bubble_rect, w, h, 0.08)
    return _expanded_rect(bbox, w, h, bool(ann.get("vertical")))


def _detect_light_bubble_rect(
    bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    vertical: bool,
) -> tuple[int, int, int, int] | None:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    search = _expanded_rect(bbox, w, h, vertical, wide_factor=3.2, tall_factor=2.4)
    sx1, sy1, sx2, sy2 = search
    crop = bgr[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    light = cv2.inRange(gray, 188, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, kernel, iterations=1)
    light = cv2.morphologyEx(light, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    local_bbox = (x1 - sx1, y1 - sy1, x2 - sx1, y2 - sy1)
    min_area = max(64, (x2 - x1) * (y2 - y1) * 1.2)
    crop_area = crop.shape[0] * crop.shape[1]
    best: tuple[int, int, int, int] | None = None
    best_score = 0

    for contour in contours:
        rx, ry, rw, rh = cv2.boundingRect(contour)
        area = rw * rh
        if area < min_area or area > crop_area * 0.95:
            continue
        overlap = _rect_overlap((rx, ry, rx + rw, ry + rh), local_bbox)
        if overlap <= 0:
            continue
        score = overlap + int(area * 0.08)
        if score > best_score:
            best_score = score
            best = (sx1 + rx, sy1 + ry, sx1 + rx + rw, sy1 + ry + rh)

    return best


def _draw_translation(
    image: Image.Image,
    ann: dict[str, Any],
    rect: tuple[int, int, int, int],
    font_path: str | None,
    image_w: int,
    image_h: int,
    warnings: list[str],
) -> None:
    text = str(ann.get("translated", "")).strip()
    if not text:
        return

    x1, y1, x2, y2 = rect
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    bg_brightness = _region_brightness(image, rect)
    if bg_brightness < 120:
        fill = (245, 245, 245, 255)
        stroke = (20, 20, 20, 210)
    else:
        fill = (18, 18, 18, 255)
        stroke = (255, 255, 255, 230)

    font, lines, spacing, stroke_width, overflow = _fit_text(
        text,
        box_w,
        box_h,
        _initial_font_size(ann, box_w, box_h),
        font_path,
    )
    if overflow:
        warnings.append(f"Long translation was tightly fitted: {text[:32]}")

    layer = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.multiline_text(
        (box_w / 2, box_h / 2),
        "\n".join(lines),
        font=font,
        fill=fill,
        anchor="mm",
        align="center",
        spacing=spacing,
        stroke_width=stroke_width,
        stroke_fill=stroke,
    )

    angle = float(ann.get("angle") or 0)
    if bool(ann.get("vertical")):
        angle = 0
    if 3 < abs(angle) < 45:
        layer = layer.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)
        paste_x = int((x1 + x2) / 2 - layer.width / 2)
        paste_y = int((y1 + y2) / 2 - layer.height / 2)
    else:
        paste_x, paste_y = x1, y1

    _composite_layer(image, layer, paste_x, paste_y, image_w, image_h)


def _fit_text(
    text: str,
    box_w: int,
    box_h: int,
    initial_size: int,
    font_path: str | None,
) -> tuple[ImageFont.ImageFont, list[str], int, int, bool]:
    max_width = max(1, int(box_w * 0.88))
    max_height = max(1, int(box_h * 0.86))
    min_size = 9
    start = max(min_size, min(initial_size, 72))
    measure = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(measure)

    last: tuple[ImageFont.ImageFont, list[str], int, int] | None = None
    for size in range(start, min_size - 1, -1):
        font = _load_font(font_path, size)
        spacing = max(1, int(size * 0.12))
        stroke_width = max(1, int(size * 0.08))
        lines = _wrap_text(text, font, max_width)
        text_bbox = draw.multiline_textbbox(
            (0, 0),
            "\n".join(lines),
            font=font,
            spacing=spacing,
            stroke_width=stroke_width,
        )
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        last = (font, lines, spacing, stroke_width)
        if text_w <= max_width and text_h <= max_height:
            return font, lines, spacing, stroke_width, False

    assert last is not None
    return (*last, True)


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    text = " ".join(text.replace("\n", " ").split())
    if not text:
        return [""]

    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if _text_width(font, candidate) <= max_width or not current:
            if _text_width(font, candidate) <= max_width:
                current = candidate
                continue
        if current:
            lines.append(current)
        if _text_width(font, word) <= max_width:
            current = word
        else:
            chunks = _break_long_word(word, font, max_width)
            lines.extend(chunks[:-1])
            current = chunks[-1]

    if current:
        lines.append(current)
    return lines or [text]


def _break_long_word(word: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for char in word:
        candidate = current + char
        if _text_width(font, candidate) <= max_width or not current:
            current = candidate
        else:
            chunks.append(current)
            current = char
    if current:
        chunks.append(current)
    return chunks or [word]


def _text_width(font: ImageFont.ImageFont, text: str) -> int:
    if hasattr(font, "getlength"):
        return int(font.getlength(text))
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


def _initial_font_size(ann: dict[str, Any], box_w: int, box_h: int) -> int:
    detected = int(ann.get("font_size") or 0)
    if detected > 0:
        return max(11, min(int(detected * 0.9), 72))
    return max(11, int(min(box_h * 0.38, box_w * 0.16)))


def _layout_initial_font_size(
    ann: dict[str, Any],
    bbox: tuple[int, int, int, int],
) -> int:
    detected = int(ann.get("font_size") or 0)
    if detected > 0:
        return max(11, min(int(detected * 0.95), 64))

    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    char_count = int(ann.get("char_count") or len(str(ann.get("text", ""))) or 1)
    glyph_area = (bw * bh) / max(1, char_count)
    char_sized = glyph_area ** 0.5
    if bool(ann.get("vertical")):
        size = char_sized * 0.82
    else:
        size = min(bh * 0.86, char_sized * 1.08)
    return max(11, min(int(size), 58))


def _select_font_path(warnings: list[str]) -> str | None:
    env_path = os.getenv("RENDER_FONT_PATH", "")
    candidates = [
        env_path,
        "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    warnings.append("No TrueType render font found; using Pillow default font.")
    return None


def _load_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _composite_layer(
    image: Image.Image,
    layer: Image.Image,
    paste_x: int,
    paste_y: int,
    image_w: int,
    image_h: int,
) -> None:
    src_x1 = max(0, -paste_x)
    src_y1 = max(0, -paste_y)
    src_x2 = min(layer.width, image_w - paste_x)
    src_y2 = min(layer.height, image_h - paste_y)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return

    dst_x = max(0, paste_x)
    dst_y = max(0, paste_y)
    image.alpha_composite(layer.crop((src_x1, src_y1, src_x2, src_y2)), (dst_x, dst_y))


def _safe_points(points: Any, image_w: int, image_h: int) -> np.ndarray | None:
    try:
        arr = np.array(points, dtype=np.int32)
        if arr.shape != (4, 2):
            return None
        arr[:, 0] = np.clip(arr[:, 0], 0, image_w - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, image_h - 1)
        return arr
    except Exception:
        return None


def _bbox_xyxy(bbox: Any, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    points = np.array(bbox, dtype=np.int32).reshape(-1, 2)
    x1 = int(np.clip(points[:, 0].min(), 0, image_w - 1))
    y1 = int(np.clip(points[:, 1].min(), 0, image_h - 1))
    x2 = int(np.clip(points[:, 0].max(), x1 + 1, image_w))
    y2 = int(np.clip(points[:, 1].max(), y1 + 1, image_h))
    return x1, y1, x2, y2


def _expanded_rect(
    bbox: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    vertical: bool,
    wide_factor: float = 2.8,
    tall_factor: float = 1.55,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if vertical:
        target_w = max(bw * wide_factor, bh * 0.82, 96)
        target_h = max(bh * 1.22, 54)
    else:
        target_w = max(bw * 1.55, 96)
        target_h = max(bh * tall_factor, 44)
    nx1 = int(max(0, cx - target_w / 2))
    ny1 = int(max(0, cy - target_h / 2))
    nx2 = int(min(image_w, cx + target_w / 2))
    ny2 = int(min(image_h, cy + target_h / 2))
    return nx1, ny1, max(nx1 + 1, nx2), max(ny1 + 1, ny2)


def _inset_rect(
    rect: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    pad_x = int((x2 - x1) * ratio)
    pad_y = int((y2 - y1) * ratio)
    return (
        max(0, x1 + pad_x),
        max(0, y1 + pad_y),
        max(x1 + pad_x + 1, min(image_w, x2 - pad_x)),
        max(y1 + pad_y + 1, min(image_h, y2 - pad_y)),
    )


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    return max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1])


def _rect_center(rect: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)


def _rect_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _union_rect(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, min(a[0], b[0])),
        max(0, min(a[1], b[1])),
        min(image_w, max(a[2], b[2])),
        min(image_h, max(a[3], b[3])),
    )


def _pad_rect(
    rect: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    pad: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, rect[0] - pad),
        max(0, rect[1] - pad),
        min(image_w, rect[2] + pad),
        min(image_h, rect[3] + pad),
    )


def _rect_around_center(
    cx: float,
    cy: float,
    width: int,
    height: int,
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    bx1, by1, bx2, by2 = bounds
    width = min(max(1, width), max(1, bx2 - bx1))
    height = min(max(1, height), max(1, by2 - by1))
    x1 = int(round(cx - width / 2))
    y1 = int(round(cy - height / 2))
    x1 = max(bx1, min(x1, bx2 - width))
    y1 = max(by1, min(y1, by2 - height))
    return x1, y1, x1 + width, y1 + height


def _mask_coverage(space: LayoutSpace, rect: tuple[int, int, int, int]) -> float:
    ox, oy = space.mask_origin
    lx1 = max(0, rect[0] - ox)
    ly1 = max(0, rect[1] - oy)
    lx2 = min(space.mask.shape[1], rect[2] - ox)
    ly2 = min(space.mask.shape[0], rect[3] - oy)
    if lx2 <= lx1 or ly2 <= ly1:
        return 0.0
    covered = int(cv2.countNonZero(space.mask[ly1:ly2, lx1:lx2]))
    return min(1.0, covered / max(1, _rect_area(rect)))


def _mask_centroid(mask: np.ndarray, origin: tuple[int, int]) -> tuple[float, float] | None:
    moments = cv2.moments(mask)
    if not moments["m00"]:
        return None
    return (
        origin[0] + moments["m10"] / moments["m00"],
        origin[1] + moments["m01"] / moments["m00"],
    )


def _normalized_distance(
    a: tuple[float, float],
    b: tuple[float, float],
    width: int,
    height: int,
) -> float:
    diag = max(1.0, float((width ** 2 + height ** 2) ** 0.5))
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 / diag)


def _collision_penalty(
    rect: tuple[int, int, int, int],
    selected: list[tuple[int, int, int, int]],
) -> float:
    penalty = 0.0
    for other in selected:
        overlap = _rect_overlap(rect, other)
        if not overlap:
            continue
        smaller = max(1, min(_rect_area(rect), _rect_area(other)))
        penalty += (overlap / smaller) * 5200.0 + overlap * 0.18
    return penalty


def _glyph_collision_penalty(candidate: TextPlacement, selected: list[TextPlacement]) -> float:
    if not selected:
        return 0.0

    candidate_rect, candidate_mask = _ensure_alpha_mask(candidate)
    candidate_pixels = max(1, int(cv2.countNonZero(candidate_mask)))
    penalty = 0.0

    for other in selected:
        other_rect, other_mask = _ensure_alpha_mask(other)
        ix1 = max(candidate_rect[0], other_rect[0])
        iy1 = max(candidate_rect[1], other_rect[1])
        ix2 = min(candidate_rect[2], other_rect[2])
        iy2 = min(candidate_rect[3], other_rect[3])
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        c_crop = candidate_mask[
            iy1 - candidate_rect[1]:iy2 - candidate_rect[1],
            ix1 - candidate_rect[0]:ix2 - candidate_rect[0],
        ]
        o_crop = other_mask[
            iy1 - other_rect[1]:iy2 - other_rect[1],
            ix1 - other_rect[0]:ix2 - other_rect[0],
        ]
        overlap_pixels = int(cv2.countNonZero(cv2.bitwise_and(c_crop, o_crop)))
        if not overlap_pixels:
            continue

        other_pixels = max(1, int(cv2.countNonZero(other_mask)))
        smaller = min(candidate_pixels, other_pixels)
        penalty += overlap_pixels * 18.0 + (overlap_pixels / smaller) * 9000.0

    return penalty


def _ensure_alpha_mask(placement: TextPlacement) -> tuple[tuple[int, int, int, int], np.ndarray]:
    if placement.alpha_rect is not None and placement.alpha_mask is not None:
        return placement.alpha_rect, placement.alpha_mask

    x1, y1, x2, y2 = placement.rect
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    layer = Image.new("L", (box_w, box_h), 0)
    draw = ImageDraw.Draw(layer)
    draw.multiline_text(
        (box_w / 2, box_h / 2),
        "\n".join(placement.lines),
        font=placement.font,
        fill=255,
        anchor="mm",
        align="center",
        spacing=placement.spacing,
        stroke_width=placement.stroke_width,
        stroke_fill=255,
    )

    paste_x, paste_y = x1, y1
    if 3 < abs(placement.angle) < 32:
        layer = layer.rotate(-placement.angle, expand=True, resample=Image.Resampling.BICUBIC)
        cx, cy = _rect_center(placement.rect)
        paste_x = int(cx - layer.width / 2)
        paste_y = int(cy - layer.height / 2)

    alpha = np.array(layer, dtype=np.uint8)
    alpha[alpha > 24] = 255
    alpha[alpha <= 24] = 0
    if np.any(alpha):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha = cv2.dilate(alpha, kernel, iterations=1)

    placement.alpha_rect = (paste_x, paste_y, paste_x + layer.width, paste_y + layer.height)
    placement.alpha_mask = alpha
    return placement.alpha_rect, placement.alpha_mask


def _measure_text_block(
    lines: list[str],
    font: ImageFont.ImageFont,
    spacing: int,
    stroke_width: int,
) -> tuple[int, int]:
    measure = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(measure)
    bbox = draw.multiline_textbbox(
        (0, 0),
        "\n".join(lines),
        font=font,
        spacing=spacing,
        stroke_width=stroke_width,
    )
    return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])


def _render_angle(ann: dict[str, Any]) -> float:
    if bool(ann.get("vertical")):
        return 0.0
    angle = float(ann.get("angle") or 0)
    return angle if 3 < abs(angle) < 32 else 0.0


def _text_colors_for_region(
    image: Image.Image,
    rect: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    bg_brightness = _region_brightness(image, rect)
    if bg_brightness < 120:
        return (245, 245, 245, 255), (20, 20, 20, 210)
    return (18, 18, 18, 255), (255, 255, 255, 230)


def _region_brightness(image: Image.Image, rect: tuple[int, int, int, int]) -> float:
    crop = image.crop(rect).convert("L")
    arr = np.array(crop, dtype=np.float32)
    if arr.size == 0:
        return 255
    return float(arr.mean())
