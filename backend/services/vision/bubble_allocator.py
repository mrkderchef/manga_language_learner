from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class AllocationSpace:
    debug: dict[str, Any]
    bubble_rect: tuple[int, int, int, int] | None = None
    placement_rect: tuple[int, int, int, int] | None = None
    mask: np.ndarray | None = None
    mask_origin: tuple[int, int] | None = None


def clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(int(round(value)), high))


def rect_area(rect: tuple[int, int, int, int]) -> int:
    return max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1])


def rect_center(rect: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)


def rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def rect_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def union_rect(
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


def pad_rect(
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


def inset_rect(
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


def expanded_rect(
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


def box_xyxy_to_xywh(rect: tuple[int, int, int, int] | None) -> list[int] | None:
    if rect is None:
        return None
    return [rect[0], rect[1], max(1, rect[2] - rect[0]), max(1, rect[3] - rect[1])]


def box_xywh_to_xyxy(box: Any, image_w: int, image_h: int) -> tuple[int, int, int, int] | None:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        x, y, width, height = [int(round(float(value))) for value in box[:4]]
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    x1 = max(0, min(image_w, x))
    y1 = max(0, min(image_h, y))
    x2 = max(x1 + 1, min(image_w, x + width))
    y2 = max(y1 + 1, min(image_h, y + height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_xyxy(bbox: Any, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    points = np.array(bbox, dtype=np.int32).reshape(-1, 2)
    x1 = int(np.clip(points[:, 0].min(), 0, image_w - 1))
    y1 = int(np.clip(points[:, 1].min(), 0, image_h - 1))
    x2 = int(np.clip(points[:, 0].max(), x1 + 1, image_w))
    y2 = int(np.clip(points[:, 1].max(), y1 + 1, image_h))
    return x1, y1, x2, y2


def extract_line_points(region: dict[str, Any]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for line in region.get("lines") or []:
        for point in line or []:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            x = point[0]
            y = point[1]
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                continue
            points.append((int(round(x)), int(round(y))))
    return points


def _line_candidate_box(region: dict[str, Any]) -> list[int] | None:
    points = extract_line_points(region)
    if not points:
        return None

    x = int(region.get("x", 0))
    y = int(region.get("y", 0))
    width = max(1, int(region.get("width", 1)))
    height = max(1, int(region.get("height", 1)))
    pad = max(int(region.get("font_size") or 0), 10)
    min_x = clamp_int(min(px for px, _ in points) - pad, x, x + width)
    min_y = clamp_int(min(py for _, py in points) - pad, y, y + height)
    max_x = clamp_int(max(px for px, _ in points) + pad, x, x + width)
    max_y = clamp_int(max(py for _, py in points) + pad, y, y + height)
    if max_x <= min_x or max_y <= min_y:
        return None
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def _rect_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    gap_x = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    gap_y = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return gap_x, gap_y


def _merge_rects(rects: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not rects:
        return None
    return (
        min(rect[0] for rect in rects),
        min(rect[1] for rect in rects),
        max(rect[2] for rect in rects),
        max(rect[3] for rect in rects),
    )


def _refine_candidate_box_with_ink(gray_img: np.ndarray, region: dict[str, Any], seed_box: list[int] | None) -> list[int] | None:
    x = int(region.get("x", 0))
    y = int(region.get("y", 0))
    width = max(1, int(region.get("width", 1)))
    height = max(1, int(region.get("height", 1)))
    roi = gray_img[y:y + height, x:x + width]
    if roi.size == 0:
        return seed_box

    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    otsu_value, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_threshold = int(min(172, max(76, otsu_value)))
    dark = cv2.threshold(blurred, dark_threshold, 255, cv2.THRESH_BINARY_INV)[1]
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8, cv2.CV_32S)
    if component_count <= 1:
        return seed_box

    font_size = int(region.get("font_size") or 0)
    gap_limit = min(max(12, int(max(font_size * 0.45, min(width, height) * 0.12))), 42)
    box_area = max(1, width * height)
    local_seed = None
    if seed_box:
        local_seed = (
            max(0, seed_box[0] - x),
            max(0, seed_box[1] - y),
            min(width, seed_box[0] - x + seed_box[2]),
            min(height, seed_box[1] - y + seed_box[3]),
        )

    components: list[tuple[int, int, int, int]] = []
    for label in range(1, component_count):
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 6 or area > box_area * 0.28:
            continue
        components.append((left, top, left + comp_w, top + comp_h))

    if not components:
        return seed_box

    selected: list[tuple[int, int, int, int]] = []
    if local_seed is not None:
        expanded_seed = (
            max(0, local_seed[0] - gap_limit),
            max(0, local_seed[1] - gap_limit),
            min(width, local_seed[2] + gap_limit),
            min(height, local_seed[3] + gap_limit),
        )
        for comp_rect in components:
            if rect_overlap(comp_rect, expanded_seed):
                selected.append(comp_rect)

    if not selected:
        center_rect = (
            max(0, width // 2 - max(10, width // 6)),
            max(0, height // 2 - max(10, height // 6)),
            min(width, width // 2 + max(10, width // 6)),
            min(height, height // 2 + max(10, height // 6)),
        )
        selected = [comp_rect for comp_rect in components if rect_overlap(comp_rect, center_rect)]

    if not selected:
        return seed_box

    grew = True
    while grew:
        grew = False
        merged = _merge_rects(selected)
        assert merged is not None
        for comp_rect in components:
            if comp_rect in selected:
                continue
            gap_x, gap_y = _rect_gap(comp_rect, merged)
            if gap_x <= gap_limit and gap_y <= gap_limit:
                selected.append(comp_rect)
                grew = True

    merged = _merge_rects(selected)
    if merged is None:
        return seed_box

    global_rect = (x + merged[0], y + merged[1], x + merged[2], y + merged[3])
    if seed_box:
        seed_rect = (seed_box[0], seed_box[1], seed_box[0] + seed_box[2], seed_box[1] + seed_box[3])
        global_rect = (
            min(global_rect[0], seed_rect[0]),
            min(global_rect[1], seed_rect[1]),
            max(global_rect[2], seed_rect[2]),
            max(global_rect[3], seed_rect[3]),
        )

    return [global_rect[0], global_rect[1], max(1, global_rect[2] - global_rect[0]), max(1, global_rect[3] - global_rect[1])]


def candidate_box_from_region(region: dict[str, Any], gray_img: np.ndarray | None = None) -> list[int] | None:
    seed_box = _line_candidate_box(region)
    if gray_img is None:
        return seed_box
    return _refine_candidate_box_with_ink(gray_img, region, seed_box)


def annotation_to_region(ann: dict[str, Any], image_w: int, image_h: int) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox_xyxy(ann.get("bbox"), image_w, image_h)
    return {
        "x": x1,
        "y": y1,
        "width": max(1, x2 - x1),
        "height": max(1, y2 - y1),
        "vertical": bool(ann.get("vertical")),
        "font_size": int(ann.get("font_size") or 0),
        "lines": ann.get("lines") or [],
    }


def _intersect_boxes(a: list[int] | None, b: list[int] | None) -> list[int] | None:
    if not a or not b:
        return None
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def _nearest_foreground_label(component_labels: np.ndarray, anchor_x: int, anchor_y: int) -> int:
    height, width = component_labels.shape[:2]
    if not height or not width:
        return 0
    anchor_x = max(0, min(anchor_x, width - 1))
    anchor_y = max(0, min(anchor_y, height - 1))
    label = int(component_labels[anchor_y, anchor_x])
    if label > 0:
        return label
    ys, xs = np.where(component_labels > 0)
    if not len(xs):
        return 0
    dist = (xs - anchor_x) ** 2 + (ys - anchor_y) ** 2
    index = int(np.argmin(dist))
    return int(component_labels[ys[index], xs[index]])


def _sample_contour_points(contour: np.ndarray, limit: int = 48) -> list[list[int]]:
    if contour is None or not len(contour):
        return []
    pts = contour.reshape(-1, 2)
    if len(pts) > limit:
        step = max(1, int(np.ceil(len(pts) / limit)))
        pts = pts[::step]
    return [[int(x), int(y)] for x, y in pts]


def estimate_allocation_space(gray_img: np.ndarray, region: dict[str, Any]) -> AllocationSpace:
    x = int(region.get("x", 0))
    y = int(region.get("y", 0))
    width = max(1, int(region.get("width", 1)))
    height = max(1, int(region.get("height", 1)))
    image_h, image_w = gray_img.shape[:2]
    line_candidate_box = _line_candidate_box(region)
    candidate_box = candidate_box_from_region(region, gray_img)
    search_pad_x = max(18, int(width * 0.9))
    search_pad_y = max(18, int(height * 0.9))
    sx1 = max(0, x - search_pad_x)
    sy1 = max(0, y - search_pad_y)
    sx2 = min(image_w, x + width + search_pad_x)
    sy2 = min(image_h, y + height + search_pad_y)
    roi = gray_img[sy1:sy2, sx1:sx2]

    debug = {
        "box": [x, y, width, height],
        "search_box": [sx1, sy1, max(1, sx2 - sx1), max(1, sy2 - sy1)],
        "candidate_box": candidate_box,
        "line_candidate_box": line_candidate_box,
        "line_count": len(region.get("lines") or []),
    }
    if roi.size == 0:
        return AllocationSpace(debug=debug)

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    otsu_value, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value = int(max(180, otsu_value))
    bright = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)[1]
    kernel_size = max(3, int(max(3, min(width, height) * 0.08))) | 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8, cv2.CV_32S)
    debug["threshold"] = threshold_value
    if component_count <= 1:
        return AllocationSpace(debug=debug)

    if candidate_box:
        anchor_x = candidate_box[0] + candidate_box[2] // 2 - sx1
        anchor_y = candidate_box[1] + candidate_box[3] // 2 - sy1
    else:
        anchor_x = x + width // 2 - sx1
        anchor_y = y + height // 2 - sy1

    label = _nearest_foreground_label(component_labels, anchor_x, anchor_y)
    if label <= 0 or label >= len(stats):
        return AllocationSpace(debug=debug)

    left = int(stats[label, cv2.CC_STAT_LEFT])
    top = int(stats[label, cv2.CC_STAT_TOP])
    comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
    comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
    area = int(stats[label, cv2.CC_STAT_AREA])
    component_mask = np.where(component_labels == label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return AllocationSpace(debug=debug)

    contour = max(contours, key=cv2.contourArea)
    epsilon = max(1.0, 0.012 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True)

    bubble_box = [sx1 + left, sy1 + top, comp_w, comp_h]
    box_area = max(1, width * height)
    bubble_area = max(1, comp_w * comp_h)
    placement_box = _intersect_boxes(candidate_box, bubble_box) or candidate_box or bubble_box
    overlap_box = _intersect_boxes([x, y, width, height], bubble_box)
    overlap_area = (overlap_box[2] * overlap_box[3]) if overlap_box else 0

    full_mask = np.zeros(bright.shape, dtype=np.uint8)
    cv2.drawContours(full_mask, [contour], -1, 255, -1)
    margin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inset_mask = cv2.erode(full_mask, margin_kernel, iterations=1)
    min_area = max(80, box_area * 1.25)
    if cv2.countNonZero(inset_mask) > min_area * 0.6:
        full_mask = inset_mask
    local_mask = full_mask[top:top + comp_h, left:left + comp_w]
    bubble_rect = box_xywh_to_xyxy(bubble_box, image_w, image_h)
    placement_rect = box_xywh_to_xyxy(placement_box, image_w, image_h) if placement_box else None

    debug.update({
        "bubble_box": bubble_box,
        "bubble_points": [[sx1 + px, sy1 + py] for px, py in _sample_contour_points(approx)],
        "placement_box": placement_box,
        "bubble_area_ratio": round(area / box_area, 3),
        "bubble_fill_ratio": round(area / bubble_area, 3),
        "bubble_overlap_ratio": round(overlap_area / box_area, 3),
        "bubble_confidence": round(min(0.99, max(0.05, (overlap_area / box_area) * 0.6 + (area / bubble_area) * 0.4)), 3),
        "source": "white_component",
    })
    return AllocationSpace(
        debug=debug,
        bubble_rect=bubble_rect,
        placement_rect=placement_rect,
        mask=local_mask,
        mask_origin=(bubble_rect[0], bubble_rect[1]) if bubble_rect else None,
    )


def allocation_space_from_annotation(
    gray_img: np.ndarray,
    ann: dict[str, Any],
    image_w: int,
    image_h: int,
) -> AllocationSpace:
    existing_debug = ((ann.get("ocr_debug") or {}).get("vision") or {}) if isinstance(ann.get("ocr_debug"), dict) else {}
    if existing_debug:
        bubble_rect = box_xywh_to_xyxy(existing_debug.get("bubble_box"), image_w, image_h)
        placement_rect = box_xywh_to_xyxy(existing_debug.get("placement_box"), image_w, image_h)
        if bubble_rect is not None:
            mask = np.full((bubble_rect[3] - bubble_rect[1], bubble_rect[2] - bubble_rect[0]), 255, dtype=np.uint8)
            return AllocationSpace(
                debug=existing_debug,
                bubble_rect=bubble_rect,
                placement_rect=placement_rect,
                mask=mask,
                mask_origin=(bubble_rect[0], bubble_rect[1]),
            )
    region = annotation_to_region(ann, image_w, image_h)
    return estimate_allocation_space(gray_img, region)
