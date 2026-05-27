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


def _sample_contour_points(contour: np.ndarray, limit: int | None = 48) -> list[list[int]]:
    if contour is None or not len(contour):
        return []
    pts = contour.reshape(-1, 2)
    if limit and len(pts) > limit:
        step = max(1, int(np.ceil(len(pts) / limit)))
        pts = pts[::step]
    return [[int(x), int(y)] for x, y in pts]


def _box_area_xywh(box: list[int] | None) -> int:
    if not box:
        return 0
    return max(0, int(box[2])) * max(0, int(box[3]))


def _magic_seed_point(
    blurred: np.ndarray,
    region: dict[str, Any],
    candidate_box: list[int] | None,
    sx1: int,
    sy1: int,
) -> tuple[int, int, int, list[int]] | None:
    height, width = blurred.shape[:2]
    if candidate_box:
        seed_box = candidate_box
    else:
        seed_box = [
            int(region.get("x", 0)),
            int(region.get("y", 0)),
            max(1, int(region.get("width", 1))),
            max(1, int(region.get("height", 1))),
        ]

    lx1 = max(0, min(width - 1, seed_box[0] - sx1))
    ly1 = max(0, min(height - 1, seed_box[1] - sy1))
    lx2 = max(lx1 + 1, min(width, seed_box[0] - sx1 + seed_box[2]))
    ly2 = max(ly1 + 1, min(height, seed_box[1] - sy1 + seed_box[3]))
    window = blurred[ly1:ly2, lx1:lx2]
    if window.size == 0:
        return None

    _, max_value, _, max_loc = cv2.minMaxLoc(window)
    return lx1 + int(max_loc[0]), ly1 + int(max_loc[1]), int(max_value), [sx1 + lx1, sy1 + ly1, lx2 - lx1, ly2 - ly1]


def _polygon_mask_from_points(
    points: Any,
    bubble_rect: tuple[int, int, int, int],
) -> np.ndarray | None:
    if not isinstance(points, list) or len(points) < 3:
        return None
    local_points: list[list[int]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x, y = point[:2]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        local_points.append([int(round(x - bubble_rect[0])), int(round(y - bubble_rect[1]))])
    if len(local_points) < 3:
        return None

    width = max(1, bubble_rect[2] - bubble_rect[0])
    height = max(1, bubble_rect[3] - bubble_rect[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    contour = np.array(local_points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [contour], 255)
    return mask if cv2.countNonZero(mask) else None


def _mask_to_points(mask: np.ndarray, origin: tuple[int, int], limit: int | None = None) -> list[list[int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    return [[origin[0] + px, origin[1] + py] for px, py in _sample_contour_points(contour, limit=limit)]


def _protect_text_geometry(
    mask: np.ndarray,
    origin: tuple[int, int],
    text_box: list[int] | None,
    pad: int,
) -> np.ndarray:
    if not text_box:
        return mask
    local_x1 = int(text_box[0] - origin[0] - pad)
    local_y1 = int(text_box[1] - origin[1] - pad)
    local_x2 = int(text_box[0] - origin[0] + text_box[2] + pad)
    local_y2 = int(text_box[1] - origin[1] + text_box[3] + pad)
    local_x1 = max(0, min(mask.shape[1], local_x1))
    local_y1 = max(0, min(mask.shape[0], local_y1))
    local_x2 = max(local_x1, min(mask.shape[1], local_x2))
    local_y2 = max(local_y1, min(mask.shape[0], local_y2))
    if local_x2 <= local_x1 or local_y2 <= local_y1:
        return mask

    repaired = mask.copy()
    cv2.rectangle(repaired, (local_x1, local_y1), (local_x2 - 1, local_y2 - 1), 255, -1)
    close_size = max(5, min(31, (pad * 2 + 1) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    repaired = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, kernel, iterations=1)
    return repaired


def _mask_bounding_rect(mask: np.ndarray, origin: tuple[int, int]) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    return (
        origin[0] + int(xs.min()),
        origin[1] + int(ys.min()),
        origin[0] + int(xs.max()) + 1,
        origin[1] + int(ys.max()) + 1,
    )


def _edge_touch_tolerance(edge_size: int) -> int:
    return int(max(8, min(edge_size * 0.28, 5 + np.log1p(max(1, edge_size)) * 4.5)))


def _border_touch_profile(mask: np.ndarray) -> tuple[dict[str, bool], dict[str, int], dict[str, int], float, int]:
    height, width = mask.shape[:2]
    if not height or not width:
        empty = {"left": False, "top": False, "right": False, "bottom": False}
        zero = {key: 0 for key in empty}
        return empty, zero, zero, 0.0, 0

    probe = 2
    left_len = int(np.count_nonzero(np.any(mask[:, :min(probe, width)] > 0, axis=1)))
    right_len = int(np.count_nonzero(np.any(mask[:, max(0, width - probe):] > 0, axis=1)))
    top_len = int(np.count_nonzero(np.any(mask[:min(probe, height), :] > 0, axis=0)))
    bottom_len = int(np.count_nonzero(np.any(mask[max(0, height - probe):, :] > 0, axis=0)))
    lengths = {
        "left": left_len,
        "top": top_len,
        "right": right_len,
        "bottom": bottom_len,
    }
    tolerances = {
        "left": _edge_touch_tolerance(height),
        "top": _edge_touch_tolerance(width),
        "right": _edge_touch_tolerance(height),
        "bottom": _edge_touch_tolerance(width),
    }
    touches = {key: value > 0 for key, value in lengths.items()}
    excess = {key: max(0, lengths[key] - tolerances[key]) for key in lengths}
    edge_pressure = sum(excess.values()) / max(1, 2 * (width + height))
    significant_touch_count = sum(1 for value in excess.values() if value > 0)
    return touches, lengths, tolerances, edge_pressure, significant_touch_count


def _discount_image_border_touches(
    touch_lengths: dict[str, int],
    tolerances: dict[str, int],
    sx1: int,
    sy1: int,
    sx2: int,
    sy2: int,
    image_w: int,
    image_h: int,
) -> tuple[dict[str, int], dict[str, bool]]:
    discounted = dict(touch_lengths)
    accepted = {key: False for key in touch_lengths}

    def substantial(side: str, edge_size: int) -> bool:
        length = int(touch_lengths.get(side, 0))
        threshold = max(int(tolerances.get(side, 0)) * 2, int(edge_size * 0.45))
        return length >= threshold

    if sx1 <= 0 and substantial("left", sy2 - sy1):
        discounted["left"] = 0
        accepted["left"] = True
    if sy1 <= 0 and substantial("top", sx2 - sx1):
        discounted["top"] = 0
        accepted["top"] = True
    if sx2 >= image_w and substantial("right", sy2 - sy1):
        discounted["right"] = 0
        accepted["right"] = True
    if sy2 >= image_h and substantial("bottom", sx2 - sx1):
        discounted["bottom"] = 0
        accepted["bottom"] = True
    return discounted, accepted


def _edge_pressure_from_lengths(
    lengths: dict[str, int],
    tolerances: dict[str, int],
    width: int,
    height: int,
) -> tuple[float, int, dict[str, int]]:
    excess = {key: max(0, lengths.get(key, 0) - tolerances.get(key, 0)) for key in tolerances}
    edge_pressure = sum(excess.values()) / max(1, 2 * (width + height))
    significant_touch_count = sum(1 for value in excess.values() if value > 0)
    return edge_pressure, significant_touch_count, excess


def _try_magic_wand_allocation(
    gray_img: np.ndarray,
    region: dict[str, Any],
    debug: dict[str, Any],
    candidate_box: list[int] | None,
    sx1: int,
    sy1: int,
    sx2: int,
    sy2: int,
) -> AllocationSpace | None:
    x = int(region.get("x", 0))
    y = int(region.get("y", 0))
    width = max(1, int(region.get("width", 1)))
    height = max(1, int(region.get("height", 1)))
    image_h, image_w = gray_img.shape[:2]
    roi = gray_img[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return None

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    seed = _magic_seed_point(blurred, region, candidate_box, sx1, sy1)
    if seed is None:
        return None
    seed_x, seed_y, seed_value, seed_box = seed

    otsu_value, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_threshold = int(min(176, max(72, otsu_value - 8)))
    dark_barrier = cv2.threshold(blurred, dark_threshold, 255, cv2.THRESH_BINARY_INV)[1]
    edges = cv2.Canny(blurred, 48, 128)
    barrier = cv2.bitwise_or(dark_barrier, cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))

    # Let the wand start even if the chosen pixel sits on a softened glyph edge.
    cv2.circle(barrier, (seed_x, seed_y), 2, 0, -1)
    flood_mask = np.zeros((roi.shape[0] + 2, roi.shape[1] + 2), dtype=np.uint8)
    flood_mask[1:-1, 1:-1] = np.where(barrier > 0, 1, 0).astype(np.uint8)

    tolerance = int(max(22, min(54, 18 + np.std(blurred) * 0.45)))
    flood_img = blurred.copy()
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    filled_count, _, _, _ = cv2.floodFill(
        flood_img,
        flood_mask,
        (seed_x, seed_y),
        255,
        loDiff=tolerance,
        upDiff=tolerance,
        flags=flags,
    )
    if filled_count <= 0:
        return None

    wand_mask = np.where(flood_mask[1:-1, 1:-1] == 255, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    wand_mask = cv2.morphologyEx(wand_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    wand_mask = cv2.morphologyEx(wand_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(wand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    if contour_area <= 0:
        return None

    left, top, comp_w, comp_h = cv2.boundingRect(contour)
    bubble_box = [sx1 + left, sy1 + top, comp_w, comp_h]
    box_area = max(1, width * height)
    search_area = max(1, roi.shape[0] * roi.shape[1])
    bubble_area = max(1, comp_w * comp_h)
    overlap_box = _intersect_boxes([x, y, width, height], bubble_box)
    overlap_area = (overlap_box[2] * overlap_box[3]) if overlap_box else 0

    touches, touch_lengths, touch_tolerances, raw_edge_pressure, raw_significant_touch_count = _border_touch_profile(wand_mask)
    scored_touch_lengths, image_border_accepted_sides = _discount_image_border_touches(
        touch_lengths,
        touch_tolerances,
        sx1,
        sy1,
        sx2,
        sy2,
        image_w,
        image_h,
    )
    edge_pressure, significant_touch_count, touch_excess = _edge_pressure_from_lengths(
        scored_touch_lengths,
        touch_tolerances,
        wand_mask.shape[1],
        wand_mask.shape[0],
    )
    touch_count = sum(1 for value in touches.values() if value)
    area_ratio_search = contour_area / search_area
    overlap_ratio = overlap_area / box_area
    seed_area = _box_area_xywh(candidate_box) or box_area
    area_gain = contour_area / max(1, seed_area)
    perimeter = max(1.0, cv2.arcLength(contour, True))
    circularity = min(1.0, (4.0 * np.pi * contour_area) / (perimeter * perimeter))
    hull = cv2.convexHull(contour)
    hull_area = max(1.0, cv2.contourArea(hull))
    solidity = min(1.0, contour_area / hull_area)
    fill_ratio = contour_area / bubble_area

    contour_mask = np.zeros_like(wand_mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    outline_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    contour_band = cv2.subtract(
        cv2.dilate(contour_mask, outline_kernel, iterations=1),
        cv2.erode(contour_mask, outline_kernel, iterations=1),
    )
    band_pixels = max(1, cv2.countNonZero(contour_band))
    outline_support = cv2.countNonZero(cv2.bitwise_and(contour_band, dark_barrier)) / band_pixels
    enclosed_shape = outline_support >= 0.14
    compact_shape = solidity >= 0.74 and (fill_ratio >= 0.5 or circularity >= 0.18)

    debug.update({
        "wand_seed": [sx1 + seed_x, sy1 + seed_y],
        "wand_seed_value": seed_value,
        "wand_seed_box": seed_box,
        "wand_tolerance": tolerance,
        "wand_dark_threshold": dark_threshold,
        "wand_area_ratio": round(area_ratio_search, 3),
        "wand_border_touches": touches,
        "wand_border_touch_lengths": touch_lengths,
        "wand_border_scored_touch_lengths": scored_touch_lengths,
        "wand_border_touch_tolerances": touch_tolerances,
        "wand_border_edge_pressure": round(edge_pressure, 3),
        "wand_border_raw_edge_pressure": round(raw_edge_pressure, 3),
        "wand_border_touch_excess": touch_excess,
        "wand_image_border_accepted": any(image_border_accepted_sides.values()),
        "wand_image_border_accepted_sides": image_border_accepted_sides,
        "wand_outline_support": round(outline_support, 3),
        "wand_circularity": round(circularity, 3),
        "wand_solidity": round(solidity, 3),
        "wand_shape_fill_ratio": round(fill_ratio, 3),
    })

    if contour_area < max(80, seed_area * 1.1):
        debug["wand_rejected"] = "too_small"
        return None
    if overlap_ratio < 0.32:
        debug["wand_rejected"] = "misses_text_region"
        return None
    if significant_touch_count >= 2 and edge_pressure > 0.055:
        debug["wand_rejected"] = "likely_background_leak"
        return None
    if area_gain > 18 and edge_pressure > 0.04:
        debug["wand_rejected"] = "overgrown_from_seed"
        return None
    if area_gain > 9 and not enclosed_shape:
        debug["wand_rejected"] = "large_unenclosed_area"
        return None
    if (comp_w > width * 4.8 or comp_h > height * 4.8) and not enclosed_shape:
        debug["wand_rejected"] = "oversized_unenclosed_area"
        return None
    if not compact_shape and not enclosed_shape:
        debug["wand_rejected"] = "not_bubbly_or_rectangular_enough"
        return None

    full_mask = np.zeros_like(wand_mask)
    cv2.drawContours(full_mask, [contour], -1, 255, -1)
    margin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inset_mask = cv2.erode(full_mask, margin_kernel, iterations=1)
    if cv2.countNonZero(inset_mask) > max(80, box_area) * 0.6:
        full_mask = inset_mask

    text_protect_pad = 0
    full_mask = _protect_text_geometry(full_mask, (sx1, sy1), candidate_box or [x, y, width, height], text_protect_pad)
    repaired_rect = _mask_bounding_rect(full_mask, (sx1, sy1))
    if repaired_rect is not None:
        bubble_box = box_xyxy_to_xywh(repaired_rect) or bubble_box
        left = repaired_rect[0] - sx1
        top = repaired_rect[1] - sy1
        comp_w = repaired_rect[2] - repaired_rect[0]
        comp_h = repaired_rect[3] - repaired_rect[1]

    local_mask = full_mask[top:top + comp_h, left:left + comp_w]
    bubble_rect = box_xywh_to_xyxy(bubble_box, image_w, image_h)
    if bubble_rect is None:
        return None
    placement_box = _intersect_boxes(candidate_box, bubble_box) or candidate_box or bubble_box
    placement_rect = box_xywh_to_xyxy(placement_box, image_w, image_h) if placement_box else None
    bubble_points = _mask_to_points(local_mask, (bubble_rect[0], bubble_rect[1]))
    confidence = (
        min(1.0, overlap_ratio) * 0.42
        + min(1.0, fill_ratio) * 0.34
        + min(1.0, area_gain / 4.0) * 0.24
        - min(0.35, edge_pressure * 2.0)
    )

    debug.update({
        "bubble_box": bubble_box,
        "bubble_points": bubble_points,
        "placement_box": placement_box,
        "bubble_area_ratio": round(contour_area / box_area, 3),
        "bubble_fill_ratio": round(fill_ratio, 3),
        "bubble_overlap_ratio": round(overlap_ratio, 3),
        "bubble_confidence": round(min(0.99, max(0.05, confidence)), 3),
        "text_geometry_protected": True,
        "source": "magic_wand",
    })
    return AllocationSpace(
        debug=debug,
        bubble_rect=bubble_rect,
        placement_rect=placement_rect,
        mask=local_mask,
        mask_origin=(bubble_rect[0], bubble_rect[1]),
    )


def _compact_text_allocation(
    region: dict[str, Any],
    debug: dict[str, Any],
    candidate_box: list[int] | None,
    image_w: int,
    image_h: int,
) -> AllocationSpace:
    x = int(region.get("x", 0))
    y = int(region.get("y", 0))
    width = max(1, int(region.get("width", 1)))
    height = max(1, int(region.get("height", 1)))
    font_size = int(region.get("font_size") or 0)
    base = box_xywh_to_xyxy(candidate_box, image_w, image_h) if candidate_box else None
    if base is None:
        base = (x, y, min(image_w, x + width), min(image_h, y + height))
    pad = max(6, int(max(font_size * 0.45, min(width, height) * 0.18)))
    rect = pad_rect(base, image_w, image_h, pad)
    mask = np.zeros((rect[3] - rect[1], rect[2] - rect[0]), dtype=np.uint8)
    radius = max(5, min(mask.shape[:2]) // 5)
    cv2.rectangle(mask, (radius, 0), (mask.shape[1] - radius - 1, mask.shape[0] - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (mask.shape[1] - 1, mask.shape[0] - radius - 1), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (mask.shape[1] - radius - 1, radius), radius, 255, -1)
    cv2.circle(mask, (radius, mask.shape[0] - radius - 1), radius, 255, -1)
    cv2.circle(mask, (mask.shape[1] - radius - 1, mask.shape[0] - radius - 1), radius, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points: list[list[int]] = []
    if contours:
        contour = max(contours, key=cv2.contourArea)
        points = [[rect[0] + px, rect[1] + py] for px, py in _sample_contour_points(contour, limit=None)]

    box = box_xyxy_to_xywh(rect)
    debug.update({
        "bubble_box": box,
        "bubble_points": points,
        "placement_box": candidate_box or box,
        "bubble_area_ratio": round(rect_area(rect) / max(1, width * height), 3),
        "bubble_fill_ratio": round(cv2.countNonZero(mask) / max(1, mask.shape[0] * mask.shape[1]), 3),
        "bubble_overlap_ratio": 1.0,
        "bubble_confidence": 0.42,
        "source": "compact_text_seed",
    })
    return AllocationSpace(
        debug=debug,
        bubble_rect=rect,
        placement_rect=box_xywh_to_xyxy(candidate_box, image_w, image_h) if candidate_box else rect,
        mask=mask,
        mask_origin=(rect[0], rect[1]),
    )


def _debug_bubble_rect(vision: dict[str, Any], image_w: int, image_h: int) -> tuple[int, int, int, int] | None:
    return box_xywh_to_xyxy(vision.get("bubble_box"), image_w, image_h)


def _debug_bubble_mask(vision: dict[str, Any], rect: tuple[int, int, int, int]) -> np.ndarray:
    mask = _polygon_mask_from_points(vision.get("bubble_points"), rect)
    if mask is None:
        mask = np.full((rect[3] - rect[1], rect[2] - rect[0]), 255, dtype=np.uint8)
    return mask


def _vision_text_rect(entry: dict[str, Any], image_w: int, image_h: int) -> tuple[int, int, int, int]:
    region = entry.get("region") or {}
    vision = ((entry.get("ocr_meta") or {}).get("vision") or {}) if isinstance(entry.get("ocr_meta"), dict) else {}
    for key in ("candidate_box", "line_candidate_box"):
        rect = box_xywh_to_xyxy(vision.get(key), image_w, image_h)
        if rect is not None:
            return rect
    return (
        int(region.get("x", 0)),
        int(region.get("y", 0)),
        min(image_w, int(region.get("x", 0)) + max(1, int(region.get("width", 1)))),
        min(image_h, int(region.get("y", 0)) + max(1, int(region.get("height", 1)))),
    )


def _rects_should_reconcile(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    overlap = rect_overlap(a, b)
    if overlap:
        return True
    gap_x, gap_y = _rect_gap(a, b)
    min_side = max(1, min(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]))
    return gap_x <= max(10, int(min_side * 0.16)) and gap_y <= max(10, int(min_side * 0.16))


def _build_union_mask(
    items: list[dict[str, Any]],
    union_rect: tuple[int, int, int, int],
) -> np.ndarray:
    ux1, uy1, ux2, uy2 = union_rect
    union_mask = np.zeros((uy2 - uy1, ux2 - ux1), dtype=np.uint8)
    for item in items:
        rect = item["bubble_rect"]
        mask = item["bubble_mask"]
        lx1 = rect[0] - ux1
        ly1 = rect[1] - uy1
        lx2 = lx1 + mask.shape[1]
        ly2 = ly1 + mask.shape[0]
        union_mask[ly1:ly2, lx1:lx2] = cv2.bitwise_or(union_mask[ly1:ly2, lx1:lx2], mask)
    return union_mask


def _split_union_mask_between_texts(
    union_mask: np.ndarray,
    union_rect: tuple[int, int, int, int],
    text_rect_a: tuple[int, int, int, int],
    text_rect_b: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    ys, xs = np.where(union_mask > 0)
    if not len(xs):
        return None

    ux1, uy1, _, _ = union_rect
    ca = np.array(rect_center(text_rect_a), dtype=np.float64) - np.array([ux1, uy1], dtype=np.float64)
    cb = np.array(rect_center(text_rect_b), dtype=np.float64) - np.array([ux1, uy1], dtype=np.float64)
    axis = cb - ca
    norm = float(np.linalg.norm(axis))
    if norm < 1:
        return None
    axis /= norm

    points = np.column_stack([xs, ys]).astype(np.float64)
    projections = points[:, 0] * float(axis[0]) + points[:, 1] * float(axis[1])
    if not np.all(np.isfinite(projections)):
        return None
    proj_a = float(ca @ axis)
    proj_b = float(cb @ axis)
    low, high = sorted((proj_a, proj_b))
    if high - low < 2:
        return None

    total_area = len(xs)
    midpoint = (low + high) / 2
    text_area_a = max(1, rect_area(text_rect_a))
    text_area_b = max(1, rect_area(text_rect_b))
    target_a_ratio = text_area_a / max(1, text_area_a + text_area_b)
    best: tuple[float, float, int, float, float] | None = None
    for offset in np.linspace(low + 1, high - 1, 49):
        left_count = int(np.count_nonzero(projections <= offset))
        right_count = total_area - left_count
        if left_count <= 0 or right_count <= 0:
            continue
        cut_cost = int(np.count_nonzero(np.abs(projections - offset) <= 0.75))
        count_a = left_count if proj_a <= proj_b else right_count
        balance = abs((count_a / total_area) - target_a_ratio)
        center_bias = abs(offset - midpoint) / max(1.0, high - low)
        score = (cut_cost / max(1.0, np.sqrt(total_area))) * 1.25 + balance * 2.0 + center_bias * 4.0
        if best is None or score < best[0]:
            best = (score, float(offset), cut_cost, balance, center_bias)
    if best is None:
        return None

    _, split_offset, cut_cost, balance, center_bias = best
    grid_y, grid_x = np.indices(union_mask.shape)
    signed = grid_x.astype(np.float64) * axis[0] + grid_y.astype(np.float64) * axis[1] - split_offset
    side_a = signed <= 0 if proj_a <= proj_b else signed > 0
    mask_a = np.where((union_mask > 0) & side_a, 255, 0).astype(np.uint8)
    mask_b = np.where((union_mask > 0) & (~side_a), 255, 0).astype(np.uint8)

    seed_a = (int(round(ca[0])), int(round(ca[1])))
    seed_b = (int(round(cb[0])), int(round(cb[1])))
    if not (0 <= seed_a[0] < union_mask.shape[1] and 0 <= seed_a[1] < union_mask.shape[0]):
        return None
    if not (0 <= seed_b[0] < union_mask.shape[1] and 0 <= seed_b[1] < union_mask.shape[0]):
        return None
    if mask_a[seed_a[1], seed_a[0]] == 0 or mask_b[seed_b[1], seed_b[0]] == 0:
        return None

    detail = {
        "cut_axis": [round(float(axis[0]), 3), round(float(axis[1]), 3)],
        "cut_offset": round(split_offset, 2),
        "cut_cost": int(cut_cost),
        "cut_balance": round(float(balance), 3),
        "cut_target_area_ratio": round(float(target_a_ratio), 3),
        "cut_center_bias": round(float(center_bias), 3),
        "combined_area": int(total_area),
    }
    return mask_a, mask_b, detail


def _trim_mask_to_rect(mask: np.ndarray, origin: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    rect = (origin[0] + x1, origin[1] + y1, origin[0] + x2, origin[1] + y2)
    return mask[y1:y2, x1:x2], rect


def _erase_rect_from_mask(
    mask: np.ndarray,
    origin: tuple[int, int],
    rect: tuple[int, int, int, int],
) -> np.ndarray:
    local_x1 = max(0, min(mask.shape[1], int(rect[0] - origin[0])))
    local_y1 = max(0, min(mask.shape[0], int(rect[1] - origin[1])))
    local_x2 = max(local_x1, min(mask.shape[1], int(rect[2] - origin[0])))
    local_y2 = max(local_y1, min(mask.shape[0], int(rect[3] - origin[1])))
    if local_x2 <= local_x1 or local_y2 <= local_y1:
        return mask
    carved = mask.copy()
    carved[local_y1:local_y2, local_x1:local_x2] = 0
    return carved


def reconcile_overlapping_bubble_spaces(
    entries: list[dict[str, Any]],
    image_w: int,
    image_h: int,
) -> None:
    """Split overlapping bubble allocations by a low-cost cut between text seeds."""
    if len(entries) < 2:
        return

    items: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        ocr_meta = entry.get("ocr_meta") or {}
        vision = ocr_meta.get("vision") or {}
        allocation = ocr_meta.get("_allocation_space")
        bubble_rect = getattr(allocation, "bubble_rect", None) or _debug_bubble_rect(vision, image_w, image_h)
        if bubble_rect is None:
            continue
        allocation_mask = getattr(allocation, "mask", None)
        items.append({
            "index": index,
            "entry": entry,
            "vision": vision,
            "bubble_rect": bubble_rect,
            "bubble_mask": allocation_mask if isinstance(allocation_mask, np.ndarray) else _debug_bubble_mask(vision, bubble_rect),
            "text_rect": _vision_text_rect(entry, image_w, image_h),
        })

    used_pairs: set[tuple[int, int]] = set()
    for i, item_a in enumerate(items):
        for item_b in items[i + 1:]:
            pair_key = tuple(sorted((item_a["index"], item_b["index"])))
            if pair_key in used_pairs:
                continue
            if not _rects_should_reconcile(item_a["bubble_rect"], item_b["bubble_rect"]):
                continue

            union_rect = _merge_rects([item_a["bubble_rect"], item_b["bubble_rect"]])
            if union_rect is None:
                continue
            union_mask = _build_union_mask([item_a, item_b], union_rect)
            split = _split_union_mask_between_texts(
                union_mask,
                union_rect,
                item_a["text_rect"],
                item_b["text_rect"],
            )
            if split is None:
                continue
            mask_a, mask_b, split_detail = split
            for item, split_mask in ((item_a, mask_a), (item_b, mask_b)):
                text_rect = item["text_rect"]
                text_box = [
                    text_rect[0],
                    text_rect[1],
                    max(1, text_rect[2] - text_rect[0]),
                    max(1, text_rect[3] - text_rect[1]),
                ]
                text_pad = 0
                split_mask = _protect_text_geometry(split_mask, (union_rect[0], union_rect[1]), text_box, text_pad)
                for other_item in items:
                    if other_item["index"] == item["index"]:
                        continue
                    split_mask = _erase_rect_from_mask(split_mask, (union_rect[0], union_rect[1]), other_item["text_rect"])
                trimmed = _trim_mask_to_rect(split_mask, (union_rect[0], union_rect[1]))
                if trimmed is None:
                    continue
                local_mask, rect = trimmed
                box = box_xyxy_to_xywh(rect)
                if not box:
                    continue
                item["vision"].update({
                    "bubble_box": box,
                    "bubble_points": _mask_to_points(local_mask, (rect[0], rect[1])),
                    "source": f"{item['vision'].get('source', 'unknown')}_split",
                    "split_from_combined_box": box_xyxy_to_xywh(union_rect),
                    "split_strategy": "shortest_between_texts",
                    "split_detail": split_detail,
                })
            used_pairs.add(pair_key)

    for item in items:
        vision = item["vision"]
        bubble_rect = _debug_bubble_rect(vision, image_w, image_h)
        if bubble_rect is None:
            continue
        mask = _debug_bubble_mask(vision, bubble_rect)
        origin = (bubble_rect[0], bubble_rect[1])
        for other_item in items:
            if other_item["index"] == item["index"]:
                continue
            mask = _erase_rect_from_mask(mask, origin, other_item["text_rect"])
        text_rect = item["text_rect"]
        text_box = [
            text_rect[0],
            text_rect[1],
            max(1, text_rect[2] - text_rect[0]),
            max(1, text_rect[3] - text_rect[1]),
        ]
        mask = _protect_text_geometry(mask, origin, text_box, 0)
        trimmed = _trim_mask_to_rect(mask, origin)
        if trimmed is None:
            continue
        local_mask, rect = trimmed
        box = box_xyxy_to_xywh(rect)
        if not box:
            continue
        vision.update({
            "bubble_box": box,
            "bubble_points": _mask_to_points(local_mask, (rect[0], rect[1])),
            "text_bbox_exclusion_applied": True,
        })


def estimate_allocation_space(gray_img: np.ndarray, region: dict[str, Any]) -> AllocationSpace:
    x = int(region.get("x", 0))
    y = int(region.get("y", 0))
    width = max(1, int(region.get("width", 1)))
    height = max(1, int(region.get("height", 1)))
    image_h, image_w = gray_img.shape[:2]
    line_candidate_box = _line_candidate_box(region)
    candidate_box = candidate_box_from_region(region, gray_img)
    vertical = bool(region.get("vertical")) or height > width * 1.25
    if vertical:
        search_pad_x = max(24, int(width * 1.8), int(height * 0.7))
        search_pad_y = max(24, int(height * 0.65), int(width * 0.8))
    else:
        search_pad_x = max(24, int(width * 1.35), int(height * 0.9))
        search_pad_y = max(24, int(height * 1.15), int(width * 0.65))
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

    wand_space = _try_magic_wand_allocation(
        gray_img,
        region,
        debug,
        candidate_box,
        sx1,
        sy1,
        sx2,
        sy2,
    )
    if wand_space is not None:
        return wand_space
    if debug.get("wand_rejected") in {
        "likely_background_leak",
        "overgrown_from_seed",
        "large_unenclosed_area",
        "oversized_unenclosed_area",
        "not_bubbly_or_rectangular_enough",
    }:
        return _compact_text_allocation(region, debug, candidate_box, image_w, image_h)

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
        "bubble_points": [[sx1 + px, sy1 + py] for px, py in _sample_contour_points(contour, limit=None)],
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
            mask = _polygon_mask_from_points(existing_debug.get("bubble_points"), bubble_rect)
            if mask is None:
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
