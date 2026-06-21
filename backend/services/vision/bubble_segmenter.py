"""Optional page-level instance segmentation for manga speech balloons.

The model is deliberately lazy and optional. OCR remains functional when the
checkpoint or Ultralytics runtime is unavailable; callers then use the
classical allocator.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import cv2
import numpy as np

from config import BUBBLE_MODEL_PATH
from services.model_assets import BUBBLE_MODEL_REPO_ID, BUBBLE_MODEL_REVISION, bubble_model_available

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = 1280
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.70

_model: Any = None
_model_error: str | None = None
_last_inference_error: str | None = None
_last_prediction_count: int | None = None


@dataclass
class BubblePrediction:
    bubble_id: str
    confidence: float
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    model_id: str = BUBBLE_MODEL_REPO_ID
    model_revision: str = BUBBLE_MODEL_REVISION


def reset_model_cache() -> None:
    global _model, _model_error, _last_inference_error, _last_prediction_count
    _model = None
    _model_error = None
    _last_inference_error = None
    _last_prediction_count = None


def _load_model() -> Any | None:
    global _model, _model_error
    if _model is not None:
        return _model
    if _model_error is not None or not bubble_model_available():
        return None
    try:
        from ultralytics import YOLO

        _model = YOLO(str(BUBBLE_MODEL_PATH))
        return _model
    except Exception as exc:
        _model_error = str(exc)
        logger.warning("Bubble segmentation model unavailable: %s", exc)
        return None


def model_unavailable_reason() -> str:
    if not bubble_model_available():
        return "checkpoint_missing"
    if _model_error:
        return f"model_load_failed:{_model_error}"
    if _last_inference_error:
        return f"inference_failed:{_last_inference_error}"
    if _model is not None:
        return "no_balloon_predictions" if _last_prediction_count == 0 else "ready"
    return "runtime_unavailable"


def _refine_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask >= 0.5, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary
    refined = np.zeros_like(binary)
    cv2.drawContours(refined, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return refined


def predict_bubbles(image_bgr: np.ndarray, options: dict[str, Any] | None = None) -> list[BubblePrediction]:
    global _last_inference_error, _last_prediction_count
    options = options or {}
    model = _load_model()
    if model is None or image_bgr.size == 0:
        _last_prediction_count = 0
        return []

    image_h, image_w = image_bgr.shape[:2]
    try:
        results = model.predict(
            source=image_bgr,
            imgsz=int(options.get("model_image_size", DEFAULT_IMAGE_SIZE) or DEFAULT_IMAGE_SIZE),
            conf=float(options.get("model_confidence", DEFAULT_CONFIDENCE) or DEFAULT_CONFIDENCE),
            iou=float(options.get("model_iou", DEFAULT_IOU) or DEFAULT_IOU),
            retina_masks=True,
            verbose=False,
        )
    except Exception as exc:
        _last_inference_error = str(exc)
        _last_prediction_count = 0
        logger.warning("Bubble segmentation inference failed: %s", exc)
        return []
    _last_inference_error = None
    if not results:
        _last_prediction_count = 0
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or getattr(masks, "data", None) is None:
        _last_prediction_count = 0
        return []

    names = getattr(result, "names", None) or getattr(model, "names", {}) or {}
    raw: list[tuple[float, tuple[int, int, int, int], np.ndarray]] = []
    for index in range(min(len(boxes), len(masks.data))):
        class_id = int(float(boxes.cls[index]))
        label = str(names.get(class_id, class_id)).lower() if isinstance(names, dict) else str(class_id)
        if label != "balloon" and class_id != 2:
            continue
        confidence = float(boxes.conf[index])
        xyxy = boxes.xyxy[index].detach().cpu().numpy().tolist()
        x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_w, x2), min(image_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        mask = masks.data[index].detach().cpu().numpy()
        if mask.shape[:2] != (image_h, image_w):
            mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
        refined = _refine_mask(mask)
        if cv2.countNonZero(refined) == 0:
            continue
        raw.append((confidence, (x1, y1, x2, y2), refined))

    raw.sort(key=lambda item: (item[1][1], item[1][0], -item[0]))
    predictions = [
        BubblePrediction(f"bubble_{index:03d}", confidence, bbox, mask)
        for index, (confidence, bbox, mask) in enumerate(raw, start=1)
    ]
    _last_prediction_count = len(predictions)
    return predictions


__all__ = [
    "BubblePrediction",
    "model_unavailable_reason",
    "predict_bubbles",
    "reset_model_cache",
]
