from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from services.storage.image_service import ImageService
import services.recognition.mangaocr as manga_ocr_service
import services.rabbithole.nlp as rabbithole_service
import services.translation.engine as translation_engine
from services.rendering.panel_renderer import render_translated_panel
from services.detection.region_detector import detect_text_regions
from services.model_assets import BUBBLE_MODEL_REVISION
import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from PIL import Image
from config import (
    BASE_DIR,
    ocr_panel_dir,
    ocr_panel_state_dir,
    panel_rabbithole_dir,
    panel_rendered_dir,
    panel_translations_dir,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/scanner", tags=["scanner"])

LEGACY_OCR_CACHE_DIR = BASE_DIR / "backend" / "data" / "ocr_cache"
OCR_CACHE_VERSION = "ocr-stepwise-state-v1"
MISSING_TRANSLATION_TEXT = "No translation available"
STATE_CACHE_BUCKETS = ("ocr",)
PANEL_CACHE_BUCKETS = ("ocr", "rabbithole", "translation")
RABBITHOLE_JOB_TTL_SECONDS = 15 * 60
RABBITHOLE_JOB_WORKERS = 2
_RABBITHOLE_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=RABBITHOLE_JOB_WORKERS, thread_name_prefix="rabbithole")
_RABBITHOLE_JOBS: dict[str, dict] = {}
_RABBITHOLE_JOBS_LOCK = threading.Lock()


def _normalize_scan_result(result: dict | None) -> dict | None:
    if not result or not result.get("success"):
        return result

    normalized = dict(result)
    has_translation_context = any(
        key in result
        for key in (
            "translation_engine_requested",
            "translation_engine_used",
            "translation_model",
            "translation_error",
        )
    )
    suppress_translation_placeholder = (
        bool(result.get("translation_error"))
        or result.get("translation_engine_used") in {None, "none"}
    )
    annotations = []
    for ann in result.get("annotations", []) or []:
        copy = dict(ann)
        if copy.get("uncomputed") or copy.get("computed") is False:
            copy["computed"] = False
            copy["uncomputed"] = True
            annotations.append(copy)
            continue
        has_bbox = bool(copy.get("bbox"))
        if not has_bbox:
            copy["localization_missing"] = True
        elif has_translation_context and str(copy.get("translated") or "").strip() in {"", "—", "..."}:
            copy["translated"] = "" if suppress_translation_placeholder else MISSING_TRANSLATION_TEXT
            copy["translation_missing"] = True
        annotations.append(copy)
    normalized["annotations"] = annotations
    return normalized


def _sanitize_ocr_annotation(annotation: dict | None) -> dict | None:
    if not annotation:
        return annotation
    allowed = {
        "id",
        "text",
        "confidence",
        "bbox",
        "char_count",
        "vertical",
        "ocr_variant",
        "region_id",
        "recognized_orientation",
        "orientation_source",
        "reading_order",
        "font_size",
        "angle",
        "lines",
        "ocr_debug",
        "box_signature",
        "computed",
        "uncomputed",
        "box_source",
        "ocr_status",
    }
    clean = {key: copy.deepcopy(value) for key, value in annotation.items() if key in allowed}
    if "translated" in clean:
        clean.pop("translated", None)
    clean["computed"] = bool(annotation.get("computed", True))
    clean["uncomputed"] = bool(annotation.get("uncomputed", False))
    return clean


def _stage_root_dir(panel_path: Path, kind: str) -> Path:
    if kind == "rabbithole":
        return panel_rabbithole_dir(panel_path)
    if kind == "translation":
        return panel_translations_dir(panel_path)
    raise ValueError(f"Unsupported persisted stage: {kind}")


def _stage_cache_dir(panel_path: Path, kind: str) -> Path:
    return _stage_root_dir(panel_path, kind) / "cache"


def _stage_cache_path(panel_path: Path, kind: str, cache_key: str) -> Path:
    return _stage_cache_dir(panel_path, kind) / f"{cache_key}.json"


def _read_persisted_stage(path: Path) -> tuple[dict | None, dict | None]:
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    if not isinstance(payload, dict):
        return None, None
    result = payload.get("result") if "result" in payload else payload
    return (result if isinstance(result, dict) else None), payload


def _translation_current_path(panel_path: Path) -> Path:
    return _stage_root_dir(panel_path, "translation") / "current.json"


def _legacy_translation_paths(panel_path: Path) -> list[Path]:
    root = _stage_root_dir(panel_path, "translation")
    paths = list((root / "cache").glob("*.json")) if (root / "cache").exists() else []
    legacy_latest = root / "latest.json"
    if legacy_latest.exists():
        paths.append(legacy_latest)
    return paths


def _remove_legacy_translation_artifacts(panel_path: Path) -> None:
    root = _stage_root_dir(panel_path, "translation")
    shutil.rmtree(root / "cache", ignore_errors=True)
    legacy_latest = root / "latest.json"
    if legacy_latest.exists():
        legacy_latest.unlink()


def _load_current_translation(panel_path: Path) -> dict | None:
    current, _payload = _read_persisted_stage(_translation_current_path(panel_path))
    if current is not None:
        _remove_legacy_translation_artifacts(panel_path)
        return current

    # One-time migration: retain only the newest result from the former
    # model/settings-keyed cache, then remove the obsolete cache directory.
    candidates: list[tuple[float, str, dict]] = []
    for path in _legacy_translation_paths(panel_path):
        result, payload = _read_persisted_stage(path)
        if result is None:
            continue
        saved_at = float((payload or {}).get("saved_at") or path.stat().st_mtime)
        cache_key = str((payload or {}).get("cache_key") or path.stem)
        candidates.append((saved_at, cache_key, result))
    if not candidates:
        return None

    _saved_at, cache_key, result = max(candidates, key=lambda item: item[0])
    _save_persisted_stage(panel_path, "translation", cache_key, result)
    return result


def _load_persisted_stage(panel_path: Path, kind: str, cache_key: str) -> dict | None:
    if kind == "translation":
        return _load_current_translation(panel_path)
    result, _payload = _read_persisted_stage(_stage_cache_path(panel_path, kind, cache_key))
    return result


def _save_persisted_stage(panel_path: Path, kind: str, cache_key: str, result: dict) -> None:
    payload = {
        "cache_key": cache_key,
        "saved_at": round(time.time(), 3),
        "result": result,
    }
    if kind == "translation":
        root = _stage_root_dir(panel_path, kind)
        root.mkdir(parents=True, exist_ok=True)
        _remove_legacy_translation_artifacts(panel_path)
        path = _translation_current_path(panel_path)
    else:
        cache_dir = _stage_cache_dir(panel_path, kind)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _stage_cache_path(panel_path, kind, cache_key)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    legacy_latest = _stage_root_dir(panel_path, kind) / "latest.json"
    if legacy_latest.exists():
        legacy_latest.unlink()


def _delete_stage_artifacts(panel_path: Path, kind: str) -> None:
    shutil.rmtree(_stage_root_dir(panel_path, kind), ignore_errors=True)


def _delete_rendered_output(panel_path: Path) -> None:
    shutil.rmtree(panel_rendered_dir(panel_path), ignore_errors=True)


def _cache_key(panel_path: Path) -> str:
    """Generate a cache key from file path + modification time."""
    stat = panel_path.stat()
    raw = f"{OCR_CACHE_VERSION}:{panel_path}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()


def _state_file(panel_path: Path) -> Path:
    return ocr_panel_state_dir(panel_path) / f"{_cache_key(panel_path)}.json"


def _default_state() -> dict:
    return {
        "version": OCR_CACHE_VERSION,
        "boxes": [],
        "auto_boxes": [],
        "annotations": {},
        "overrides": {},
        "detection": [],
        "cache": {bucket: {} for bucket in STATE_CACHE_BUCKETS},
    }


def _ensure_state_shape(data: dict) -> dict:
    data.setdefault("version", OCR_CACHE_VERSION)
    data["boxes"] = [_normalize_box(box) for box in data.get("boxes", [])]
    data["auto_boxes"] = [_normalize_box(box, source="detected") for box in data.get("auto_boxes", [])]
    data["annotations"] = {
        region_id: _sanitize_ocr_annotation(annotation)
        for region_id, annotation in (data.get("annotations", {}) or {}).items()
        if _sanitize_ocr_annotation(annotation)
    }
    data.setdefault("overrides", {})
    data.setdefault("detection", [])
    ocr_cache = ((data.get("cache") or {}).get("ocr") or {})
    data["cache"] = {"ocr": ocr_cache}
    data.pop("ocr_logic_version", None)
    data.pop("derived", None)
    return data


def _load_state(panel_path: Path) -> dict:
    path = _state_file(panel_path)
    if not path.exists():
        return _default_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _ensure_state_shape(data)
    except Exception:
        return _default_state()


def _save_state(panel_path: Path, state: dict) -> None:
    state["version"] = OCR_CACHE_VERSION
    state.pop("ocr_logic_version", None)
    _state_file(panel_path).parent.mkdir(parents=True, exist_ok=True)
    _state_file(panel_path).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    _export_panel_metadata(panel_path, state)


def _export_panel_metadata(panel_path: Path, state: dict) -> None:
    """
    Capture and export panel-level metadata (source, import date, edit history).
    
    This provides context about the panel's origin, import time, and edits,
    enabling data provenance and panel management workflows.
    """
    try:
        from config import panel_metadata_path
        
        # Load existing metadata if present (to preserve history)
        metadata_file = panel_metadata_path(panel_path)
        if metadata_file.exists():
            panel_meta = json.loads(metadata_file.read_text(encoding="utf-8"))
        else:
            panel_meta = {
                "source_path": str(panel_path),
                "filename": panel_path.name,
                "created_at": time.time(),
                "edits": [],
            }
        
        # Update metadata
        panel_meta["last_modified"] = time.time()
        stat = panel_path.stat()
        panel_meta["file_size"] = stat.st_size
        panel_meta["file_mtime"] = stat.st_mtime
        
        # Track if this is an edit (by comparing state versions)
        if "last_state_hash" in panel_meta:
            # Check if state has meaningfully changed
            state_copy = dict(state)
            state_copy.pop("cache", None)  # Don't track cache changes
            current_hash = hashlib.md5(json.dumps(state_copy, sort_keys=True, default=str).encode()).hexdigest()
            if current_hash != panel_meta.get("last_state_hash"):
                panel_meta["edits"].append({
                    "timestamp": time.time(),
                    "state_hash": current_hash,
                    "annotation_count": len(state.get("annotations", {})),
                })
        
        # Update state hash
        state_copy = dict(state)
        state_copy.pop("cache", None)
        panel_meta["last_state_hash"] = hashlib.md5(json.dumps(state_copy, sort_keys=True, default=str).encode()).hexdigest()
        panel_meta["last_annotation_count"] = len(state.get("annotations", {}))
        
        # Save metadata
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.write_text(json.dumps(panel_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        
        logger.debug(f"Exported panel metadata for {panel_path.name} to {metadata_file}")
    except Exception as e:
        logger.warning(
            'component=metadata panel="%s" status=failed error=%r msg="Could not export panel metadata"',
            panel_path.name,
            e,
        )


def _region_id(region: dict) -> str:
    raw = f"{region.get('x')}:{region.get('y')}:{region.get('width')}:{region.get('height')}"
    return "region_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]


def _detect_regions_with_ids(panel_path: Path, options: dict | None = None) -> list[dict]:
    regions = []
    options = options or {}
    for region in detect_text_regions(str(panel_path), options=options):
        copy = dict(region)
        copy["region_id"] = _region_id(copy)
        regions.append(copy)
    return regions


def _apply_overrides(regions: list[dict], overrides: dict) -> list[dict]:
    by_id = {region["region_id"]: dict(region) for region in regions}
    removed = set(overrides.get("removed", []))
    for region_id in removed:
        by_id.pop(region_id, None)
    for region_id, patch in overrides.get("regions", {}).items():
        if region_id in by_id:
            by_id[region_id].update(patch)
    for region in overrides.get("added", []):
        copy = dict(region)
        copy.setdefault("region_id", _region_id(copy))
        if copy["region_id"] in removed:
            continue
        copy.setdefault("lines", [])
        copy.setdefault("font_size", 0)
        copy.setdefault("angle", 0)
        by_id[copy["region_id"]] = copy
    return list(by_id.values())


def _image_dimensions(panel_path: Path) -> tuple[int, int]:
    with Image.open(panel_path) as image:
        return image.size


def _normalize_box(region: dict, source: str | None = None, computed: bool | None = None) -> dict:
    x = max(0, int(region.get("x") or 0))
    y = max(0, int(region.get("y") or 0))
    width = max(1, int(region.get("width") or 1))
    height = max(1, int(region.get("height") or 1))
    forced_orientation = region.get("forced_orientation") or region.get("orientation_override") or region.get("orientation")
    vertical = bool(region.get("vertical")) if region.get("vertical") is not None else height >= width
    if forced_orientation in {"vertical", "horizontal"}:
        vertical = forced_orientation == "vertical"
    normalized = {
        "region_id": str(region.get("region_id") or _region_id({"x": x, "y": y, "width": width, "height": height})),
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "vertical": vertical,
        "font_size": int(region.get("font_size") or 0),
        "angle": int(region.get("angle") or 0),
        "lines": region.get("lines") or [],
        "source": region.get("source") or source or "manual",
        "computed": bool(region.get("computed")) if computed is None else bool(computed),
    }
    if forced_orientation in {"vertical", "horizontal"}:
        normalized["forced_orientation"] = forced_orientation
    return normalized


def _box_bbox(box: dict) -> list[list[int]]:
    x = int(box["x"])
    y = int(box["y"])
    width = int(box["width"])
    height = int(box["height"])
    return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]


def _box_signature(box: dict) -> str:
    payload = {
        "x": int(box.get("x") or 0),
        "y": int(box.get("y") or 0),
        "width": int(box.get("width") or 0),
        "height": int(box.get("height") or 0),
        "vertical": bool(box.get("vertical")),
        "forced_orientation": box.get("forced_orientation") or "",
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _annotation_matches_box(annotation: dict | None, box: dict) -> bool:
    if not annotation:
        return False
    signature = _box_signature(box)
    debug = annotation.get("ocr_debug") or {}
    return annotation.get("box_signature") == signature or debug.get("box_signature") == signature


def _uncomputed_annotation(box: dict, index: int) -> dict:
    orientation = box.get("forced_orientation") or ("vertical" if box.get("vertical") else "horizontal")
    return {
        "id": f"box_{index:04d}",
        "region_id": box["region_id"],
        "text": "",
        "confidence": 0.0,
        "bbox": _box_bbox(box),
        "char_count": 0,
        "vertical": bool(box.get("vertical")),
        "recognized_orientation": orientation,
        "orientation_source": "manual" if box.get("forced_orientation") else "box",
        "reading_order": index,
        "font_size": int(box.get("font_size") or 0),
        "angle": int(box.get("angle") or 0),
        "lines": box.get("lines") or [],
        "computed": False,
        "uncomputed": True,
        "box_source": box.get("source", "manual"),
        "ocr_status": "uncomputed",
    }


def _box_ids(boxes: list[dict]) -> set[str]:
    return {str(box.get("region_id")) for box in boxes if box.get("region_id")}


def _manual_boxes(state: dict) -> list[dict]:
    state["boxes"] = [_normalize_box(box) for box in state.get("boxes", [])]
    return state["boxes"]


def _auto_boxes(state: dict) -> list[dict]:
    state["auto_boxes"] = [_normalize_box(box, source="detected") for box in state.get("auto_boxes", [])]
    return state["auto_boxes"]


def _display_boxes(state: dict) -> list[dict]:
    manual = _manual_boxes(state)
    boxes = manual if manual else _auto_boxes(state)
    valid_ids = _box_ids(state["boxes"])
    if not manual:
        valid_ids = _box_ids(state["auto_boxes"])
    annotations = state.setdefault("annotations", {})
    for region_id in list(annotations.keys()):
        if region_id not in valid_ids:
            annotations.pop(region_id, None)
    return boxes


def _promote_display_boxes_to_manual(state: dict) -> list[dict]:
    manual = _manual_boxes(state)
    if manual:
        return manual
    promoted = []
    for box in _auto_boxes(state):
        copy = _normalize_box(box, source="manual", computed=box.get("computed", False))
        copy["source"] = "manual"
        promoted.append(copy)
    state["boxes"] = promoted
    state["auto_boxes"] = []
    return state["boxes"]


def _set_box_computed(state: dict, region_id: str, computed: bool) -> None:
    for box in state.get("boxes", []) + state.get("auto_boxes", []):
        if box.get("region_id") == region_id:
            box["computed"] = computed
            return


def _annotation_from_computed(box: dict, annotation: dict, index: int) -> dict:
    ann = copy.deepcopy(annotation)
    ann["region_id"] = box["region_id"]
    ann["bbox"] = _box_bbox(box)
    ann["vertical"] = bool(box.get("vertical"))
    ann["recognized_orientation"] = ann.get("recognized_orientation") or box.get("forced_orientation") or ("vertical" if box.get("vertical") else "horizontal")
    ann["orientation_source"] = ann.get("orientation_source") or ("manual" if box.get("forced_orientation") else "box")
    ann["reading_order"] = index
    ann["font_size"] = int(box.get("font_size") or ann.get("font_size") or 0)
    ann["angle"] = int(box.get("angle") or ann.get("angle") or 0)
    ann["lines"] = box.get("lines") or ann.get("lines", [])
    ann["computed"] = True
    ann["uncomputed"] = False
    ann["box_source"] = box.get("source", "manual")
    ann["ocr_status"] = "computed"
    return ann


def _build_ocr_result_from_state(panel_path: Path, state: dict, options: dict | None = None, base_result: dict | None = None) -> dict:
    boxes = _display_boxes(state)
    annotations_by_region = state.setdefault("annotations", {})
    image_w, image_h = _image_dimensions(panel_path)
    annotations: list[dict] = []
    for index, box in enumerate(boxes, start=1):
        region_id = box["region_id"]
        stored = annotations_by_region.get(region_id)
        if stored and box.get("computed") and _annotation_matches_box(stored, box):
            annotations.append(_annotation_from_computed(box, stored, index))
        else:
            if stored and not _annotation_matches_box(stored, box):
                annotations_by_region.pop(region_id, None)
                box["computed"] = False
            annotations.append(_uncomputed_annotation(box, index))
    text = "\n".join(ann.get("text", "") for ann in annotations if ann.get("computed") and ann.get("text"))
    result = copy.deepcopy(base_result or {})
    result.update({
        "success": True,
        "text": text,
        "annotations": annotations,
        "method": result.get("method") or "panel-box-state",
        "image_width": result.get("image_width") or image_w,
        "image_height": result.get("image_height") or image_h,
        "ocr_engine_requested": result.get("ocr_engine_requested") or (options or {}).get("ocr_engine", "mangaocr"),
        "ocr_engine_used": result.get("ocr_engine_used") or (options or {}).get("ocr_engine", "mangaocr"),
        "fallback_used": result.get("fallback_used", False),
        "fallback_reason": result.get("fallback_reason"),
    })
    return _normalize_scan_result(result)


def _store_partial_ocr_result(state: dict, partial_result: dict, target_regions: list[dict]) -> None:
    target_ids = _box_ids(target_regions)
    annotations = state.setdefault("annotations", {})
    for region_id in target_ids:
        annotations.pop(region_id, None)
        _set_box_computed(state, region_id, False)

    target_by_id = {box["region_id"]: box for box in target_regions}
    returned_ids: set[str] = set()
    for ann in partial_result.get("annotations", []) or []:
        region_id = str(ann.get("region_id") or ann.get("id") or "")
        if region_id not in target_by_id:
            continue
        returned_ids.add(region_id)
        box = target_by_id[region_id]
        signature = _box_signature(box)
        clean = _sanitize_ocr_annotation(ann) or {}
        clean["region_id"] = region_id
        clean["box_signature"] = signature
        clean["bbox"] = _box_bbox(box)
        clean["computed"] = True
        clean["uncomputed"] = False
        debug = clean.setdefault("ocr_debug", {})
        debug["box_signature"] = signature
        debug["target_box"] = [int(box["x"]), int(box["y"]), int(box["width"]), int(box["height"])]
        annotations[region_id] = clean
        _set_box_computed(state, region_id, True)

    missing_auto_ids = {
        region_id
        for region_id in (target_ids - returned_ids)
        if target_by_id.get(region_id, {}).get("source", "detected") == "detected"
    }
    if missing_auto_ids:
        state["auto_boxes"] = [
            box for box in state.get("auto_boxes", [])
            if box.get("region_id") not in missing_auto_ids
        ]


def _regions_payload(panel_path: Path, state: dict) -> dict:
    result = _build_ocr_result_from_state(panel_path, state)
    return {
        "success": True,
        "regions": copy.deepcopy(_display_boxes(state)),
        "annotations": result["annotations"],
        "text": result.get("text", ""),
        "image_width": result.get("image_width"),
        "image_height": result.get("image_height"),
        "panel_cache": _panel_cache_status(state, panel_path),
    }


def _json_hash(payload: dict | list | str) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _ocr_settings_key(options: dict) -> str:
    relevant = {
        "confidence_threshold": options.get("confidence_threshold", 0.4),
        "nms_threshold": options.get("nms_threshold", 0.35),
        "mask_threshold": options.get("mask_threshold", 0.3),
        "box_threshold": options.get("box_threshold", 0.6),
        "max_regions": options.get("max_regions"),
        "ocr_quality_mode": options.get("ocr_quality_mode", "balanced"),
        "semantic_rerank": options.get("semantic_rerank", "close"),
        "vertical_preference": options.get("vertical_preference", "normal"),
        "rotation_win_margin": options.get("rotation_win_margin", 15),
        "preprocessing_set": options.get("preprocessing_set", "standard"),
        "crop_upscale": options.get("crop_upscale", 3),
        "crop_padding_ratio": options.get("crop_padding_ratio", 0.05),
        "crop_padding_min": options.get("crop_padding_min", 4),
        "enable_rotated_variants": options.get("enable_rotated_variants", True),
        "bubble": {
            "pipeline_version": "hybrid-v1",
            "model_revision": BUBBLE_MODEL_REVISION,
            "mode": (options.get("bubble") or {}).get("mode", "hybrid") if isinstance(options.get("bubble"), dict) else "hybrid",
            "model_confidence": (options.get("bubble") or {}).get("model_confidence", 0.25) if isinstance(options.get("bubble"), dict) else 0.25,
            "model_iou": (options.get("bubble") or {}).get("model_iou", 0.7) if isinstance(options.get("bubble"), dict) else 0.7,
            "search_scale": (options.get("bubble") or {}).get("search_scale", 1.0) if isinstance(options.get("bubble"), dict) else 1.0,
            "wand_enabled": (options.get("bubble") or {}).get("wand_enabled", True) if isinstance(options.get("bubble"), dict) else True,
            "overlap_reconciliation": (options.get("bubble") or {}).get("overlap_reconciliation", True) if isinstance(options.get("bubble"), dict) else True,
        },
    }
    return _json_hash(relevant)


def _ocr_options_key(options: dict, boxes: list[dict] | dict | None) -> str:
    normalized_boxes = [
        {
            "region_id": box.get("region_id"),
            "x": box.get("x"),
            "y": box.get("y"),
            "width": box.get("width"),
            "height": box.get("height"),
            "vertical": box.get("vertical"),
            "forced_orientation": box.get("forced_orientation"),
        }
        for box in (boxes or [])
    ] if isinstance(boxes, list) else boxes
    return _json_hash({
        "settings": _ocr_settings_key(options),
        "boxes": normalized_boxes or [],
    })


def _texts_hash(annotations: list[dict]) -> str:
    texts = [ann.get("text", "") for ann in annotations if ann.get("computed", True) and ann.get("text")]
    return _json_hash(texts)


def _translation_units_hash(annotations: list[dict]) -> str:
    units = [
        {
            "region_id": str(ann.get("region_id") or ann.get("id") or f"ann_{index + 1:04d}"),
            "text": ann.get("text", ""),
        }
        for index, ann in enumerate(annotations)
        if ann.get("computed", True) and ann.get("text")
    ]
    return _json_hash(units)


def _translation_engine(options: dict) -> str:
    return "ollama"


def _translation_model(options: dict, resolve_auto: bool = False) -> str | None:
    selected = options.get("translation_model") or None
    if selected:
        return selected
    if resolve_auto:
        return translation_engine.preferred_ollama_text_model()
    return None


def _translation_options_key(options: dict, annotations: list[dict]) -> str:
    engine = _translation_engine(options)
    relevant = {
        "translation_units_hash": _translation_units_hash(annotations),
        "association_version": "region-id-v1",
        "translation_engine": engine,
        "translation_model": _translation_model(options, resolve_auto=True) or "",
        "target_lang": options.get("target_lang", "en"),
        "translation_style": options.get("translation_style", "natural"),
        "temperature": options.get("temperature", 0.1),
        "prompt_version": translation_engine.PROMPT_VERSION,
        "translation_strategy": translation_engine.MANGA_DIALOGUE_STRATEGY,
        "target_chunk_size": translation_engine.MANGA_TARGET_CHUNK_SIZE,
        "max_context_lines": translation_engine.MANGA_CONTEXT_LINE_LIMIT,
    }
    return _json_hash(relevant)


def _rabbithole_options_key(annotations: list[dict]) -> str:
    return _json_hash({
        "text_hash": _texts_hash(annotations),
        "contract_version": rabbithole_service.RABBITHOLE_CONTRACT_VERSION,
    })


def _new_scan_id() -> str:
    return uuid.uuid4().hex[:8]


def _trace(trace: list[dict], scan_id: str, stage: str, status: str, message: str, started_at: float | None = None, **fields) -> None:
    event = {
        "scan_id": scan_id,
        "stage": stage,
        "status": status,
        "message": message,
        "ts": round(time.time(), 3),
    }
    if started_at is not None:
        event["elapsed_ms"] = int((time.perf_counter() - started_at) * 1000)
    event.update({k: v for k, v in fields.items() if v is not None})
    log_fields = " ".join(
        f'{key}="{str(value).replace(chr(34), chr(39))}"' if isinstance(value, str) else f"{key}={value}"
        for key, value in event.items()
        if key not in {"scan_id", "stage", "status", "message", "ts"}
    )
    suffix = f" {log_fields}" if log_fields else ""
    logger.info('scan_id=%s stage=%s status=%s msg="%s"%s', scan_id, stage, status, message.replace('"', "'"), suffix)
    trace.append(event)


def _panel_cache_status(state: dict, panel_path: Path) -> dict:
    boxes = _display_boxes(state)
    manual_boxes = _manual_boxes(state)
    cache = state.get("cache", {})
    legacy_key = _cache_key(panel_path)
    legacy_file = LEGACY_OCR_CACHE_DIR / f"{legacy_key}.json"
    legacy_count = int(legacy_file.exists())
    has_current_translation = _translation_current_path(panel_path).exists() or bool(_legacy_translation_paths(panel_path))
    buckets = {
        "ocr": {
            "has_cache": bool(cache.get("ocr")),
            "entries": len(cache.get("ocr", {})),
        },
        "rabbithole": {
            "has_cache": any(_stage_cache_dir(panel_path, "rabbithole").glob("*.json")) if _stage_cache_dir(panel_path, "rabbithole").exists() else False,
            "entries": len(list(_stage_cache_dir(panel_path, "rabbithole").glob("*.json"))) if _stage_cache_dir(panel_path, "rabbithole").exists() else 0,
        },
        "translation": {
            "has_cache": has_current_translation,
            "entries": int(has_current_translation),
        },
    }
    if state.get("annotations"):
        buckets["ocr"]["has_cache"] = True
        buckets["ocr"]["entries"] = max(buckets["ocr"]["entries"], len(state.get("annotations", {})))
    if legacy_count:
        buckets["ocr"]["has_cache"] = True
        buckets["ocr"]["entries"] += legacy_count
    return {
        "has_cache": any(item["has_cache"] for item in buckets.values()),
        "buckets": buckets,
        "legacy_cache": {
            "has_cache": legacy_count > 0,
            "entries": legacy_count,
        },
        "has_overrides": bool(manual_boxes),
        "box_count": len(boxes),
        "uncomputed_box_count": sum(1 for box in boxes if not box.get("computed")),
        "state_file": _state_file(panel_path).name,
    }


def _clear_panel_cache_buckets(state: dict, kinds: list[str] | None = None) -> list[str]:
    requested = kinds or list(PANEL_CACHE_BUCKETS)
    full_panel_reset = kinds is None
    cleared = []
    cache = state.setdefault("cache", {})
    for kind in requested:
        if kind in STATE_CACHE_BUCKETS:
            cache[kind] = {}
            cleared.append(kind)
            if kind == "ocr":
                state["annotations"] = {}
                state["detection"] = []
                state.pop("ocr_settings_key", None)
                for box in state.get("boxes", []) + state.get("auto_boxes", []):
                    box["computed"] = False
        elif kind in {"rabbithole", "translation"}:
            cleared.append(kind)
    if full_panel_reset:
        state["boxes"] = []
        state["auto_boxes"] = []
        state["overrides"] = {}
    return cleared


def _delete_panel_ocr_tree(panel_path: Path) -> None:
	shutil.rmtree(ocr_panel_dir(panel_path), ignore_errors=True)


def _delete_panel_ocr_artifacts(panel_path: Path) -> None:
    ocr_root = ocr_panel_dir(panel_path)
    for child in ("debug", "cache"):
        shutil.rmtree(ocr_root / child, ignore_errors=True)


def _clear_cached_results(state: dict, kinds: list[str]) -> None:
    cache = state.setdefault("cache", {})
    for kind in kinds:
        if kind in STATE_CACHE_BUCKETS:
            cache[kind] = {}


def _prepare_panel_regions(panel_path: Path, options: dict, state: dict, trace: list[dict], scan_id: str) -> tuple[list[dict], list[dict]]:
    if options.get("reset_manual_edits"):
        state["boxes"] = []
        state["auto_boxes"] = []
        state["annotations"] = {}
        state["overrides"] = {}
        state["cache"] = {bucket: {} for bucket in STATE_CACHE_BUCKETS}
        state.pop("ocr_settings_key", None)
        _delete_stage_artifacts(panel_path, "rabbithole")
        _delete_stage_artifacts(panel_path, "translation")
        _delete_rendered_output(panel_path)
        _trace(trace, scan_id, "overrides", "reset", "Manual edits reset by scan option")

    start = time.perf_counter()
    manual_boxes = _manual_boxes(state)
    boxes = manual_boxes
    detected_count = None
    if not manual_boxes:
        detected = _detect_regions_with_ids(panel_path, options)
        state["detection"] = detected
        boxes = [_normalize_box(region, source="detected", computed=False) for region in detected]
        state["auto_boxes"] = boxes
        detected_count = len(detected)

    settings_key = _ocr_settings_key(options)
    settings_changed = bool(state.get("annotations")) and bool(state.get("ocr_settings_key")) and state.get("ocr_settings_key") != settings_key
    annotations = state.setdefault("annotations", {})
    target_regions = boxes if detected_count is not None or options.get("fresh") or settings_changed else [
        box for box in boxes
        if not box.get("computed") or not _annotation_matches_box(annotations.get(box.get("region_id")), box)
    ]
    _trace(
        trace,
        scan_id,
        "ocr",
        "regions",
        "Prepared active OCR boxes",
        start,
        detected_regions=detected_count,
        active_regions=len(boxes),
        target_regions=len(target_regions),
        incremental=detected_count is None,
        settings_changed=settings_changed,
    )
    return boxes, copy.deepcopy(target_regions)


def _strip_embedded_translation(result: dict) -> dict:
    clean = copy.deepcopy(result)
    for ann in clean.get("annotations", []) or []:
        ann.pop("translated", None)
    for key in (
        "translation_engine_requested",
        "translation_engine_used",
        "translation_model",
        "translation_target_lang",
        "translation_style",
        "translation_prompt_version",
        "translation_prompt_payload",
        "translation_error",
    ):
        clean.pop(key, None)
    return clean


def _run_selected_ocr_engine(panel_path: Path, options: dict, regions: list[dict], trace: list[dict], scan_id: str) -> dict:
    from services.recognition.ocr_provider import run_ocr

    ocr_engine = "mangaocr"
    start = time.perf_counter()
    _trace(trace, scan_id, "ocr", "start", "OCR stage started", engine=ocr_engine)

    # Delegate OCR invocation to the MangaOCR recognition service.
    result = run_ocr(panel_path, options=options, regions=regions)

    # Normalize and enrich result as before
    result = _strip_embedded_translation(_normalize_scan_result(result))
    result["ocr_engine_requested"] = ocr_engine
    result["ocr_engine_used"] = result.get("ocr_engine_used") or ocr_engine
    result["fallback_used"] = False
    result["fallback_reason"] = None
    _trace(
        trace,
        scan_id,
        "ocr",
        "done",
        "OCR stage completed",
        start,
        engine=result.get("ocr_engine_used"),
        annotations=len(result.get("annotations", []) or []),
        text_chars=len(result.get("text") or ""),
    )
    return result


def _run_ocr_stage(panel_path: Path, options: dict, state: dict, trace: list[dict], scan_id: str) -> dict:
    """Run or reuse OCR only. Rabbithole and translation are separate stages."""
    options = options or {}
    use_cache = bool(options.get("use_cache", True)) and not bool(options.get("fresh", False))

    try:
        boxes, target_regions = _prepare_panel_regions(panel_path, options, state, trace, scan_id)
        cache_key = _ocr_options_key(options, boxes)
        cache_bucket = state.setdefault("cache", {}).setdefault("ocr", {})

        if not boxes:
            result = _build_ocr_result_from_state(panel_path, state, options)
            _trace(trace, scan_id, "ocr", "empty", "No OCR boxes available", cache_key=cache_key)
            return result

        if not target_regions:
            result = _build_ocr_result_from_state(panel_path, state, options)
            cache_bucket[cache_key] = result
            _trace(trace, scan_id, "ocr", "noop", "All boxes already computed", cache_key=cache_key)
            return result

        if use_cache:
            _trace(trace, scan_id, "ocr", "incremental", "OCR will compute uncomputed boxes", cache_key=cache_key, target_regions=len(target_regions))
        else:
            _trace(trace, scan_id, "ocr", "fresh", "OCR cache bypassed", cache_key=cache_key)

        _delete_stage_artifacts(panel_path, "rabbithole")
        _delete_stage_artifacts(panel_path, "translation")
        _delete_rendered_output(panel_path)
        partial = _run_selected_ocr_engine(panel_path, options, target_regions, trace, scan_id)
        _store_partial_ocr_result(state, partial, target_regions)
        state["ocr_settings_key"] = _ocr_settings_key(options)
        result = _build_ocr_result_from_state(panel_path, state, options, partial)
        cache_bucket[cache_key] = result
        return result
    except Exception as e:
        logger.exception('stage=ocr status=failed msg="OCR stage exception"')
        _trace(trace, scan_id, "ocr", "error", str(e))
        return {
            "success": False,
            "error": str(e),
            "text": "",
            "annotations": [],
            "ocr_engine_requested": options.get("ocr_engine", "mangaocr"),
            "ocr_engine_used": None,
            "fallback_used": False,
            "fallback_reason": None,
            "available_ocr_engines": _ocr_engine_status(),
        }


def _run_ocr(panel_path: Path, options: dict | None = None) -> dict:
    """Run OCR only with explicit engine selection and no hidden fallback."""
    scan_id = _new_scan_id()
    trace: list[dict] = []
    state = _load_state(panel_path)
    result = _run_ocr_stage(panel_path, options or {}, state, trace, scan_id)
    result["scan_trace"] = trace
    result["panel_cache"] = _panel_cache_status(state, panel_path)
    _save_state(panel_path, state)
    return result


def _empty_translation_result(text_count: int, options: dict, error: str | None = None) -> dict:
    engine = _translation_engine(options)
    return {
        "success": error is None,
        "translations": [""] * text_count,
        "translation_engine_requested": engine,
        "translation_engine_used": None if error else "ollama",
        "translation_model": _translation_model(options, resolve_auto=True),
        "translation_target_lang": options.get("target_lang", "en"),
        "translation_style": options.get("translation_style", "natural"),
        "translation_prompt_version": translation_engine.PROMPT_VERSION,
        "translation_prompt_payload": None,
        "fallback_used": False,
        "translation_error": error,
    }


def _compact_bbox(bbox: list | tuple | None) -> dict | None:
    if not bbox:
        return None
    try:
        if len(bbox) == 4 and all(isinstance(item, (int, float)) for item in bbox):
            x, y, width, height = bbox
            return {"x": int(x), "y": int(y), "width": int(width), "height": int(height)}
        points = [
            (float(point[0]), float(point[1]))
            for point in bbox
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if not points:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return {
            "x": int(round(min_x)),
            "y": int(round(min_y)),
            "width": int(round(max_x - min_x)),
            "height": int(round(max_y - min_y)),
        }
    except Exception:
        return None


def _translation_context_unit(annotation: dict, annotation_index: int) -> dict:
    region_id = annotation.get("region_id") or annotation.get("id") or f"ann_{annotation_index + 1:04d}"
    return {
        "region_id": str(region_id),
        "index": annotation_index,
        "text": annotation.get("text", ""),
        "reading_order": annotation.get("reading_order") or annotation_index + 1,
        "orientation": annotation.get("recognized_orientation") or ("vertical" if annotation.get("vertical") else "horizontal"),
        "vertical": bool(annotation.get("vertical")),
        "box": _compact_bbox(annotation.get("bbox")),
    }


def _run_translation_stage(ocr_result: dict, options: dict, state: dict, trace: list[dict], scan_id: str) -> tuple[dict, str | None, bool]:
    annotations = ocr_result.get("annotations", []) or []
    indexed_units = [
        (index, _translation_context_unit(ann, index))
        for index, ann in enumerate(annotations)
        if ann.get("computed", True) and ann.get("text")
    ]
    texts = [unit["text"] for _, unit in indexed_units]
    context_units = [unit for _, unit in indexed_units]
    if options.get("cache_only"):
        cached = _load_persisted_stage(state["panel_path"], "translation", "current")
        if cached:
            _trace(trace, scan_id, "translation", "current_hit", "Current translation loaded")
            return copy.deepcopy(cached), None, False
        _trace(trace, scan_id, "translation", "current_miss", "Current translation is missing")
        return {
            "success": True,
            "translations": [""] * len(annotations),
            "cache_miss": True,
        }, None, False

    engine = _translation_engine(options)
    model = _translation_model(options, resolve_auto=True)
    requested_model = _translation_model(options, resolve_auto=False)
    cache_key = _translation_options_key(options, annotations)

    if not texts:
        result = _empty_translation_result(len(annotations), options)
        return result, cache_key, True

    prompt_payload = {
        "engine": engine,
        "model": model,
        "target_lang": options.get("target_lang", "en"),
        "style": options.get("translation_style", "natural"),
        "temperature": float(options.get("temperature", 0.1)),
        "texts": texts,
        "strategy": translation_engine.MANGA_DIALOGUE_STRATEGY,
        "target_chunk_size": translation_engine.MANGA_TARGET_CHUNK_SIZE,
        "max_context_lines": translation_engine.MANGA_CONTEXT_LINE_LIMIT,
    }

    _trace(
        trace,
        scan_id,
        "translation",
        "start",
        "Translation stage started",
        engine=engine,
        model=model,
        text_blocks=len(texts),
    )
    start = time.perf_counter()
    try:
        result = translation_engine.translate_batch(
            texts,
            target_lang=options.get("target_lang", "en"),
            engine=engine,
            model=requested_model,
            style=options.get("translation_style", "natural"),
            temperature=float(options.get("temperature", 0.1)),
            context_units=context_units,
        )
        translated_by_region: dict[str, str] = {}
        expanded = [""] * len(annotations)
        for (annotation_index, _unit), translated in zip(indexed_units, result.get("translations", [])):
            expanded[annotation_index] = translated
            translated_by_region[str(_unit["region_id"])] = translated
        result["translations"] = expanded
        result["translations_by_region"] = translated_by_region
        result["translation_prompt_payload"] = result.get("translation_prompt_payload") or prompt_payload
        _trace(
            trace,
            scan_id,
            "translation",
            "done",
            "Translation stage completed",
            start,
            engine=result.get("translation_engine_used"),
            model=result.get("translation_model"),
        )
        return result, cache_key, True
    except Exception as exc:
        result = _empty_translation_result(len(annotations), options, str(exc))
        result["translation_prompt_payload"] = prompt_payload
        _trace(trace, scan_id, "translation", "error", str(exc), start, engine=engine, model=model)
        return result, cache_key, False


def _run_rabbithole_stage(ocr_result: dict, panel_path: Path, options: dict, trace: list[dict], scan_id: str) -> tuple[dict, str | None, bool]:
    annotations = ocr_result.get("annotations", []) or []
    cache_key = _rabbithole_options_key(annotations)
    use_cache = bool(options.get("use_cache", True)) and not bool(options.get("fresh", False))

    if use_cache:
        cached = _load_persisted_stage(panel_path, "rabbithole", cache_key)
        if cached:
            _trace(trace, scan_id, "rabbithole", "cache_hit", "Rabbithole cache hit", cache_key=cache_key)
            return copy.deepcopy(cached), cache_key, False
    if options.get("cache_only"):
        _trace(trace, scan_id, "rabbithole", "cache_miss", "Rabbithole cache miss", cache_key=cache_key)
        return {
            "success": True,
            "cache_miss": True,
            "by_region": {},
            "global_lookup_hits": 0,
            "global_lookup_misses": 0,
        }, cache_key, False

    _trace(trace, scan_id, "rabbithole", "start", "Rabbithole stage started", text_blocks=len(annotations))
    start = time.perf_counter()
    try:
        result = rabbithole_service.build_panel_rabbithole(annotations)
        _trace(
            trace,
            scan_id,
            "rabbithole",
            "done",
            "Rabbithole stage completed",
            start,
            token_regions=len(result.get("by_region", {})),
            lookup_hits=result.get("global_lookup_hits", 0),
            lookup_misses=result.get("global_lookup_misses", 0),
        )
        return result, cache_key, True
    except Exception as exc:
        result = {
            "success": False,
            "by_region": {},
            "global_lookup_hits": 0,
            "global_lookup_misses": 0,
            "rabbithole_error": str(exc),
        }
        _trace(trace, scan_id, "rabbithole", "error", str(exc), start)
        return result, cache_key, False


def _merge_rabbithole_result(ocr_result: dict, rabbithole_result: dict) -> dict:
    result = copy.deepcopy(ocr_result)
    annotations = result.get("annotations", []) or []
    by_region = rabbithole_result.get("by_region", {}) if rabbithole_result else {}
    for index, ann in enumerate(annotations):
        region_id = ann.get("region_id") or ann.get("id") or f"region_{index + 1:04d}"
        rabbit = by_region.get(region_id)
        if rabbit:
            ann["rabbithole"] = rabbit
    result["annotations"] = annotations
    result["rabbithole_success"] = bool(rabbithole_result.get("success"))
    result["rabbithole_source"] = rabbithole_result.get("source")
    result["rabbithole_lookup_hits"] = rabbithole_result.get("global_lookup_hits", 0)
    result["rabbithole_lookup_misses"] = rabbithole_result.get("global_lookup_misses", 0)
    if rabbithole_result.get("rabbithole_error"):
        result["rabbithole_error"] = rabbithole_result["rabbithole_error"]
    return result


def _merge_translation_result(base_result: dict, translation_result: dict, options: dict) -> dict:
    result = copy.deepcopy(base_result)
    annotations = result.get("annotations", []) or []
    translations = translation_result.get("translations", [])
    translations_by_region = translation_result.get("translations_by_region", {}) or {}
    for index, ann in enumerate(annotations):
        region_id = str(ann.get("region_id") or ann.get("id") or f"ann_{index + 1:04d}")
        ann["translated"] = (
            translations_by_region[region_id]
            if region_id in translations_by_region
            else translations[index] if index < len(translations) else ""
        )
    result["annotations"] = annotations
    result.update({key: value for key, value in translation_result.items() if key != "translations"})
    result["translation_engine_requested"] = translation_result.get(
        "translation_engine_requested",
        options.get("translation_engine", "ollama"),
    )
    return _normalize_scan_result(result)


def _attach_cached_rabbithole(panel_path: Path, result: dict) -> dict:
    cache_key = _rabbithole_options_key(result.get("annotations", []) or [])
    cached = _load_persisted_stage(panel_path, "rabbithole", cache_key)
    if cached:
        return _merge_rabbithole_result(result, cached)
    return result


def _run_rabbithole_existing(panel_path: Path, options: dict | None = None) -> dict:
    """Build or reuse Rabbithole analysis for the current OCR state."""
    options = options or {}
    scan_id = _new_scan_id()
    trace: list[dict] = []
    state = _load_state(panel_path)
    _trace(trace, scan_id, "rabbithole", "start", "Rabbithole started", panel=panel_path.name)

    ocr_result = _build_ocr_result_from_state(panel_path, state, options)
    computed_count = sum(1 for ann in ocr_result.get("annotations", []) if ann.get("computed") and ann.get("text"))
    if not computed_count:
        result = {
            "success": False,
            "error": "Run Scan before Rabbithole. OCR cache for the current OCR settings is missing.",
            "text": "",
            "annotations": ocr_result.get("annotations", []),
            "scan_trace": trace,
            "panel_cache": _panel_cache_status(state, panel_path),
        }
        _trace(trace, scan_id, "ocr", "missing", "No computed OCR text available for Rabbithole")
        return result

    _trace(trace, scan_id, "ocr", "state_hit", "Using computed OCR boxes from panel state", computed_regions=computed_count)
    rabbithole_result, rabbithole_key, should_cache_rabbithole = _run_rabbithole_stage(ocr_result, panel_path, options, trace, scan_id)
    if rabbithole_result.get("cache_miss"):
        result = copy.deepcopy(ocr_result)
        result["cache_miss"] = True
        result["scan_trace"] = trace
        result["panel_cache"] = _panel_cache_status(state, panel_path)
        return result
    if rabbithole_key and should_cache_rabbithole and rabbithole_result.get("success"):
        _save_persisted_stage(panel_path, "rabbithole", rabbithole_key, rabbithole_result)
    enriched = _merge_rabbithole_result(ocr_result, rabbithole_result)
    _trace(trace, scan_id, "rabbithole", "done", "Rabbithole completed", annotations=len(enriched.get("annotations", []) or []))
    enriched["scan_trace"] = trace
    enriched["panel_cache"] = _panel_cache_status(state, panel_path)
    _save_state(panel_path, state)
    return enriched


def _prune_rabbithole_jobs(now: float | None = None) -> None:
    now = now or time.time()
    with _RABBITHOLE_JOBS_LOCK:
        stale = [
            job_id
            for job_id, job in _RABBITHOLE_JOBS.items()
            if job.get("status") in {"done", "error"} and now - float(job.get("updated_at", now)) > RABBITHOLE_JOB_TTL_SECONDS
        ]
        for job_id in stale:
            _RABBITHOLE_JOBS.pop(job_id, None)


def _public_rabbithole_job(job: dict | None) -> dict | None:
    if not job:
        return None
    payload = {
        "success": True,
        "rabbithole_job": True,
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "filename": job.get("filename"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }
    if job.get("error"):
        payload["error"] = job.get("error")
    if job.get("result") is not None:
        payload["result"] = job.get("result")
    return payload


def _update_rabbithole_job(job_id: str, **updates) -> dict | None:
    with _RABBITHOLE_JOBS_LOCK:
        job = _RABBITHOLE_JOBS.get(job_id)
        if not job:
            return None
        job.update(updates)
        job["updated_at"] = round(time.time(), 3)
        return copy.deepcopy(job)


def _run_rabbithole_job(job_id: str, panel_path: Path, options: dict) -> None:
    _update_rabbithole_job(job_id, status="running")
    try:
        result = _run_rabbithole_existing(panel_path, options)
        if result and result.get("success"):
            _update_rabbithole_job(job_id, status="done", result=result, error=None)
        else:
            error = (result or {}).get("error") or (result or {}).get("rabbithole_error") or "Rabbithole processing failed"
            _update_rabbithole_job(job_id, status="error", result=result, error=error)
    except Exception as exc:
        logger.exception('panel="%s" stage=rabbithole status=failed job_id=%s msg="Rabbithole background job failed"', panel_path.name, job_id)
        _update_rabbithole_job(job_id, status="error", error=str(exc))


def _start_rabbithole_job(filename: str, panel_path: Path, options: dict) -> dict:
    _prune_rabbithole_jobs()
    job_id = uuid.uuid4().hex[:12]
    now = round(time.time(), 3)
    job = {
        "job_id": job_id,
        "filename": filename,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "result": None,
        "error": None,
    }
    with _RABBITHOLE_JOBS_LOCK:
        _RABBITHOLE_JOBS[job_id] = job
    _RABBITHOLE_JOB_EXECUTOR.submit(_run_rabbithole_job, job_id, panel_path, copy.deepcopy(options))
    logger.info(
        'panel="%s" stage=rabbithole status=queued job_id=%s msg="Rabbithole background job queued"',
        filename,
        job_id,
    )
    return copy.deepcopy(job)


def _run_render_stage(panel_path: Path, enriched: dict, trace: list[dict], scan_id: str) -> dict:
    render_start = time.perf_counter()
    render_result = render_translated_panel(panel_path, enriched)
    _trace(
        trace,
        scan_id,
        "render",
        "done",
        "Panel render stage completed",
        render_start,
        translated_image=bool(render_result.get("translated_image_url")),
    )
    return render_result


def _run_translate_existing(panel_path: Path, options: dict | None = None) -> dict:
    """Translate/render the OCR result produced by a previous scan."""
    options = options or {}
    scan_id = _new_scan_id()
    trace: list[dict] = []
    state = _load_state(panel_path)
    _trace(trace, scan_id, "translation", "start", "Translate started", panel=panel_path.name)

    ocr_result = _build_ocr_result_from_state(panel_path, state, options)
    computed_count = sum(1 for ann in ocr_result.get("annotations", []) if ann.get("computed") and ann.get("text"))
    if not computed_count:
        result = {
            "success": False,
            "error": "Run Scan before Translate. OCR cache for the current OCR settings is missing.",
            "text": "",
            "annotations": ocr_result.get("annotations", []),
            "scan_trace": trace,
            "panel_cache": _panel_cache_status(state, panel_path),
        }
        _trace(trace, scan_id, "ocr", "missing", "No computed OCR text available for translation")
        return result

    _trace(trace, scan_id, "ocr", "state_hit", "Using computed OCR boxes from panel state", computed_regions=computed_count)
    translation_result, translation_key, should_cache_translation = _run_translation_stage(
        ocr_result,
        options,
        {"panel_path": panel_path},
        trace,
        scan_id,
    )
    if translation_result.get("cache_miss"):
        result = copy.deepcopy(ocr_result)
        result["cache_miss"] = True
        result["scan_trace"] = trace
        result["panel_cache"] = _panel_cache_status(state, panel_path)
        return result
    if translation_key and should_cache_translation and not translation_result.get("translation_error"):
        _save_persisted_stage(panel_path, "translation", translation_key, translation_result)

    enriched = _merge_translation_result(ocr_result, translation_result, options)
    enriched = _attach_cached_rabbithole(panel_path, enriched)
    render_result = _run_render_stage(panel_path, enriched, trace, scan_id)
    enriched.update(render_result)
    _trace(trace, scan_id, "translation", "done", "Translate completed", annotations=len(enriched.get("annotations", []) or []))
    enriched["scan_trace"] = trace
    enriched["panel_cache"] = _panel_cache_status(state, panel_path)
    _save_state(panel_path, state)
    return enriched


class ScanOptions(BaseModel):
    use_cache: bool = True
    cache_only: bool = False
    fresh: bool = False
    detection: dict = Field(default_factory=dict)
    ocr: dict = Field(default_factory=dict)
    bubble: dict = Field(default_factory=dict)
    render: dict = Field(default_factory=dict)
    translation: dict = Field(default_factory=dict)
    ocr_quality_mode: str = "balanced"
    semantic_rerank: str = "close"
    vertical_preference: str = "normal"
    rotation_win_margin: int = 15
    preprocessing_set: str = "standard"
    confidence_threshold: float = 0.4
    nms_threshold: float = 0.35
    mask_threshold: float = 0.3
    box_threshold: float = 0.6
    max_regions: int | None = None
    crop_upscale: int = 3
    crop_padding_ratio: float = 0.05
    crop_padding_min: int = 4
    enable_rotated_variants: bool = True
    translation_engine: str = "ollama"
    translation_model: str | None = None
    target_lang: str = "en"
    translation_style: str = "natural"
    temperature: float = 0.1
    reset_manual_edits: bool = False


def _flatten_scan_options(options: dict | None) -> dict:
    """Accept the new grouped settings payload while keeping old flat callers working."""
    raw = dict(options or {})
    grouped = {
        "detection": raw.pop("detection", {}) or {},
        "ocr": raw.pop("ocr", {}) or {},
        "bubble": raw.pop("bubble", {}) or {},
        "render": raw.pop("render", {}) or {},
        "translation": raw.pop("translation", {}) or {},
    }
    aliases = {
        "detection": {
            "confidence_threshold": "confidence_threshold",
            "nms_threshold": "nms_threshold",
            "mask_threshold": "mask_threshold",
            "box_threshold": "box_threshold",
            "max_regions": "max_regions",
        },
        "ocr": {
            "quality_mode": "ocr_quality_mode",
            "semantic_rerank": "semantic_rerank",
            "vertical_preference": "vertical_preference",
            "rotation_win_margin": "rotation_win_margin",
            "preprocessing_set": "preprocessing_set",
            "crop_upscale": "crop_upscale",
            "crop_padding_ratio": "crop_padding_ratio",
            "crop_padding_min": "crop_padding_min",
            "enable_rotated_variants": "enable_rotated_variants",
        },
        "translation": {
            "engine": "translation_engine",
            "model": "translation_model",
            "target_lang": "target_lang",
            "style": "translation_style",
            "temperature": "temperature",
        },
        "bubble": {
            "mode": "mode",
            "model_confidence": "model_confidence",
            "model_iou": "model_iou",
            "search_scale": "search_scale",
            "wand_enabled": "wand_enabled",
            "overlap_reconciliation": "overlap_reconciliation",
        },
    }
    for group_name, group_aliases in aliases.items():
        group = grouped[group_name]
        if not isinstance(group, dict):
            continue
        for source, target in group_aliases.items():
            if source in group and group[source] is not None:
                raw[target] = group[source]
    raw.update({name: value for name, value in grouped.items() if isinstance(value, dict)})
    raw["ocr_engine"] = "mangaocr"
    return raw


def _scan_options_dict(options: ScanOptions | None) -> dict:
    return _flatten_scan_options(options.model_dump() if options else {})


class RegionOverrideRequest(BaseModel):
    orientation: str | None = None
    x: int | None = None
    y: int | None = None
    width: int | None = None
    height: int | None = None
    vertical: bool | None = None


class RegionCreateRequest(BaseModel):
    x: int
    y: int
    width: int
    height: int
    orientation: str = "vertical"


def _ocr_engine_status() -> list[dict]:
    return [{
        "id": "mangaocr",
        "label": "MangaOCR",
        "available": manga_ocr_service.is_available(),
        "default": True,
    }]


@router.get("/panels")
async def list_panels():
    return ImageService.get_all_panels()


@router.post("/upload")
async def upload_panel(file: UploadFile = File(...)):
    try:
        if file.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(status_code=400, detail="Only JPEG and PNG uploads are supported")
        content = await file.read()
        result = ImageService.save_uploaded_panel(content, file.filename)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Upload failed"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception('component=upload status=failed msg="Upload failed"')
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{filename}/ocr")
async def scan_panel(filename: str, options: ScanOptions | None = None):
    """OCR - Text extrahieren mit Bounding Boxes"""
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")

    try:
        # Run blocking OCR in thread pool to avoid blocking the event loop
        opts = _scan_options_dict(options)
        result = await asyncio.get_event_loop().run_in_executor(None, _run_ocr, panel_path, opts)
    except Exception as e:
        logger.exception('panel="%s" stage=ocr status=failed msg="OCR request failed"', filename)
        raise HTTPException(status_code=500, detail=f"OCR processing error: {e}")

    if result and result.get("success"):
        return result
    error = (result or {}).get("error", "OCR engine unavailable")
    raise HTTPException(
        status_code=503,
        detail=f"OCR unavailable: {error}. Check Runtime in settings and download OCR assets if needed.",
    )


@router.post("/{filename}/rabbithole")
async def build_rabbithole(filename: str, options: ScanOptions | None = None):
    """Queue Rabbithole data generation for the latest OCR result."""
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")

    opts = _scan_options_dict(options)
    if not opts.get("cache_only"):
        job = _start_rabbithole_job(filename, panel_path, opts)
        return _public_rabbithole_job(job)

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run_rabbithole_existing, panel_path, opts)
    except Exception as e:
        logger.exception('panel="%s" stage=rabbithole status=failed msg="Rabbithole request failed"', filename)
        raise HTTPException(status_code=500, detail=f"Rabbithole processing error: {e}")

    if result and result.get("success"):
        return result
    raise HTTPException(status_code=409, detail=(result or {}).get("error", "Run Scan before Rabbithole"))


@router.get("/{filename}/rabbithole/jobs/{job_id}")
async def get_rabbithole_job(filename: str, job_id: str):
    """Return the current status or completed result for a Rabbithole job."""
    _prune_rabbithole_jobs()
    with _RABBITHOLE_JOBS_LOCK:
        job = copy.deepcopy(_RABBITHOLE_JOBS.get(job_id))
    if not job or job.get("filename") != filename:
        raise HTTPException(status_code=404, detail="Rabbithole job not found")
    return _public_rabbithole_job(job)


@router.post("/{filename}/translate")
async def translate_panel(filename: str, options: ScanOptions | None = None):
    """Translate/render the latest OCR result for the current OCR settings."""
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")

    try:
        opts = _scan_options_dict(options)
        result = await asyncio.get_event_loop().run_in_executor(None, _run_translate_existing, panel_path, opts)
    except Exception as e:
        logger.exception('panel="%s" stage=translation status=failed msg="Translate request failed"', filename)
        raise HTTPException(status_code=500, detail=f"Translation processing error: {e}")

    if result and result.get("success"):
        return result
    raise HTTPException(status_code=409, detail=(result or {}).get("error", "Run Scan before Translate"))


@router.get("/{filename}/cache-status")
async def cache_status(filename: str):
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")
    state = _load_state(panel_path)
    return _panel_cache_status(state, panel_path)


@router.delete("/{filename}/cache")
async def delete_cache(filename: str, kind: str | None = None):
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")
    state = _load_state(panel_path)
    kinds = [kind] if kind else None
    cleared = _clear_panel_cache_buckets(state, kinds)
    if kind is None:
        _delete_panel_ocr_tree(panel_path)
        _delete_stage_artifacts(panel_path, "rabbithole")
        _delete_stage_artifacts(panel_path, "translation")
        _delete_rendered_output(panel_path)
        legacy = LEGACY_OCR_CACHE_DIR / f"{_cache_key(panel_path)}.json"
        if legacy.exists():
            legacy.unlink()
    elif kind == "ocr":
        _delete_panel_ocr_artifacts(panel_path)
        legacy = LEGACY_OCR_CACHE_DIR / f"{_cache_key(panel_path)}.json"
        if legacy.exists():
            legacy.unlink()
    elif kind == "rabbithole":
        _delete_stage_artifacts(panel_path, "rabbithole")
    elif kind == "translation":
        _delete_stage_artifacts(panel_path, "translation")
        _delete_rendered_output(panel_path)
    _save_state(panel_path, state)
    status = _panel_cache_status(state, panel_path)
    return {"success": True, "cleared": cleared, "overrides_preserved": kind is not None, **status}


@router.get("/translation-engines")
async def translation_engines():
    return {"engines": translation_engine.engine_status()}


@router.get("/ollama/models")
async def ollama_models():
    return translation_engine.ollama_model_discovery_status()


@router.get("/{filename}/regions")
async def list_regions(filename: str):
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")
    state = _load_state(panel_path)
    payload = _regions_payload(panel_path, state)
    _save_state(panel_path, state)
    return payload


@router.post("/{filename}/regions/{region_id}/override")
async def override_region(filename: str, region_id: str, req: RegionOverrideRequest):
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")
    state = _load_state(panel_path)
    boxes = _promote_display_boxes_to_manual(state)
    patch = next((box for box in boxes if box.get("region_id") == region_id), None)
    if not patch:
        raise HTTPException(status_code=404, detail="Region not found")
    for key in ("x", "y", "width", "height", "vertical"):
        value = getattr(req, key)
        if value is not None:
            patch[key] = value
    if req.orientation in {"vertical", "horizontal"}:
        patch["forced_orientation"] = req.orientation
        patch["vertical"] = req.orientation == "vertical"
    normalized = _normalize_box(patch, source=patch.get("source"), computed=False)
    boxes[:] = [normalized if box.get("region_id") == region_id else box for box in boxes]
    state.setdefault("annotations", {}).pop(region_id, None)
    _clear_cached_results(state, ["ocr"])
    _delete_stage_artifacts(panel_path, "rabbithole")
    _delete_stage_artifacts(panel_path, "translation")
    _delete_rendered_output(panel_path)
    _save_state(panel_path, state)
    return {"success": True, "region_id": region_id, "region": normalized, **_regions_payload(panel_path, state)}


@router.post("/{filename}/regions")
async def add_region(filename: str, req: RegionCreateRequest):
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")
    state = _load_state(panel_path)
    region = {
        "x": req.x,
        "y": req.y,
        "width": req.width,
        "height": req.height,
        "vertical": req.orientation == "vertical",
        "forced_orientation": req.orientation,
        "font_size": 0,
        "angle": 0,
        "lines": [],
        "source": "manual",
        "computed": False,
    }
    region["region_id"] = _region_id(region)
    boxes = _promote_display_boxes_to_manual(state)
    boxes.append(_normalize_box(region, source="manual", computed=False))
    _clear_cached_results(state, ["ocr"])
    _delete_stage_artifacts(panel_path, "rabbithole")
    _delete_stage_artifacts(panel_path, "translation")
    _delete_rendered_output(panel_path)
    _save_state(panel_path, state)
    return {"success": True, "region": region, **_regions_payload(panel_path, state)}


@router.delete("/{filename}/regions/{region_id}")
async def remove_region(filename: str, region_id: str):
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")
    state = _load_state(panel_path)
    boxes = _promote_display_boxes_to_manual(state)
    state["boxes"] = [box for box in boxes if box.get("region_id") != region_id]
    state.setdefault("annotations", {}).pop(region_id, None)
    _clear_cached_results(state, ["ocr"])
    _delete_stage_artifacts(panel_path, "rabbithole")
    _delete_stage_artifacts(panel_path, "translation")
    _delete_rendered_output(panel_path)
    _save_state(panel_path, state)
    return {"success": True, "region_id": region_id, **_regions_payload(panel_path, state)}


@router.post("/{filename}/regions/{region_id}/recompute")
async def recompute_region(filename: str, region_id: str, options: ScanOptions | None = None):
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")
    opts = _scan_options_dict(options)
    result = await asyncio.get_event_loop().run_in_executor(None, _run_ocr, panel_path, opts)
    if result and result.get("success"):
        annotation = next((a for a in result.get("annotations", []) if a.get("region_id") == region_id), None)
        return {"success": True, "annotation": annotation, "result": result}
    raise HTTPException(status_code=503, detail=(result or {}).get("error", "Recompute failed"))
