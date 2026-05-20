from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from services.image_service import ImageService
from services.ollama_service import OllamaOCRService
from services import manga_ocr_service
from services import japanese_learning_service
from services import translation_engine
from services.panel_renderer import render_translated_panel
from services.text_region_detector import detect_text_regions
import asyncio
import copy
import hashlib
import json
import logging
import traceback
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from config import BASE_DIR, GEMINI_API_KEY

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/scanner", tags=["scanner"])

# Selectable OCR engines. No hidden fallback is used.
ollama_service = OllamaOCRService()

# OCR result cache directory
OCR_CACHE_DIR = BASE_DIR / "backend" / "data" / "ocr_cache"
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OCR_STATE_DIR = BASE_DIR / "backend" / "data" / "ocr_state"
OCR_STATE_DIR.mkdir(parents=True, exist_ok=True)
OCR_CACHE_VERSION = "ocr-stepwise-state-v1"
MISSING_TRANSLATION_TEXT = "No translation available"
CACHE_BUCKETS = ("ocr", "translation", "learning")


def _normalize_scan_result(result: dict | None) -> dict | None:
    if not result or not result.get("success"):
        return result

    normalized = dict(result)
    suppress_translation_placeholder = bool(result.get("translation_error")) or result.get("translation_engine_used") in {None, "none"}
    annotations = []
    for ann in result.get("annotations", []) or []:
        copy = dict(ann)
        translated = str(copy.get("translated") or "").strip()
        has_bbox = bool(copy.get("bbox"))
        if not has_bbox:
            copy["localization_missing"] = True
        elif translated in {"", "—", "..."}:
            copy["translated"] = "" if suppress_translation_placeholder else MISSING_TRANSLATION_TEXT
            copy["translation_missing"] = True
        annotations.append(copy)
    normalized["annotations"] = annotations
    return normalized


def _cache_key(panel_path: Path) -> str:
    """Generate a cache key from file path + modification time."""
    stat = panel_path.stat()
    raw = f"{OCR_CACHE_VERSION}:{panel_path}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()


def _state_file(panel_path: Path) -> Path:
    return OCR_STATE_DIR / f"{_cache_key(panel_path)}.json"


def _default_state() -> dict:
    return {
        "version": OCR_CACHE_VERSION,
        "overrides": {},
        "detection": [],
        "cache": {bucket: {} for bucket in CACHE_BUCKETS},
    }


def _ensure_state_shape(data: dict) -> dict:
    data.setdefault("version", OCR_CACHE_VERSION)
    data.setdefault("overrides", {})
    data.setdefault("detection", [])
    cache = data.setdefault("cache", {})
    for bucket in CACHE_BUCKETS:
        cache.setdefault(bucket, {})
    # Older builds stored one opaque final result under "derived"; keep it out
    # of the new buckets but report/clear it as legacy panel cache.
    data.setdefault("derived", {})
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
    _state_file(panel_path).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _region_id(region: dict) -> str:
    raw = f"{region.get('x')}:{region.get('y')}:{region.get('width')}:{region.get('height')}"
    return "region_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]


def _detect_regions_with_ids(panel_path: Path) -> list[dict]:
    regions = []
    for region in detect_text_regions(str(panel_path)):
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


def _json_hash(payload: dict | list | str) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _ocr_options_key(options: dict, overrides: dict) -> str:
    relevant = {
        "ocr_engine": options.get("ocr_engine", "mangaocr"),
        "ocr_quality_mode": options.get("ocr_quality_mode", "balanced"),
        "semantic_rerank": options.get("semantic_rerank", "close"),
        "vertical_preference": options.get("vertical_preference", "normal"),
        "rotation_win_margin": options.get("rotation_win_margin", 15),
        "preprocessing_set": options.get("preprocessing_set", "standard"),
        "detection_sensitivity": options.get("detection_sensitivity", "normal"),
        "overrides": overrides or {},
    }
    return _json_hash(relevant)


def _texts_hash(annotations: list[dict]) -> str:
    texts = [ann.get("text", "") for ann in annotations]
    return _json_hash(texts)


def _translation_options_key(options: dict, annotations: list[dict]) -> str:
    relevant = {
        "text_hash": _texts_hash(annotations),
        "translation_engine": options.get("translation_engine", "ollama"),
        "translation_model": options.get("translation_model") or "",
        "target_lang": options.get("target_lang", "en"),
        "translation_style": options.get("translation_style", "natural"),
        "temperature": options.get("temperature", 0.1),
        "prompt_version": translation_engine.PROMPT_VERSION,
    }
    return _json_hash(relevant)


def _learning_options_key(annotations: list[dict]) -> str:
    return _json_hash({
        "text_hash": _texts_hash(annotations),
        "learning_version": "sudachi-kanjiapi-v1",
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
    logger.info("[scan:%s] %s %s - %s %s", scan_id, stage, status, message, fields or "")
    trace.append(event)


def _panel_cache_status(state: dict, panel_path: Path) -> dict:
    cache = state.get("cache", {})
    legacy_key = _cache_key(panel_path)
    legacy_entries = len(state.get("derived", {}))
    legacy_file = OCR_CACHE_DIR / f"{legacy_key}.json"
    legacy_count = legacy_entries + int(legacy_file.exists())
    buckets = {
        bucket: {
            "has_cache": bool(cache.get(bucket)),
            "entries": len(cache.get(bucket, {})),
        }
        for bucket in CACHE_BUCKETS
    }
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
        "has_overrides": bool(state.get("overrides", {}).get("regions") or state.get("overrides", {}).get("added") or state.get("overrides", {}).get("removed")),
        "state_file": _state_file(panel_path).name,
    }


def _clear_panel_cache_buckets(state: dict, kinds: list[str] | None = None) -> list[str]:
    requested = kinds or list(CACHE_BUCKETS)
    cleared = []
    cache = state.setdefault("cache", {})
    for kind in requested:
        if kind in CACHE_BUCKETS:
            cache[kind] = {}
            cleared.append(kind)
    state["derived"] = {}
    return cleared


def _prepare_panel_regions(panel_path: Path, options: dict, state: dict, trace: list[dict], scan_id: str) -> list[dict]:
    if options.get("reset_manual_edits"):
        state["overrides"] = {}
        state["cache"] = {bucket: {} for bucket in CACHE_BUCKETS}
        state["derived"] = {}
        _trace(trace, scan_id, "overrides", "reset", "Manual edits reset by scan option")

    start = time.perf_counter()
    detected = _detect_regions_with_ids(panel_path)
    state["detection"] = detected
    regions = _apply_overrides(detected, state.get("overrides", {}))
    _trace(
        trace,
        scan_id,
        "ocr",
        "regions",
        "Detected text regions and applied manual overrides",
        start,
        detected_regions=len(detected),
        active_regions=len(regions),
        overrides=bool(state.get("overrides")),
    )
    return regions


def _strip_embedded_translation(result: dict) -> dict:
    clean = copy.deepcopy(result)
    for ann in clean.get("annotations", []) or []:
        ann["translated"] = ""
    for key in (
        "translation_engine_requested",
        "translation_engine_used",
        "translation_model",
        "translation_target_lang",
        "translation_style",
        "translation_prompt_version",
        "translation_error",
    ):
        clean.pop(key, None)
    return clean


def _run_selected_ocr_engine(panel_path: Path, options: dict, regions: list[dict], trace: list[dict], scan_id: str) -> dict:
    ocr_engine = options.get("ocr_engine", "mangaocr")
    start = time.perf_counter()
    _trace(trace, scan_id, "ocr", "start", "OCR stage started", engine=ocr_engine)

    if ocr_engine == "mangaocr":
        if not manga_ocr_service.is_available():
            raise RuntimeError("MangaOCR is not installed")
        ocr_only_options = dict(options)
        ocr_only_options["translation_engine"] = "none"
        result = manga_ocr_service.extract_and_translate(str(panel_path), options=ocr_only_options, regions_override=regions)
    elif ocr_engine == "ollama":
        if not ollama_service.is_available():
            raise RuntimeError("Ollama vision OCR is not available")
        result = ollama_service.extract_and_translate(str(panel_path), options.get("target_lang") or "en")
        result["ocr_engine_requested"] = "ollama"
        result["ocr_engine_used"] = "ollama"
        result["fallback_used"] = False
        result["fallback_reason"] = None
    elif ocr_engine == "gemini":
        from services.gemini_service import GeminiOCRService
        svc = GeminiOCRService()
        if not svc.is_available():
            raise RuntimeError("Gemini OCR is not configured")
        result = svc.extract_and_translate(str(panel_path), options.get("target_lang") or "en")
        result["ocr_engine_requested"] = "gemini"
        result["ocr_engine_used"] = "gemini"
        result["fallback_used"] = False
        result["fallback_reason"] = None
    else:
        raise RuntimeError(f"Unsupported OCR engine: {ocr_engine}")

    if not result or not result.get("success"):
        raise RuntimeError((result or {}).get("error", "OCR engine failed"))

    result = _strip_embedded_translation(_normalize_scan_result(result))
    result["ocr_engine_requested"] = ocr_engine
    result["ocr_engine_used"] = result.get("ocr_engine_used") or ocr_engine
    result["fallback_used"] = False
    result["fallback_reason"] = None
    result = _attach_learning_metadata(result)
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
    """Run or reuse OCR only. Translation and learning are separate stages."""
    options = options or {}
    use_cache = bool(options.get("use_cache", True)) and not bool(options.get("fresh", False))
    regions = _prepare_panel_regions(panel_path, options, state, trace, scan_id)
    cache_key = _ocr_options_key(options, state.get("overrides", {}))
    cache_bucket = state.setdefault("cache", {}).setdefault("ocr", {})

    if use_cache and cache_bucket.get(cache_key):
        result = copy.deepcopy(cache_bucket[cache_key])
        _trace(trace, scan_id, "ocr", "cache_hit", "OCR cache hit", cache_key=cache_key)
        return result

    if use_cache:
        _trace(trace, scan_id, "ocr", "cache_miss", "OCR cache miss", cache_key=cache_key)
    else:
        _trace(trace, scan_id, "ocr", "fresh", "OCR cache bypassed", cache_key=cache_key)

    try:
        result = _run_selected_ocr_engine(panel_path, options, regions, trace, scan_id)
        cache_bucket[cache_key] = result
        return result
    except Exception as e:
        logger.error("OCR stage exception: %s\n%s", e, traceback.format_exc())
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
    return {
        "success": error is None,
        "translations": [""] * text_count,
        "translation_engine_requested": options.get("translation_engine", "ollama"),
        "translation_engine_used": None if error else options.get("translation_engine", "ollama"),
        "translation_model": options.get("translation_model"),
        "translation_target_lang": options.get("target_lang", "en"),
        "translation_style": options.get("translation_style", "natural"),
        "translation_prompt_version": translation_engine.PROMPT_VERSION,
        "fallback_used": False,
        "translation_error": error,
    }


def _run_translation_stage(ocr_result: dict, options: dict, state: dict, trace: list[dict], scan_id: str) -> tuple[dict, str | None, bool]:
    annotations = ocr_result.get("annotations", []) or []
    texts = [ann.get("text", "") for ann in annotations]
    cache_key = _translation_options_key(options, annotations)
    cache_bucket = state.setdefault("cache", {}).setdefault("translation", {})
    use_cache = bool(options.get("use_cache", True)) and not bool(options.get("fresh", False))

    if use_cache and cache_bucket.get(cache_key):
        _trace(trace, scan_id, "translation", "cache_hit", "Translation cache hit", cache_key=cache_key)
        return copy.deepcopy(cache_bucket[cache_key]), cache_key, False

    if not texts:
        result = _empty_translation_result(0, options)
        return result, cache_key, True

    _trace(
        trace,
        scan_id,
        "translation",
        "start",
        "Translation stage started",
        engine=options.get("translation_engine", "ollama"),
        model=options.get("translation_model"),
        text_blocks=len(texts),
    )
    start = time.perf_counter()
    try:
        result = translation_engine.translate_batch(
            texts,
            target_lang=options.get("target_lang", "en"),
            engine=options.get("translation_engine", "ollama"),
            model=options.get("translation_model") or None,
            style=options.get("translation_style", "natural"),
            temperature=float(options.get("temperature", 0.1)),
        )
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
        result = _empty_translation_result(len(texts), options, str(exc))
        _trace(trace, scan_id, "translation", "error", str(exc), start, engine=options.get("translation_engine", "ollama"), model=options.get("translation_model"))
        return result, cache_key, False


def _run_learning_stage(ocr_result: dict, state: dict, trace: list[dict], scan_id: str) -> tuple[dict, str | None, bool]:
    annotations = ocr_result.get("annotations", []) or []
    cache_key = _learning_options_key(annotations)
    cache_bucket = state.setdefault("cache", {}).setdefault("learning", {})

    if cache_bucket.get(cache_key):
        _trace(trace, scan_id, "learning", "cache_hit", "Learning cache hit", cache_key=cache_key)
        return copy.deepcopy(cache_bucket[cache_key]), cache_key, False

    _trace(trace, scan_id, "learning", "start", "Learning/NLP stage started", text_blocks=len(annotations))
    start = time.perf_counter()
    try:
        result = japanese_learning_service.build_panel_learning(annotations)
        _trace(
            trace,
            scan_id,
            "learning",
            "done",
            "Learning/NLP stage completed",
            start,
            token_regions=len(result.get("by_region", {})),
            global_lookup_hits=result.get("global_lookup_hits", 0),
            global_lookup_misses=result.get("global_lookup_misses", 0),
        )
        return result, cache_key, True
    except Exception as exc:
        result = {
            "success": False,
            "by_region": {},
            "global_lookup_hits": 0,
            "global_lookup_misses": 0,
            "learning_error": str(exc),
        }
        _trace(trace, scan_id, "learning", "error", str(exc), start)
        return result, cache_key, False


def _merge_stage_results(ocr_result: dict, translation_result: dict, learning_result: dict, options: dict) -> dict:
    result = copy.deepcopy(ocr_result)
    annotations = result.get("annotations", []) or []
    translations = translation_result.get("translations", [])
    learning_by_region = learning_result.get("by_region", {}) if learning_result else {}

    for index, ann in enumerate(annotations):
        ann["translated"] = translations[index] if index < len(translations) else ""
        region_id = ann.get("region_id") or ann.get("id") or f"region_{index + 1:04d}"
        learning = learning_by_region.get(region_id)
        if learning:
            ann["reading_kana"] = learning.get("reading_kana", ann.get("reading_kana", ""))
            ann["reading_romaji"] = learning.get("reading_romaji", ann.get("reading_romaji", ""))
            ann["tokens"] = learning.get("tokens", ann.get("tokens", []))
            ann["kanji_spans"] = learning.get("kanji_spans", ann.get("kanji_spans", []))
            ann["learning"] = {
                "kanji": learning.get("kanji", []),
                "words": learning.get("words", []),
                "readings": learning.get("reading_lookups", []),
            }

    result["annotations"] = annotations
    result.update({k: v for k, v in translation_result.items() if k != "translations"})
    result["learning_success"] = bool(learning_result.get("success"))
    result["learning_source"] = learning_result.get("source")
    result["global_lookup_hits"] = learning_result.get("global_lookup_hits", 0)
    result["global_lookup_misses"] = learning_result.get("global_lookup_misses", 0)
    if learning_result.get("learning_error"):
        result["learning_error"] = learning_result["learning_error"]
    result["translation_engine_requested"] = translation_result.get("translation_engine_requested", options.get("translation_engine", "ollama"))
    return _normalize_scan_result(result)


def _attach_learning_metadata(result: dict) -> dict:
    for index, ann in enumerate(result.get("annotations", []) or [], start=1):
        ann.setdefault("region_id", ann.get("id") or f"region_{index:04d}")
        learning = japanese_learning_service.tokenize_text(ann.get("text", ""))
        ann.setdefault("reading_kana", learning.get("reading_kana", ""))
        ann.setdefault("reading_romaji", learning.get("reading_romaji", ""))
        ann.setdefault("tokens", learning.get("tokens", []))
        ann.setdefault("kanji_spans", learning.get("kanji_spans", []))
    return result


def _run_scan_translate(panel_path: Path, options: dict | None = None) -> dict:
    """Run OCR first, then translation and learning as separate cacheable stages."""
    options = options or {}
    scan_id = _new_scan_id()
    trace: list[dict] = []
    state = _load_state(panel_path)
    _trace(trace, scan_id, "scan", "start", "Scan started", panel=panel_path.name)

    ocr_result = _run_ocr_stage(panel_path, options, state, trace, scan_id)
    if not ocr_result or not ocr_result.get("success"):
        ocr_result["scan_trace"] = trace
        ocr_result["panel_cache"] = _panel_cache_status(state, panel_path)
        _save_state(panel_path, state)
        return ocr_result

    with ThreadPoolExecutor(max_workers=2) as pool:
        translation_future = pool.submit(_run_translation_stage, ocr_result, options, copy.deepcopy(state), trace, scan_id)
        learning_future = pool.submit(_run_learning_stage, ocr_result, copy.deepcopy(state), trace, scan_id)
        translation_result, translation_key, should_cache_translation = translation_future.result()
        learning_result, learning_key, should_cache_learning = learning_future.result()

    if translation_key and should_cache_translation and not translation_result.get("translation_error"):
        state.setdefault("cache", {}).setdefault("translation", {})[translation_key] = translation_result
    if learning_key and should_cache_learning and learning_result.get("success"):
        state.setdefault("cache", {}).setdefault("learning", {})[learning_key] = learning_result

    enriched = _merge_stage_results(ocr_result, translation_result, learning_result, options)
    render_start = time.perf_counter()
    render_result = render_translated_panel(panel_path, enriched)
    enriched.update(render_result)
    _trace(
        trace,
        scan_id,
        "render",
        "done",
        "Panel render stage completed",
        render_start,
        translated_image=bool(render_result.get("translated_image_url")),
    )
    _trace(trace, scan_id, "scan", "done", "Scan completed", annotations=len(enriched.get("annotations", []) or []))
    enriched["scan_trace"] = trace
    enriched["panel_cache"] = _panel_cache_status(state, panel_path)
    _save_state(panel_path, state)
    return enriched


class TranslateRequest(BaseModel):
    text: str
    engine: str = "ollama"
    model: str | None = None
    target_lang: str = "en"
    style: str = "natural"
    temperature: float = 0.1


class ScanOptions(BaseModel):
    use_cache: bool = True
    fresh: bool = False
    ocr_engine: str = "mangaocr"
    ocr_quality_mode: str = "balanced"
    semantic_rerank: str = "close"
    vertical_preference: str = "normal"
    rotation_win_margin: int = 15
    preprocessing_set: str = "standard"
    detection_sensitivity: str = "normal"
    translation_engine: str = "ollama"
    translation_model: str | None = None
    target_lang: str = "en"
    translation_style: str = "natural"
    temperature: float = 0.1
    reset_manual_edits: bool = False


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
    engines = [{
        "id": "mangaocr",
        "label": "MangaOCR",
        "available": manga_ocr_service.is_available(),
        "default": True,
    }, {
        "id": "ollama",
        "label": "Ollama Vision",
        "available": bool(translation_engine.list_ollama_models()),
        "default": False,
    }]
    engines.append({
        "id": "gemini",
        "label": "Gemini Vision",
        "available": bool(GEMINI_API_KEY),
        "default": False,
    })
    return engines


@router.get("/panels")
async def list_panels():
    return ImageService.get_all_panels()


@router.post("/upload")
async def upload_panel(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = ImageService.save_uploaded_panel(content, file.filename)
        return result
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{filename}/ocr")
async def scan_panel(filename: str, options: ScanOptions | None = None):
    """OCR - Text extrahieren mit Bounding Boxes"""
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")

    try:
        # Run blocking OCR in thread pool to avoid blocking the event loop
        opts = options.model_dump() if options else {}
        result = await asyncio.get_event_loop().run_in_executor(None, _run_ocr, panel_path, opts)
    except Exception as e:
        logger.error(f"OCR failed for {filename}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"OCR processing error: {e}")

    if result and result.get("success"):
        return result
    raise HTTPException(status_code=503, detail=(result or {}).get("error", "OCR engine unavailable"))


@router.post("/{filename}/scan-translate")
async def scan_and_translate(filename: str, options: ScanOptions | None = None):
    """OCR + translation in one step with explicit engines."""
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")

    try:
        # Run blocking OCR in thread pool to avoid blocking the event loop
        opts = options.model_dump() if options else {}
        result = await asyncio.get_event_loop().run_in_executor(None, _run_scan_translate, panel_path, opts)
    except Exception as e:
        logger.error(f"OCR+translate failed for {filename}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"OCR processing error: {e}")

    if result and result.get("success"):
        return result
    raise HTTPException(status_code=503, detail=(result or {}).get("error", "OCR engine unavailable"))


@router.post("/translate")
async def translate_text(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        return translation_engine.translate_text(
            req.text,
            engine=req.engine,
            model=req.model,
            target_lang=req.target_lang,
            style=req.style,
            temperature=req.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


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
    _save_state(panel_path, state)
    legacy = OCR_CACHE_DIR / f"{_cache_key(panel_path)}.json"
    if legacy.exists() and (kind is None or kind == "ocr"):
        legacy.unlink()
    status = _panel_cache_status(state, panel_path)
    return {"success": True, "cleared": cleared, "overrides_preserved": True, **status}


@router.get("/ocr-engines")
async def ocr_engines():
    return {"engines": _ocr_engine_status()}


@router.get("/translation-engines")
async def translation_engines():
    return {"engines": translation_engine.engine_status()}


@router.get("/ollama/models")
async def ollama_models():
    return translation_engine.ollama_model_discovery_status()


@router.post("/{filename}/regions/{region_id}/override")
async def override_region(filename: str, region_id: str, req: RegionOverrideRequest):
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")
    state = _load_state(panel_path)
    overrides = state.setdefault("overrides", {})
    regions = overrides.setdefault("regions", {})
    patch = regions.setdefault(region_id, {})
    for key in ("x", "y", "width", "height", "vertical"):
        value = getattr(req, key)
        if value is not None:
            patch[key] = value
    if req.orientation in {"vertical", "horizontal"}:
        patch["forced_orientation"] = req.orientation
        patch["vertical"] = req.orientation == "vertical"
    _clear_panel_cache_buckets(state)
    _save_state(panel_path, state)
    return {"success": True, "region_id": region_id, "override": patch}


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
    }
    region["region_id"] = _region_id(region)
    state.setdefault("overrides", {}).setdefault("added", []).append(region)
    _clear_panel_cache_buckets(state)
    _save_state(panel_path, state)
    return {"success": True, "region": region}


@router.delete("/{filename}/regions/{region_id}")
async def remove_region(filename: str, region_id: str):
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")
    state = _load_state(panel_path)
    state.setdefault("overrides", {}).setdefault("removed", [])
    if region_id not in state["overrides"]["removed"]:
        state["overrides"]["removed"].append(region_id)
    _clear_panel_cache_buckets(state)
    _save_state(panel_path, state)
    return {"success": True, "region_id": region_id}


@router.post("/{filename}/regions/{region_id}/recompute")
async def recompute_region(filename: str, region_id: str, options: ScanOptions | None = None):
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")
    opts = options.model_dump() if options else {}
    opts["fresh"] = True
    result = await asyncio.get_event_loop().run_in_executor(None, _run_scan_translate, panel_path, opts)
    if result and result.get("success"):
        annotation = next((a for a in result.get("annotations", []) if a.get("region_id") == region_id), None)
        return {"success": True, "annotation": annotation, "result": result}
    raise HTTPException(status_code=503, detail=(result or {}).get("error", "Recompute failed"))
