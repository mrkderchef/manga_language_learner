"""Runtime health checks and explicit setup helpers."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from config import BUBBLE_MODEL_PATH, MANGA_OCR_MODEL_DIR, OLLAMA_TEXT_MODEL
import services.translation.engine as translation_engine
from services.detection.region_detector import _MODEL_PATH as TEXT_REGION_MODEL_PATH
from services.detection.region_detector import _ensure_model as ensure_text_region_model
from services.model_assets import (
    BUBBLE_MODEL_REPO_ID,
    BUBBLE_MODEL_REVISION,
    BUBBLE_MODEL_SHA256,
    MANGA_OCR_REPO_ID,
    MANGA_OCR_REVISION,
    bubble_model_available,
    download_bubble_model,
    download_manga_ocr_model,
    missing_manga_ocr_files,
)

logger = logging.getLogger(__name__)


def _component(available: bool, *, status: str, error: str | None = None, **extra) -> dict[str, Any]:
    return {
        "available": bool(available),
        "status": status,
        "error": error,
        **extra,
    }


def _manga_ocr_package_available() -> bool:
    try:
        module = importlib.import_module("manga_ocr")
        return getattr(module, "MangaOcr", None) is not None
    except Exception:
        return False


def _manga_ocr_model_status() -> dict[str, Any]:
    missing = missing_manga_ocr_files()
    if not missing:
        return _component(
            True,
            status="ready",
            repo_id=MANGA_OCR_REPO_ID,
            revision=MANGA_OCR_REVISION,
            model_path=str(MANGA_OCR_MODEL_DIR),
        )
    return _component(
        False,
        status="missing",
        repo_id=MANGA_OCR_REPO_ID,
        revision=MANGA_OCR_REVISION,
        model_path=str(MANGA_OCR_MODEL_DIR),
        missing_files=missing,
        error=f"MangaOCR model is incomplete; missing: {', '.join(missing)}",
    )


def _detector_status() -> dict[str, Any]:
    if TEXT_REGION_MODEL_PATH.is_file() and TEXT_REGION_MODEL_PATH.stat().st_size > 1_000_000:
        return _component(
            True,
            status="ready",
            path=str(TEXT_REGION_MODEL_PATH),
            size_bytes=TEXT_REGION_MODEL_PATH.stat().st_size,
        )
    return _component(False, status="missing", path=str(TEXT_REGION_MODEL_PATH), error="Detector ONNX model is not present")


def _bubble_model_status() -> dict[str, Any]:
    available = bubble_model_available()
    package_error = None
    try:
        module = importlib.import_module("ultralytics")
        package_available = getattr(module, "YOLO", None) is not None
    except Exception as exc:
        package_available = False
        package_error = str(exc)
    error = None
    if not available:
        error = "Optional bubble segmentation checkpoint is not present; classical fallback will be used"
    elif not package_available:
        error = f"Ultralytics cannot be imported: {package_error or 'YOLO is unavailable'}; classical fallback will be used"
    return _component(
        available and package_available,
        status="ready" if available and package_available else "missing",
        repo_id=BUBBLE_MODEL_REPO_ID,
        revision=BUBBLE_MODEL_REVISION,
        sha256=BUBBLE_MODEL_SHA256,
        path=str(BUBBLE_MODEL_PATH),
        checkpoint_available=available,
        package_available=package_available,
        error=error,
    )


def _ollama_status() -> dict[str, Any]:
    discovery = translation_engine.ollama_model_discovery_status(force=True)
    models = discovery.get("models", []) or []
    configured_model = OLLAMA_TEXT_MODEL
    reachable = bool(discovery.get("discovery_available"))
    model_installed = bool(configured_model and configured_model in models)
    model_available = bool(models)
    preferred_model = discovery.get("preferred_model")
    return {
        "available": reachable and model_available,
        "status": "ready" if reachable and model_available else "missing",
        "reachable": reachable,
        "configured_model": configured_model,
        "model_installed": model_installed,
        "models": models,
        "preferred_model": preferred_model,
        "error": discovery.get("discovery_error"),
    }


def check_runtime_status() -> dict[str, Any]:
    """Return runtime readiness without downloading assets or pulling models."""
    warnings: list[str] = []

    package_available = _manga_ocr_package_available()
    package = _component(
        package_available,
        status="ready" if package_available else "missing",
        package="manga_ocr",
        error=None if package_available else "Python package manga-ocr is not installed",
    )
    # Report package and model independently. This is especially important on a
    # fresh machine, where assets may download successfully before an import issue
    # in the Python package has been resolved.
    model = _manga_ocr_model_status()
    detector = _detector_status()
    bubble_model = _bubble_model_status()
    ollama = _ollama_status()

    if not package.get("available"):
        warnings.append("MangaOCR Python package is missing")
    if not model.get("available"):
        warnings.append("MangaOCR model is missing")
    if not detector.get("available"):
        warnings.append("Text detector model is missing")
    if not bubble_model.get("available"):
        warnings.append("Bubble segmentation model is unavailable; classical fallback is active")
    if not ollama.get("available"):
        warnings.append("Ollama translation model is not ready")

    ocr_ready = bool(package.get("available") and model.get("available") and detector.get("available"))
    status = {
        "success": True,
        "ocr": {
            "ready": ocr_ready,
            "package": package,
            "mangaocr_model": model,
            "detector": detector,
        },
        "ollama": ollama,
        "bubble_segmentation": bubble_model,
        "warnings": warnings,
    }
    logger.info(
        'component=runtime status=checked ocr_ready=%s ollama_ready=%s warnings=%s msg="Runtime status checked"',
        ocr_ready,
        ollama.get("available"),
        len(warnings),
    )
    return status


def ensure_bubble_assets() -> dict[str, Any]:
    """Download the optional bubble model without making OCR depend on it."""
    setup = {"success": True, "actions": [], "errors": []}
    try:
        path = download_bubble_model()
        setup["actions"].append({"component": "bubble_segmentation", "status": "ready", "path": str(path)})
    except Exception as exc:
        setup["success"] = False
        setup["errors"].append(f"Bubble model download failed: {exc}")
        logger.warning('component=bubble_segmenter action=download status=failed error=%r', exc)
    status = check_runtime_status()
    status["setup"] = setup
    return status


def ensure_ocr_assets() -> dict[str, Any]:
    """Download/check OCR assets only. This never pulls Ollama models."""
    setup = {"success": True, "actions": [], "errors": []}
    logger.info('component=runtime action=download_ocr_assets status=start msg="OCR asset setup started"')

    # Model acquisition must not depend on manga-ocr already being importable.
    # A fresh install may need to download assets while package setup is repaired.
    try:
        path = download_manga_ocr_model()
        setup["actions"].append({"component": "mangaocr_model", "status": "ready", "path": str(path)})
    except Exception as exc:
        setup["success"] = False
        setup["errors"].append(f"MangaOCR model download failed: {exc}")
        logger.warning(
            'component=mangaocr action=download status=failed error=%r msg="MangaOCR model download failed"',
            exc,
        )

    try:
        ensure_text_region_model()
        setup["actions"].append({"component": "text_region_detector", "status": "ready", "path": str(TEXT_REGION_MODEL_PATH)})
    except Exception as exc:
        setup["success"] = False
        setup["errors"].append(f"Text detector model download failed: {exc}")
        logger.warning(
            'component=detector action=download status=failed error=%r msg="Text detector model download failed"',
            exc,
        )

    status = check_runtime_status()
    status["setup"] = setup
    logger.info(
        'component=runtime action=download_ocr_assets status=%s errors=%s msg="OCR asset setup finished"',
        "done" if setup["success"] else "failed",
        len(setup["errors"]),
    )
    return status


def ensure_runtime_assets() -> dict[str, Any]:
    """Compatibility alias: startup now checks only and never downloads."""
    return check_runtime_status()
