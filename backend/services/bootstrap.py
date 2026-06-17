"""Runtime health checks and explicit setup helpers."""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

from config import OLLAMA_TEXT_MODEL
import services.translation.engine as translation_engine
from services.detection.region_detector import _MODEL_PATH as TEXT_REGION_MODEL_PATH
from services.detection.region_detector import _ensure_model as ensure_text_region_model

logger = logging.getLogger(__name__)

MANGA_OCR_REPO_ID = "kha-white/manga-ocr-base"


def _component(available: bool, *, status: str, error: str | None = None, **extra) -> dict[str, Any]:
    return {
        "available": bool(available),
        "status": status,
        "error": error,
        **extra,
    }


def _manga_ocr_package_available() -> bool:
    return importlib.util.find_spec("manga_ocr") is not None


def _manga_ocr_cache_status() -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(MANGA_OCR_REPO_ID, local_files_only=True)
        return _component(True, status="ready", repo_id=MANGA_OCR_REPO_ID, cache_path=path)
    except Exception as exc:
        return _component(False, status="missing", repo_id=MANGA_OCR_REPO_ID, error=str(exc))


def _detector_status() -> dict[str, Any]:
    if TEXT_REGION_MODEL_PATH.exists():
        return _component(
            True,
            status="ready",
            path=str(TEXT_REGION_MODEL_PATH),
            size_bytes=TEXT_REGION_MODEL_PATH.stat().st_size,
        )
    return _component(False, status="missing", path=str(TEXT_REGION_MODEL_PATH), error="Detector ONNX model is not present")


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
    cache = _manga_ocr_cache_status() if package_available else _component(
        False,
        status="blocked",
        repo_id=MANGA_OCR_REPO_ID,
        error="MangaOCR package is missing",
    )
    detector = _detector_status()
    ollama = _ollama_status()

    if not package.get("available"):
        warnings.append("MangaOCR Python package is missing")
    if not cache.get("available"):
        warnings.append("MangaOCR model cache is missing")
    if not detector.get("available"):
        warnings.append("Text detector model is missing")
    if not ollama.get("available"):
        warnings.append("Ollama translation model is not ready")

    ocr_ready = bool(package.get("available") and cache.get("available") and detector.get("available"))
    status = {
        "success": True,
        "ocr": {
            "ready": ocr_ready,
            "package": package,
            "mangaocr_cache": cache,
            "detector": detector,
        },
        "ollama": ollama,
        "warnings": warnings,
    }
    logger.info(
        'component=runtime status=checked ocr_ready=%s ollama_ready=%s warnings=%s msg="Runtime status checked"',
        ocr_ready,
        ollama.get("available"),
        len(warnings),
    )
    return status


def ensure_ocr_assets() -> dict[str, Any]:
    """Download/check OCR assets only. This never pulls Ollama models."""
    setup = {"success": True, "actions": [], "errors": []}
    logger.info('component=runtime action=download_ocr_assets status=start msg="OCR asset setup started"')

    if not _manga_ocr_package_available():
        setup["success"] = False
        setup["errors"].append("Python package manga-ocr is not installed")
    else:
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(MANGA_OCR_REPO_ID)
            setup["actions"].append({"component": "mangaocr_cache", "status": "ready", "path": path})
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
