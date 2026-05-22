"""Startup helpers for model and service availability checks."""

from __future__ import annotations

import logging
from typing import Any

import requests
from huggingface_hub import snapshot_download

from config import OLLAMA_BASE_URL, OLLAMA_TEXT_MODEL
import services.recognition.mangaocr as manga_ocr_service
import services.translation.engine as translation_engine
from services.detection.region_detector import _ensure_model as ensure_text_region_model

logger = logging.getLogger(__name__)

MANGA_OCR_REPO_ID = "kha-white/manga-ocr-base"


def ensure_runtime_assets() -> dict[str, Any]:
    """Ensure the detector model, MangaOCR cache, and Ollama models are ready."""
    manga_ocr_installed = manga_ocr_service.is_available()
    manga_ocr_cache_present = ensure_manga_ocr_cache() if manga_ocr_installed else False

    status = {
        "text_region_detector_ready": False,
        "manga_ocr_installed": manga_ocr_installed,
        "manga_ocr_cached": manga_ocr_cache_present,
        "manga_ocr_ready": manga_ocr_installed and manga_ocr_cache_present,
        "ollama_models_ready": False,
        "pulled_models": [],
    }

    ensure_text_region_model()
    status["text_region_detector_ready"] = True

    pulled = ensure_ollama_models([OLLAMA_TEXT_MODEL])
    status["pulled_models"] = pulled
    current_models = set(translation_engine.list_ollama_models(force=True))
    status["ollama_models_ready"] = OLLAMA_TEXT_MODEL in current_models
    return status


def ensure_manga_ocr_cache() -> bool:
    """Verify the MangaOCR HF snapshot is cached and download it if needed."""
    try:
        snapshot_download(MANGA_OCR_REPO_ID, local_files_only=True)
        logger.info("MangaOCR snapshot already cached: %s", MANGA_OCR_REPO_ID)
    except Exception:
        logger.info("Downloading MangaOCR snapshot: %s", MANGA_OCR_REPO_ID)
        snapshot_download(MANGA_OCR_REPO_ID)
    return True


def ensure_ollama_models(models: list[str]) -> list[str]:
    """Pull any missing Ollama models through the configured Ollama server."""
    available = set(translation_engine.list_ollama_models(force=True))
    missing = [model for model in models if model and model not in available]
    if not missing:
        logger.info("Ollama models already available: %s", ", ".join(sorted(set(models))))
        return []

    pulled: list[str] = []
    for model in missing:
        if _pull_ollama_model(model):
            pulled.append(model)
    if pulled:
        logger.info("Pulled Ollama models: %s", ", ".join(pulled))
    return pulled


def _pull_ollama_model(model: str) -> bool:
    """Request a model pull from the Ollama API."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model, "model": model, "stream": False},
            timeout=None,
        )
        response.raise_for_status()
        logger.info("Requested Ollama pull for %s", model)
        return True
    except Exception as exc:
        logger.warning("Could not pull Ollama model %s: %s", model, exc)
        return False
