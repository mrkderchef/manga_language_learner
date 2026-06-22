"""Paths, manifests, and download helpers for backend-owned model assets."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

from config import BUBBLE_MODEL_DIR, BUBBLE_MODEL_PATH, MANGA_OCR_MODEL_DIR


MANGA_OCR_REPO_ID = "kha-white/manga-ocr-base"
MANGA_OCR_REVISION = "aa6573bd10b0d446cbf622e29c3e084914df9741"
MANGA_OCR_REQUIRED_FILES = (
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
)

BUBBLE_MODEL_REPO_ID = "ShadowB/Manga109-panel-balloon-text-yolov26-segmentation"
BUBBLE_MODEL_REVISION = "3a860269ee0beb43ce9f31d82c7851441eb178ae"
BUBBLE_MODEL_FILENAME = "best.pt"
BUBBLE_MODEL_SHA256 = "0b4376e426fa96af3976afa6a2602421dacf2dec96ef87b4a44f5e8d4971cb6f"

_MANGA_OCR_DOWNLOAD_LOCK = threading.Lock()
_BUBBLE_DOWNLOAD_LOCK = threading.Lock()


def missing_manga_ocr_files(model_dir: Path = MANGA_OCR_MODEL_DIR) -> list[str]:
    """Return required MangaOCR files that are absent from the local model directory."""
    return [
        name for name in MANGA_OCR_REQUIRED_FILES
        if not (model_dir / name).is_file() or (model_dir / name).stat().st_size == 0
    ]


def download_manga_ocr_model(model_dir: Path = MANGA_OCR_MODEL_DIR) -> Path:
    """Download the pinned MangaOCR snapshot into the backend-owned model directory."""
    with _MANGA_OCR_DOWNLOAD_LOCK:
        if not missing_manga_ocr_files(model_dir):
            return model_dir

        from huggingface_hub import snapshot_download

        model_dir.mkdir(parents=True, exist_ok=True)
        path = Path(snapshot_download(
            repo_id=MANGA_OCR_REPO_ID,
            revision=MANGA_OCR_REVISION,
            local_dir=str(model_dir),
            allow_patterns=list(MANGA_OCR_REQUIRED_FILES),
        ))
        missing = missing_manga_ocr_files(model_dir)
        if missing:
            raise RuntimeError(f"MangaOCR download is incomplete; missing: {', '.join(missing)}")
        return path


def bubble_model_available(model_path: Path = BUBBLE_MODEL_PATH) -> bool:
    if not model_path.is_file() or model_path.stat().st_size <= 1_000_000:
        return False
    digest = hashlib.sha256()
    with model_path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == BUBBLE_MODEL_SHA256


def download_bubble_model(model_dir: Path = BUBBLE_MODEL_DIR) -> Path:
    """Download the pinned optional manga balloon instance-segmentation checkpoint."""
    with _BUBBLE_DOWNLOAD_LOCK:
        target = model_dir / BUBBLE_MODEL_FILENAME
        if bubble_model_available(target):
            return target

        from huggingface_hub import hf_hub_download

        model_dir.mkdir(parents=True, exist_ok=True)
        downloaded = Path(hf_hub_download(
            repo_id=BUBBLE_MODEL_REPO_ID,
            filename=BUBBLE_MODEL_FILENAME,
            revision=BUBBLE_MODEL_REVISION,
            local_dir=str(model_dir),
            force_download=target.exists(),
        ))
        if not bubble_model_available(downloaded):
            raise RuntimeError(f"Bubble segmentation checkpoint checksum mismatch; expected {BUBBLE_MODEL_SHA256}")
        return downloaded


__all__ = [
    "MANGA_OCR_REPO_ID",
    "MANGA_OCR_REVISION",
    "MANGA_OCR_REQUIRED_FILES",
    "BUBBLE_MODEL_FILENAME",
    "BUBBLE_MODEL_REPO_ID",
    "BUBBLE_MODEL_REVISION",
    "BUBBLE_MODEL_SHA256",
    "bubble_model_available",
    "download_bubble_model",
    "download_manga_ocr_model",
    "missing_manga_ocr_files",
]
