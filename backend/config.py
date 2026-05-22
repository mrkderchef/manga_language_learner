"""Central application settings and filesystem paths.

All runtime environment access should flow through this module so the backend
has one place for defaults, paths, and external service configuration.
"""

from __future__ import annotations

import os
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
	return os.getenv(name, default)


def _env_configured(name: str, default: str = "", placeholders: set[str] | None = None) -> str:
	value = _env(name, default).strip()
	if placeholders and value.lower() in {placeholder.lower() for placeholder in placeholders}:
		return ""
	return value


def _env_int(name: str, default: int) -> int:
	value = os.getenv(name)
	if value is None or value.strip() == "":
		return default
	try:
		return int(value)
	except ValueError:
		return default


@dataclass(frozen=True)
class Paths:
	base_dir: Path = PROJECT_ROOT
	panels_dir: Path = PROJECT_ROOT / "panels"
	uploads_dir: Path = PROJECT_ROOT / "panels" / "uploads"
	lookup_cache_dir: Path = PROJECT_ROOT / "backend" / "data" / "lookup_cache"
	thumbs_dir: Path = PROJECT_ROOT / "backend" / "data" / "thumbs"


@dataclass(frozen=True)
class Services:
	google_application_credentials: str = _env_configured("GOOGLE_APPLICATION_CREDENTIALS", "", {"path/to/your/credentials.json"})
	google_project_id: str = _env_configured("GOOGLE_PROJECT_ID", "", {"your-project-id"})
	gemini_api_key: str = _env_configured("GEMINI_API_KEY", "", {"your-gemini-api-key"})
	ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
	ollama_text_model: str = _env("OLLAMA_TEXT_MODEL", "hf.co/sugoitoolkit/Sugoi-14B-Ultra-GGUF:Q4_K_M")
	kanjiapi_base_url: str = _env("KANJIAPI_BASE_URL", "https://kanjiapi.dev/v1").rstrip("/")
	render_font_path: str = _env("RENDER_FONT_PATH", "")


@dataclass(frozen=True)
class ApiSettings:
	host: str = _env("API_HOST", "0.0.0.0")
	port: int = _env_int("API_PORT", 8000)
	title: str = "Manga Language Learner API"
	version: str = "0.1.0"


@dataclass(frozen=True)
class FileSettings:
	max_file_size: int = 10 * 1024 * 1024
	allowed_extensions: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(frozen=True)
class LanguageSettings:
	ocr_language_hints: tuple[str, ...] = ("ja",)
	translation_source_language: str = "ja"
	translation_target_language: str = "en"


PATHS = Paths()
SERVICES = Services()
API = ApiSettings()
FILES = FileSettings()
LANGUAGE = LanguageSettings()

# Root directories for data organization
DATA_DIR = PROJECT_ROOT / "backend" / "data"
PANEL_DATA_DIR = DATA_DIR / "panel_data"

for path in (
	PATHS.panels_dir,
	PATHS.uploads_dir,
	PATHS.lookup_cache_dir,
	PATHS.thumbs_dir,
	PANEL_DATA_DIR,
):
	path.mkdir(parents=True, exist_ok=True)

# Backward-compatible flat exports used across the codebase.
BASE_DIR = PATHS.base_dir
PANELS_DIR = PATHS.panels_dir
UPLOADS_DIR = PATHS.uploads_dir
LOOKUP_CACHE_DIR = PATHS.lookup_cache_dir
THUMBS_DIR = PATHS.thumbs_dir


def _safe_fs_component(value: str) -> str:
	value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
	return value.strip("._") or "panel"


def ocr_panel_slug(panel_path: Path) -> str:
	try:
		relative_path = panel_path.relative_to(PROJECT_ROOT).as_posix()
	except ValueError:
		relative_path = panel_path.as_posix()
	base_name = _safe_fs_component(panel_path.stem or panel_path.name)
	digest = hashlib.md5(relative_path.encode("utf-8")).hexdigest()[:8]
	return f"{base_name}-{digest}"


def ocr_panel_dir(panel_path: Path) -> Path:
	return panel_ocr_dir(panel_path)


def ocr_panel_state_dir(panel_path: Path) -> Path:
	return ocr_panel_dir(panel_path) / "state"


def ocr_panel_cache_dir(panel_path: Path) -> Path:
	return ocr_panel_dir(panel_path) / "cache"


def ocr_panel_debug_dir(panel_path: Path) -> Path:
	return ocr_panel_dir(panel_path) / "debug"


# ===== Panel-Centric Directory Structure (Target Architecture) =====
# Each panel gets a directory structure:
# backend/data/panel_data/<panel_id>/{ocr, translations, rendered, rabbithole, metadata.json}
# This provides relational traceability: all artifacts for a panel are co-located under its ID.
#
# Migration path:
# - Current: backend/data/ocr/<panel_id>/{state,debug}, backend/data/rendered_panels/, etc. (scattered)
# - Target: backend/data/panel_data/<panel_id>/{ocr/{state,debug}, translations/, rendered/, rabbithole/, metadata.json}
# - Global: backend/data/lookup_cache/ (NLP cache, panel-agnostic)


def panel_data_dir(panel_path: Path) -> Path:
	"""Root directory for all panel-specific data: backend/data/panel_data/<panel_id>/"""
	panel_id = ocr_panel_slug(panel_path)
	return PANEL_DATA_DIR / panel_id


def panel_ocr_dir(panel_path: Path) -> Path:
	"""OCR data for panel: backend/data/panel_data/<panel_id>/ocr/"""
	return panel_data_dir(panel_path) / "ocr"


def panel_ocr_state_dir(panel_path: Path) -> Path:
	"""OCR state and cache for panel: backend/data/panel_data/<panel_id>/ocr/state/"""
	return panel_ocr_dir(panel_path) / "state"


def panel_ocr_debug_dir(panel_path: Path) -> Path:
	"""OCR debug images for panel: backend/data/panel_data/<panel_id>/ocr/debug/"""
	return panel_ocr_dir(panel_path) / "debug"


def panel_ocr_cache_dir(panel_path: Path) -> Path:
	"""OCR intermediate cache for panel: backend/data/panel_data/<panel_id>/ocr/cache/"""
	return panel_ocr_dir(panel_path) / "cache"


def panel_translations_dir(panel_path: Path) -> Path:
	"""Translation cache and snapshots: backend/data/panel_data/<panel_id>/translations/"""
	return panel_data_dir(panel_path) / "translations"


def panel_rendered_dir(panel_path: Path) -> Path:
	"""Rendered output and history: backend/data/panel_data/<panel_id>/rendered/"""
	return panel_data_dir(panel_path) / "rendered"


def panel_rabbithole_dir(panel_path: Path) -> Path:
	"""Rabbithole metadata: backend/data/panel_data/<panel_id>/rabbithole/"""
	return panel_data_dir(panel_path) / "rabbithole"


def panel_metadata_path(panel_path: Path) -> Path:
	"""Panel metadata file: backend/data/panel_data/<panel_id>/metadata.json"""
	return panel_data_dir(panel_path) / "metadata.json"


# Note: The following directories remain global (not per-panel):
# - backend/data/lookup_cache/ ... NLP cache entries (kanji, words, readings)
# - backend/data/lookup_cache/kanji/, lookup/, readings/, words/ ... keyed by content, not panel

GOOGLE_APPLICATION_CREDENTIALS = SERVICES.google_application_credentials
GOOGLE_PROJECT_ID = SERVICES.google_project_id
GEMINI_API_KEY = SERVICES.gemini_api_key
OLLAMA_BASE_URL = SERVICES.ollama_base_url
OLLAMA_TEXT_MODEL = SERVICES.ollama_text_model
KANJIAPI_BASE_URL = SERVICES.kanjiapi_base_url
RENDER_FONT_PATH = SERVICES.render_font_path

API_HOST = API.host
API_PORT = API.port
API_TITLE = API.title
API_VERSION = API.version

MAX_FILE_SIZE = FILES.max_file_size
ALLOWED_EXTENSIONS = FILES.allowed_extensions

OCR_LANGUAGE_HINTS = LANGUAGE.ocr_language_hints
TRANSLATION_SOURCE_LANGUAGE = LANGUAGE.translation_source_language
TRANSLATION_TARGET_LANGUAGE = LANGUAGE.translation_target_language
