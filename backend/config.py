import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (parent of backend/)
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")

# Paths
BASE_DIR = Path(__file__).parent.parent
PANELS_DIR = BASE_DIR / "panels"
UPLOADS_DIR = PANELS_DIR / "uploads"

PANELS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Google Cloud Settings
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "")

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Ollama (remote GPU server)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://10.108.19.14:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "minicpm-v:8b")

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Manga Language Learner API"
API_VERSION = "0.1.0"

# File Settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# OCR Settings
OCR_LANGUAGE_HINTS = ["ja"]

# Translation Settings
TRANSLATION_SOURCE_LANGUAGE = "ja"
TRANSLATION_TARGET_LANGUAGE = "en"
