import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
PANELS_DIR = BASE_DIR / "backend" / "panels"
UPLOADS_DIR = PANELS_DIR / "uploads"

# Create directories if they don't exist
PANELS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Google Cloud Settings
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "")

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Manga Language Learner API"
API_VERSION = "1.0.0"

# File Settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# OCR Settings
OCR_LANGUAGE_HINTS = ["ja"]  # Japanese

# Translation Settings
TRANSLATION_SOURCE_LANGUAGE = "ja"
TRANSLATION_TARGET_LANGUAGE = "en"
