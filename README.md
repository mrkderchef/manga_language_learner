# Manga Language Learner

Learn Japanese playfully through manga panels. Upload a manga panel, detect Japanese text, explore Rabbithole analysis, and translate it when you want a rendered English panel.

## Core Features

- **Panel Management:** Load and list manga panels from `panels/` or `panels/uploads/`.
- **Text Detection:** Detect text regions per panel and process them as isolated OCR crops.
- **Deep Inspection:** View OCR results, translations, reading data (furigana/romanji), and debug crop-previews per region.
- **Rendering:** Automatically render translated dialogue back into panels (saved as `current.png`).
- **Rabbithole Mode:** Explore vocabulary, Kanji details, glosses, and reading lookups directly from scanned text.
- **Granular Caching:** Clear caches at the panel level without destroying global NLP lookup data.

## Architecture & Data Flow

```text
manga_language_learner/
|-- backend/                  Python FastAPI Backend
|   |-- app.py                FastAPI app, static files, thumbnail cache
|   |-- config.py             Central configuration, env loading, path resolutions
|   |-- data/                 Runtime data storage (gitignored)
|   |   |-- lookup_cache/     Global NLP and Kanji caches
|   |   |-- panel_data/       Isolated panel artifacts (state, OCR, renders)
|   |   `-- thumbs/           Generated image thumbnails
|   |-- routes/               HTTP endpoints (scanner, rabbithole)
|   `-- services/
|       |-- text_region_detector.py  ONNX-based text region boundary detection
|       |-- manga_ocr_service.py     Crop preprocessing and MangaOCR execution
|       |-- translation_engine.py    Ollama/Gemini translation backends
|       `-- rabbithole/                NLP tokenization, glossing, and dictionary lookups
|-- frontend/                 Vanilla HTML/CSS/JS frontend
|-- panels/                   Manga images (synthetic tests & uploads)
`-- .env.example              Configuration template
```

## Data Storage Layout

Panel data is stored in isolated directories to prevent cluttering the global data folder. Each panel's runtime artifacts live under `backend/data/panel_data/<panel_id>/`:

```text
backend/data/
|-- lookup_cache/                 Global cached NLP lookups
|-- panel_data/
|   |-- <panel_id>/               e.g. ch01_p042-a1b2c3d4
|   |   |-- ocr/
|   |   |   |-- state/            OCR state JSON (boxes, annotations)
|   |   |   |-- debug/            OCR crop debug previews
|   |   |   `-- cache/            Intermediary OCR cache
|   |   |-- rendered/
|   |   |   `-- current.png       Target language rendered panel
|   |   |-- rabbithole/
|   |   |   |-- cache/
|   |   |   `-- latest.json       Panel-specific Rabbithole analysis
|   |   |-- translations/
|   |   |   |-- cache/
|   |   |   `-- latest.json       Panel-specific translation snapshots
|   |   `-- metadata.json         Source tracking, timestamps, edit history
|   `-- ...
`-- thumbs/                       Panel thumbnails
```

## Pipeline Components

| Component | Status | Role |
| --- | --- | --- |
| Text Detection Engine | active | Detects text regions, orientation, and reading order |
| MangaOCR | active | Primary Japanese manga OCR (HuggingFace-based, runs locally) |
| Preprocessing | active | Crop upscaling, contrast scaling, denoising, adaptive thresholding |
| Sugoi Translation Model | active/default | Primary translation model: provides smooth, fluent dialogue translation |
| Ollama Text (llama3.1) | active | Fallback translation model |
| Gemini | optional | Highly capable remote translation fallback |

## Setup & Installation

### 1. Start the Backend

```powershell
# Setup virtual environment
cd manga_language_learner
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1
# Activate (macOS/Linux)
# source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Prepare config
cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env

# Run FastAPI server
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

The UI will be accessible at: `http://localhost:8000`

### 2. Configure Translation (Ollama)

Translations run locally via Ollama by default. You can run Ollama on your machine or connect to a remote GPU host.

Required models:

```bash
ollama pull hf.co/sugoitoolkit/Sugoi-14B-Ultra-GGUF:Q4_K_M
ollama pull llama3.1:8b  # Fallback translation model
```

If connecting to a remote host:
```bash
ssh user@GPU_HOST
bash setup_ollama_remote.sh
```

### 3. Environment Variables

Define your configuration in the `.env` file (copied from `.env.example`). The application centralizes all configuration inside `backend/config.py`.

```env
API_HOST=0.0.0.0
API_PORT=8000

OLLAMA_BASE_URL=http://127.0.0.1:11434
# Translation model (MangaOCR handles text recognition locally)
OLLAMA_TEXT_MODEL=hf.co/sugoitoolkit/Sugoi-14B-Ultra-GGUF:Q4_K_M

# Optional external integrations
GEMINI_API_KEY=optional
GOOGLE_APPLICATION_CREDENTIALS=optional
GOOGLE_PROJECT_ID=optional

# Dictionary and rendering dependencies
KANJIAPI_BASE_URL=https://kanjiapi.dev/v1
RENDER_FONT_PATH=optional/path/to/font.ttf
```

> **Note:** The backend automatically ensures that necessary model caches (like the text-detector ONNX model and the `kha-white/manga-ocr-base` HuggingFace snapshot) are present during startup.

## API Overview

```text
# Scanner & Processing
GET    /api/scanner/panels
POST   /api/scanner/upload
POST   /api/scanner/{filename}/ocr
POST   /api/scanner/{filename}/scan-translate
POST   /api/scanner/translate
GET    /api/scanner/{filename}/cache-status
POST   /api/scanner/{filename}/rabbithole
DELETE /api/scanner/{filename}/cache?kind=ocr|translation|rabbithole

# Rabbithole & Vocabulary
GET    /api/rabbithole/panels
GET    /api/rabbithole/{filename}/vocab
POST   /api/rabbithole/{filename}/answer
GET    /api/rabbithole/progress
```

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Detection & Vision:** ONNX, OpenCV DNN, Pillow, NumPy
- **OCR:** MangaOCR (with heavily optimized crop-preprocessing)
- **Geometry Processing:** Shapely, pyclipper
- **Translation:** Ollama (default), Gemini (optional)
- **Frontend:** Vanilla HTML/CSS/JS with modular components

## Dictionary Sources

- Word dictionary entries in Rabbithole come from `kanjiapi.dev` `/words` and are JMdict-backed.
- Tokenization, lemmas, and part-of-speech tags come from `SudachiPy` (`sudachidict_core`).
- Kanji metadata comes from `kanjiapi.dev` `/kanji/{character}`.
- Reading lookups come from `kanjiapi.dev` `/reading/{kana}`.
- Extended local entries are added for particles/auxiliaries, punctuation symbols, and short hiragana symbol references.

## Detector Notes

- The text-region detector lives in `backend/services/detection/region_detector.py` and follows the `comic-text-detector` pipeline.
- Current stages are: letterbox resize, YOLOv5 text blocks, UNet text mask, DBNet line extraction, grouping, mask refinement, and manga reading-order sorting.
- The raw `mask` is a coarse text-region prediction. It is useful for text localization, but it is not a speech-bubble boundary.
- The `mask_refined` output is a per-text-block cleanup pass around detected text. It is a better candidate for allocatable text area than the raw mask, but it still does not represent a stable balloon shape by itself.
- The reader debug overlay therefore treats the current detector output as a candidate text-space estimate. True bubble extraction would need an additional contour/shape pass that grows outward from the text mask and separates adjacent balloons reliably.
