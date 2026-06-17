# Manga Language Learner

Learn Japanese playfully through manga panels. Upload a manga panel, detect Japanese text, explore Rabbithole analysis, and translate it when you want a rendered English panel.

## Core Features

- **Panel Management:** Load and list manga panels from `panels/` or `panels/uploads/`.
- **Text Detection:** Detect text regions per panel and process them as isolated OCR crops.
- **Deep Inspection:** View OCR results, translations, reading data (furigana/romanji), and debug crop-previews per region.
- **Rendering:** Automatically render translated dialogue back into panels (saved as `current.png`).
- **Reader Rabbithole:** Explore vocabulary, Kanji details, glosses, and reading lookups from the Reader side panel or click-open popups.
- **Granular Caching:** Clear OCR, translation, Rabbithole, or full panel data without destroying global NLP lookup data.

## Architecture & Data Flow

```text
manga_language_learner/
|-- backend/                  Python FastAPI Backend
|   |-- app.py                FastAPI app, validated media routes, thumbnail cache
|   |-- config.py             Central configuration, env loading, path resolutions
|   |-- data/                 Runtime data storage (gitignored)
|   |   |-- lookup_cache/     Global NLP and Kanji caches
|   |   |-- panel_data/       Isolated panel artifacts (state, OCR, renders)
|   |   `-- thumbs/           Generated image thumbnails
|   |-- routes/               HTTP endpoints (scanner, rabbithole)
|   `-- services/
|       |-- detection/         ONNX-based text region boundary detection
|       |-- recognition/       MangaOCR crop preprocessing and OCR execution
|       |-- translation/       Ollama translation backend
|       `-- rabbithole/        NLP tokenization, glossing, and dictionary lookups
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
|   |   |   `-- cache/            Panel-specific Rabbithole analysis
|   |   |-- translations/
|   |   |   `-- cache/            Panel-specific translation snapshots
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
| Reader Rabbithole | active | Builds dictionary, reading, kanji, and segmentation data from OCR text |
| MangaOCR-only OCR | active | The only OCR path; all recognition preprocessing is tailored to MangaOCR |
| Ollama Translation Service | active/default | The only translation service; runs locally or on a configured Ollama host |
| Sugoi Translation Model | recommended/default | Recommended Ollama model for manga dialogue translation |

## Reader Flow

The app is centered on `Home -> Reader`.

1. Upload or select a panel.
2. Click **Scan** to run OCR only.
3. The Reader automatically starts Rabbithole analysis after OCR, but translation remains a separate action.
4. Click OCR boxes to open multiple independent Rabbithole popups. Click the same box or the popup close button to close only that popup.
5. Click **Translate** after OCR to render the translated panel.

The settings panel exposes real detector, OCR, bubble-allocation, and translation controls, including detection thresholds, preprocessing breadth, vertical-text preference, crop scaling, bubble search scale, wand allocation, and translation options. Settings are persisted locally in the browser and can be reset to defaults.

Cache actions are intentionally split:

- **Clear OCR:** deletes backend OCR annotations/cache only. The current Reader view stays visible until reload or the next edit.
- **Clear translation:** clears translation cache and rendered output.
- **Clear Rabbithole:** clears Reader-integrated Rabbithole cache.
- **Clear Panel Data:** removes OCR state, manual boxes, Rabbithole data, translation outputs, rendered output, and panel metadata.

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
pip install -r requirements.txt

# Prepare config
cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env

# Run FastAPI server
./run_backend.sh
# or:
# cd backend
# python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload --no-access-log
```

The UI will be accessible at: `http://localhost:8000`

### 2. Configure Translation (Ollama)

Translations run through Ollama only. Sugoi is the recommended default model because the manga dialogue payload is tuned for it, but any installed Ollama text model can be selected.

Install/start Ollama locally or on a remote GPU host:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

Recommended model:

```bash
ollama pull hf.co/sugoitoolkit/Sugoi-14B-Ultra-GGUF:Q4_K_M
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
# Default Ollama translation model. Sugoi is recommended for manga dialogue.
OLLAMA_TEXT_MODEL=hf.co/sugoitoolkit/Sugoi-14B-Ultra-GGUF:Q4_K_M

# Dictionary and rendering dependencies
KANJIAPI_BASE_URL=https://kanjiapi.dev/v1
RENDER_FONT_PATH=optional/path/to/font.ttf
```

> **Note:** Startup checks runtime readiness but does not download large assets. The Reader settings panel shows MangaOCR, detector, and Ollama status. Use **Download OCR assets** there, or call `POST /api/runtime/ocr-assets/download`, to fetch the MangaOCR snapshot and detector ONNX model.

## API Overview

```text
# Scanner & Processing
GET    /api/scanner/panels
POST   /api/scanner/upload
POST   /api/scanner/{filename}/ocr
POST   /api/scanner/{filename}/rabbithole
POST   /api/scanner/{filename}/translate
GET    /api/scanner/{filename}/cache-status
GET    /api/scanner/{filename}/regions
DELETE /api/scanner/{filename}/cache?kind=ocr|translation|rabbithole
POST   /api/scanner/{filename}/regions
POST   /api/scanner/{filename}/regions/{region_id}/override
POST   /api/scanner/{filename}/regions/{region_id}/recompute
DELETE /api/scanner/{filename}/regions/{region_id}
GET    /api/scanner/translation-engines
GET    /api/scanner/ollama/models

# Runtime Health
GET    /api/runtime/status
POST   /api/runtime/ocr-assets/download

# Validated Media
GET    /api/media/panel/{filename}
GET    /api/thumb/{filename}?size=160
GET    /api/media/rendered/{panel_id}/current.png
GET    /api/media/ocr-debug/{debug_path}

# Rabbithole Lookup
GET    /api/rabbithole/lookup?text=...
GET    /api/rabbithole/kanji/{character}
GET    /api/rabbithole/word?text=...
GET    /api/rabbithole/reading/{reading}
```

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Detection & Vision:** ONNX, OpenCV DNN, Pillow, NumPy
- **OCR:** MangaOCR (with heavily optimized crop-preprocessing)
- **Geometry Processing:** Shapely, pyclipper
- **Translation:** Ollama only; Sugoi is the recommended default model
- **Frontend:** Vanilla HTML/CSS/JS with modular components

## Tests

```bash
python -m unittest discover -s tests
```

The current automated tests mock the heavy OCR/Rabbithole/translation services and verify the regular stage contract, removed legacy endpoints, media path validation, and upload rejection behavior.

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
