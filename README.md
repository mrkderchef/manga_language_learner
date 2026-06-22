# Manga Language Learner

Manga Language Learner is a local-first reader for Japanese manga panels. It detects text regions, runs MangaOCR on panel crops, builds learner-oriented dictionary and kanji context, and renders translations back onto the panel.

## Architecture Rules

- **OCR engine: MangaOCR.** Recognition code is tailored to MangaOCR crop quality, orientation, and manga text behavior.
- **Translation service: Ollama.** Translation requests go through the local Ollama runtime.
- **Recommended model: Sugoi.** `OLLAMA_TEXT_MODEL` defaults to `hf.co/sugoitoolkit/Sugoi-14B-Ultra-GGUF:Q4_K_M`, and any installed Ollama text model can be selected.
- **Runtime data is generated.** Panel state, OCR debug crops, translation cache, rendered panels, and thumbnails live under `backend/data/` as local runtime artifacts.

## Folder Guide

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Install and start Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

Pull the recommended manga dialogue model:

```bash
ollama pull hf.co/sugoitoolkit/Sugoi-14B-Ultra-GGUF:Q4_K_M
```

Run the app:

```bash
./run_backend.sh
```

The UI is served at `http://localhost:8000`.

## Runtime Configuration

Key `.env` values:

```env
API_HOST=0.0.0.0
API_PORT=8000
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_TEXT_MODEL=hf.co/sugoitoolkit/Sugoi-14B-Ultra-GGUF:Q4_K_M
KANJIAPI_BASE_URL=https://kanjiapi.dev/v1
RENDER_FONT_PATH=optional/path/to/font.ttf
```

Startup checks runtime readiness. Asset downloads happen through the Reader runtime panel or `POST /api/runtime/ocr-assets/download`.

Backend-owned model assets are stored under `backend/models/`. The text detector is
`backend/models/comictextdetector.pt.onnx`; the explicit OCR asset download installs
the pinned MangaOCR snapshot into `backend/models/manga-ocr-base/`. OCR requests never
download model files implicitly. Ollama remains an external service and manages its own
translation-model storage.

Bubble allocation defaults to hybrid mode. The optional, revision- and checksum-pinned
balloon segmentation checkpoint is installed explicitly through the Reader runtime panel
or `POST /api/runtime/bubble-assets/download`; when absent or unusable, scans fall back to
the classical adaptive-topology allocator.

## Main API

```text
GET    /api/runtime/status
POST   /api/runtime/ocr-assets/download

GET    /api/scanner/panels
POST   /api/scanner/upload
POST   /api/scanner/{filename}/ocr
POST   /api/scanner/{filename}/rabbithole
GET    /api/scanner/{filename}/rabbithole/jobs/{job_id}
POST   /api/scanner/{filename}/translate
GET    /api/scanner/{filename}/cache-status
GET    /api/scanner/{filename}/regions
DELETE /api/scanner/{filename}/cache?kind=ocr|translation|rabbithole
GET    /api/scanner/translation-engines
GET    /api/scanner/ollama/models

GET    /api/rabbithole/lookup?text=...
GET    /api/rabbithole/kanji/{character}
GET    /api/rabbithole/word?text=...
GET    /api/rabbithole/reading/{reading}
POST   /api/runtime/bubble-assets/download

GET    /api/learning/panels
GET    /api/learning/{filename}/vocab
POST   /api/learning/{filename}/answer
GET    /api/learning/progress
```

## Tests

```bash
python -m unittest discover -s tests
```
