# Manga Language Learner

Lerne Japanisch spielerisch durch Manga-Panels. Lade ein Manga-Panel hoch, erkenne japanischen Text, lass ihn uebersetzen und nutze die erkannten Texte direkt zum Lernen.

## Features

### Scanner

```text
Panel hochladen
-> Textregionen erkennen
-> OCR pro Textbox
-> Uebersetzung anzeigen
-> Hover-Overlay im Panel
```

### Manga Text Detection Engine

Die App behandelt Text Detection als eigenes Produktmodul. Die Engine erkennt Textbloecke, Textzeilen, Orientierung und Lesereihenfolge und liefert robuste Regionen fuer die OCR-Pipeline.

Intern nutzt sie eine ONNX-basierte Manga/Text-Detection-Pipeline mit:

- **Block Detection:** Bounding Boxes fuer Textbloecke und Speech-Bubble-Text.
- **Segmentation Mask:** Pixel-Level Textbereiche fuer feinere Trennung.
- **Text Line Detection:** einzelne Zeilen/Spalten innerhalb eines Blocks.
- **Grouping:** Linien werden zu Bloecken zusammengefuehrt.
- **Orientation:** vertikal/horizontal, Winkel und Fontgroesse.
- **Merge/Split:** verstreute Linien verbinden, zu grosse Bloecke trennen.
- **Mask Refinement:** Textmasken pro Block verbessern.
- **Reading Order:** Manga-Lesereihenfolge rechts-nach-links, oben-nach-unten.

### MangaOCR Pipeline

Die OCR laeuft nicht auf der ganzen Seite, sondern auf den erkannten Textbox-Crops:

```text
Textregion
-> Crop mit Padding
-> OCR Preprocessing
-> MangaOCR
-> OCR Cleanup
-> Translation
```

Vor OCR werden Crops verbessert:

- 2x bis 4x Upscaling
- Grayscale
- CLAHE-Kontrastverbesserung
- Denoising
- Adaptive Thresholding
- Rotationsvarianten fuer vertikalen Text

### Lernmodus

Vokabeln aus gescannten Panels lernen:

- Panel auswaehlen
- erkannte Texte als Lernkarten nutzen
- Bedeutung raten
- Fortschritt dateibasiert speichern

## Architektur

```text
manga_language_learner/
|-- backend/                  Python FastAPI Backend
|   |-- app.py                App, Thumbnail-Cache, Static-File-Serving
|   |-- config.py             Pfade, API-Keys, Ollama-URL
|   |-- requirements.txt
|   |-- data/                 Lernfortschritt, OCR-Cache, Thumbnails (gitignored)
|   |-- routes/
|   |   |-- scanner.py        Upload, Panel-Liste, OCR/Translation
|   |   `-- learning.py       Lernmodus, Vokabel-Extraktion, Fortschritt
|   `-- services/
|       |-- text_region_detector.py  Manga Text Detection Engine
|       |-- manga_ocr_service.py     MangaOCR + Preprocessing + Translation
|       |-- ollama_service.py        Ollama Vision/Text Fallback
|       |-- gemini_service.py        Gemini Vision Service
|       |-- ocr_service.py           Legacy OCR Services
|       |-- translation_service.py   Google Cloud Translate Legacy
|       `-- image_service.py         Panel-Verwaltung
|-- frontend/                 Vanilla HTML/CSS/JS
|   |-- index.html
|   |-- css/
|   |   |-- main.css
|   |   |-- scanner.css
|   |   `-- learning.css
|   `-- js/
|       |-- api.js
|       |-- app.js
|       |-- scanner.js
|       `-- learning.js
|-- panels/                   Manga-Panel-Bilder
|   |-- uploads/              Benutzer-Uploads (gitignored)
|   `-- test_synthetic/       Synthetische Test-Panels
|-- setup_ollama_remote.sh    Ollama Setup fuer GPU-Rechner
`-- .env.example              Konfigurationsvorlage
```

## OCR/Translation Strategie

| Komponente | Status | Rolle |
| --- | --- | --- |
| Manga Text Detection Engine | aktiv | erkennt Textregionen, Orientierung und Lesereihenfolge |
| MangaOCR | aktiv | primaere japanische Manga-OCR pro Crop |
| OCR Preprocessing | aktiv | verbessert kleine, verrauschte oder kontrastarme Crops |
| Ollama `llama3.1:8b` | aktiv | kontextbewusste Textuebersetzung |
| Ollama `minicpm-v:8b` | Fallback | Vision-basierte OCR/Translation, falls noetig |
| Google/Gemini Services | optional/legacy | alternative Services fuer Experimente |

## Setup

### 1. Backend

```powershell
cd manga_language_learner
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r backend/requirements.txt

copy .env.example .env

cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Website:

```text
http://localhost:8000
```

### 2. Ollama lokal oder remote

Die App kann Ollama lokal oder auf einem GPU-Rechner verwenden.

Benoetigte Modelle:

```bash
ollama pull minicpm-v:8b
ollama pull llama3.1:8b
```

Remote-Setup:

```bash
ssh user@GPU_HOST
bash setup_ollama_remote.sh
```

### 3. Umgebungsvariablen

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=minicpm-v:8b

GEMINI_API_KEY=optional
GOOGLE_APPLICATION_CREDENTIALS=optional
GOOGLE_PROJECT_ID=optional
```

## API Kurzueberblick

```text
GET  /api/scanner/panels
POST /api/scanner/upload
POST /api/scanner/{filename}/ocr
POST /api/scanner/{filename}/scan-translate
POST /api/scanner/translate

GET  /api/learning/panels
GET  /api/learning/{filename}/vocab
POST /api/learning/{filename}/answer
GET  /api/learning/progress
```

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Detection:** eigene Manga Text Detection Engine mit ONNX/OpenCV DNN
- **OCR:** MangaOCR mit Crop-Preprocessing
- **Translation:** Ollama Textmodell, optional externe Services
- **Frontend:** Vanilla HTML/CSS/JS
- **Bildverarbeitung:** OpenCV, Pillow, NumPy
- **Geometry/Postprocessing:** pyclipper, shapely

## Aktueller Fokus

Die UI und Detection sind stabil genug fuer Iteration. Der groesste Hebel liegt aktuell bei:

- OCR Debug Mode
- bessere Confidence-Anzeige
- Furigana Handling
- Panel-aware Processing
- Cleanup Layer vor der Uebersetzung
