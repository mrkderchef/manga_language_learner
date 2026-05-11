# Manga Language Learner

Lerne Japanisch spielerisch durch Manga-Panels. Lade ein Manga-Panel hoch, erkenne den japanischen Text per OCR und lass ihn übersetzen – oder lerne Vokabeln direkt aus den Panels.

## Features

### Scanner
Panel hochladen → OCR erkennt japanischen Text (inkl. Bounding Boxes) → Übersetzung anzeigen.

### Lernmodus
Vokabeln aus gescannten Panels lernen. Wort anzeigen → Bedeutung raten → Fortschritt tracken (dateibasiert).

## Architektur

```
manga_language_learner/
├── backend/                  Python (FastAPI) Backend
│   ├── app.py                Hauptanwendung, Thumbnail-Cache, Static-File-Serving
│   ├── config.py             Zentrale Konfiguration (Pfade, API-Keys, Ollama-URL)
│   ├── requirements.txt
│   ├── data/                 Lernfortschritt, OCR-Cache, Thumbnails (gitignored)
│   ├── routes/
│   │   ├── scanner.py        POST /api/scanner/upload, GET /api/scanner/panels, OCR-Cache
│   │   └── learning.py       GET /api/learning/panels, Vokabel-Extraktion, Fortschritt
│   └── services/
│       ├── gemini_service.py  Gemini Vision OCR + Übersetzung (primär)
│       ├── ollama_service.py  Ollama Vision OCR + Übersetzung (Fallback)
│       ├── ocr_service.py     Legacy: Google Vision, manga-ocr, EasyOCR (nicht aktiv)
│       ├── translation_service.py  Google Cloud Translate (standalone)
│       └── image_service.py   Panel-Verwaltung (Dateisystem)
├── frontend/                 Vanilla HTML/CSS/JS (kein Framework)
│   ├── index.html
│   ├── css/
│   │   ├── main.css          Globale Styles, Navigation, Home
│   │   ├── scanner.css       Scanner-Ansicht
│   │   └── learning.css      Lernmodus-Ansicht
│   └── js/
│       ├── api.js            API-Client mit Response-Cache
│       ├── app.js            Router, Navigation, View-Management
│       ├── scanner.js        Panel-Upload, OCR-Anzeige, Bounding-Box-Overlay
│       └── learning.js       Vokabel-Karten, Fortschritt
├── panels/                   Manga-Panel-Bilder
│   ├── uploads/              Vom Benutzer hochgeladene Panels (gitignored)
│   └── test_synthetic/       Synthetische Test-Panels mit Ground-Truth
├── setup_ollama_remote.sh    Script für Ollama-Installation auf dem GPU-Rechner
└── .env.example              Vorlage für Konfiguration
```

## OCR-Ansätze (Chronologie)

| Ansatz | Status | Anmerkungen |
|--------|--------|-------------|
| **Google Cloud Vision** | ❌ Verworfen | Gute Ergebnisse, aber kostenpflichtig und erfordert GCP-Projekt-Setup |
| **manga-ocr** (kha-white) | ❌ Verworfen | Spezialisiert auf Manga, aber braucht PyTorch + GPU, zu langsam auf CPU |
| **EasyOCR** | ❌ Verworfen | Japanisch-Support schwach, schlechte Erkennung bei vertikalem Text |
| **Gemini Vision** (gemini-2.0-flash) | ✅ Primär | Bester Ansatz: OCR + Übersetzung + Bounding Boxes in einem API-Call. Gratis API-Key |
| **Ollama Vision** (minicpm-v:8b) | ✅ Fallback | Läuft auf Remote-GPU-Rechner (10.100.10.112). 2-Step: Text lesen → Übersetzen |

## Setup

### 1. Backend (dieser Rechner)

```powershell
# Virtual Environment erstellen
cd manga_language_learner
python -m venv .venv
.venv\Scripts\Activate.ps1

# Abhängigkeiten installieren
pip install -r backend/requirements.txt
pip install Pillow python-dotenv google-genai

# Konfiguration
copy .env.example .env
# .env bearbeiten: GEMINI_API_KEY eintragen (https://aistudio.google.com/apikey)

# Server starten
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

→ Website öffnen: **http://localhost:8000**

### 2. Ollama auf Remote-GPU-Rechner (optional, Fallback)

```bash
# Auf dem GPU-Rechner (10.100.10.112) ausführen:
ssh user@10.100.10.112
bash setup_ollama_remote.sh
```

Das Script installiert Ollama, konfiguriert Netzwerk-Zugriff und lädt die Modelle:
- `minicpm-v:8b` – Vision-Modell für Manga-Text-Erkennung
- `llama3.1:8b` – Text-Modell für Übersetzungen

### 3. Umgebungsvariablen (.env)

```env
# Gemini API (primär, empfohlen)
GEMINI_API_KEY=dein-api-key

# Ollama Remote (Fallback)
OLLAMA_BASE_URL=http://10.100.10.112:11434
OLLAMA_MODEL=minicpm-v:8b

# Google Cloud (optional, Legacy)
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
GOOGLE_PROJECT_ID=your-project-id
```

## Tech Stack

- **Backend:** Python 3.13, FastAPI, Uvicorn
- **OCR/Translation:** Google Gemini Vision API (primär), Ollama + minicpm-v (Fallback)
- **Frontend:** HTML, CSS, JavaScript (vanilla, kein Framework)
- **Infrastruktur:** Ollama auf Remote-GPU-Rechner (10.100.10.112, selber wie KORA-E)
