# Manga Language Learner - Backend API

FastAPI-based REST API for OCR and translation of manga panels.

## Features

- 📸 **Manga Panel Management** - Upload, list, and manage manga panels
- 🔤 **OCR (Optical Character Recognition)** - Extract Japanese text from images using Google Cloud Vision
- 🌐 **Translation** - Translate extracted text to English using Google Cloud Translate
- 📊 **Service Status Monitoring** - Check health of OCR and translation services

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Google Cloud Services (Optional)

For Google Cloud Vision and Translation APIs:

1. Create a Google Cloud Project: https://cloud.google.com/docs/authentication/getting-started
2. Enable Vision API and Translate API
3. Create a Service Account and download the JSON key
4. Copy `.env.example` to `.env` and add your credentials:

```bash
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
GOOGLE_PROJECT_ID=your-project-id
```

**Note:** If Google Cloud credentials are not provided, the app will use fallback services:
- OCR: pytesseract (local)
- Translation: MyMemory API (free)

### 3. Run the Server

```bash
python app.py
```

The API will be available at `http://localhost:8000`

Interactive API documentation: `http://localhost:8000/docs`

## API Endpoints

### Panel Management

**List all panels**
```
GET /api/panels/list
```

**Upload a panel**
```
POST /api/panels/upload
Content-Type: multipart/form-data
```

**Delete a panel**
```
DELETE /api/panels/{filename}
```

### OCR & Translation

**Extract text from panel (OCR only)**
```
POST /api/panels/{filename}/ocr
```

Response:
```json
{
  "success": true,
  "filename": "Natsume_1b.jpg",
  "text": "Full extracted text...",
  "annotations": [
    {
      "text": "日本語",
      "confidence": 0.95,
      "vertices": [...],
      "translation": "Japanese"
    }
  ]
}
```

**Extract and translate**
```
POST /api/panels/{filename}/extract-and-translate
```

**Translate text**
```
POST /api/panels/translate?text=こんにちは&target_language=en
```

### Service Status

**Check service status**
```
GET /api/panels/status
```

Response:
```json
{
  "ocr": {
    "ocr_service": "google_vision",
    "available": true
  },
  "translation": {
    "translation_service": "google_translate",
    "available": true
  }
}
```

## Project Structure

```
backend/
├── app.py                 # FastAPI application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── services/
│   ├── ocr_service.py   # Google Vision OCR
│   ├── translation_service.py  # Google Translate
│   └── image_service.py  # Image management
├── routes/
│   └── panels.py        # Panel API endpoints
└── panels/
    └── uploads/         # Uploaded panels storage
```

## Supported File Formats

- JPG / JPEG
- PNG

## Notes

- Maximum file size: 10 MB
- Free tier limits:
  - Google Vision: 1,000 requests/month
  - Google Translate: 500,000 characters/month
