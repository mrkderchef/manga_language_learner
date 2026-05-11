# Frontend - Manga Language Learner

React + Vite frontend for the Manga Language Learner application.

## Features

- 📖 Upload and manage manga panels
- 🔤 Extract Japanese text (OCR)
- 🌐 Translate to English
- 📊 View word-by-word translations with confidence scores
- 🎨 Beautiful UI with real-time panel preview

## Setup

### Install Dependencies

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

Frontend will be available at `http://localhost:5173`

### Build for Production

```bash
npm run build
```

## API Integration

Frontend communicates with backend at `http://localhost:8000` via Vite proxy.

Proxied endpoints:
- `/api/panels/list` - Get all panels
- `/api/panels/upload` - Upload new panel
- `/api/panels/{filename}/ocr` - Extract text
- `/api/panels/{filename}/extract-and-translate` - Extract and translate
- `/api/panels/{filename}/delete` - Delete panel

## Project Structure

```
frontend/
├── index.html                    # HTML entry point
├── package.json                  # Dependencies
├── vite.config.js               # Vite config
├── src/
│   ├── main.jsx                 # React entry
│   ├── App.jsx                  # Root component
│   ├── api/
│   │   └── client.js            # API client
│   ├── components/
│   │   ├── PanelUpload.jsx      # Upload component
│   │   └── PanelViewer.jsx      # Viewer component
│   ├── pages/
│   │   └── HomePage.jsx         # Main page
│   └── styles/
│       ├── index.css            # Global styles
│       ├── HomePage.css         # Page styles
│       ├── PanelUpload.css      # Upload styles
│       └── PanelViewer.css      # Viewer styles
```

## Technologies

- React 18
- Vite 5
- Axios for API calls
- CSS3 with modern features
