from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
from pathlib import Path
from PIL import Image
import io
from config import API_HOST, API_PORT, API_TITLE, API_VERSION, PANELS_DIR, BASE_DIR, UPLOADS_DIR
from routes.scanner import router as scanner_router
from routes.learning import router as learning_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = BASE_DIR / "frontend"
THUMB_CACHE_DIR = BASE_DIR / "backend" / "data" / "thumbs"
THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Manga Language Learner - Learn Japanese through manga panels"
)

# GZip compression for all responses > 500 bytes
app.add_middleware(GZipMiddleware, minimum_size=500)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve panel images as static files
app.mount("/panels", StaticFiles(directory=str(PANELS_DIR)), name="panels")

# Routes
app.include_router(scanner_router)
app.include_router(learning_router)

# Serve frontend static files (CSS, JS)
app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


@app.get("/api/thumb/{filename}")
async def get_thumbnail(filename: str, size: int = 160):
    """Serve a cached thumbnail of a panel image."""
    size = min(size, 400)  # Cap max size
    # Find original
    panel_path = PANELS_DIR / filename
    if not panel_path.exists():
        panel_path = UPLOADS_DIR / filename
    if not panel_path.exists():
        return Response(status_code=404)

    thumb_name = f"{panel_path.stem}_{size}{panel_path.suffix}"
    thumb_path = THUMB_CACHE_DIR / thumb_name

    if not thumb_path.exists() or thumb_path.stat().st_mtime < panel_path.stat().st_mtime:
        img = Image.open(panel_path)
        img.thumbnail((size, size), Image.LANCZOS)
        img.save(thumb_path, quality=80, optimize=True)

    return FileResponse(
        str(thumb_path),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True)
