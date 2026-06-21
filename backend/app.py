from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from config import (
    API_HOST,
    API_PORT,
    API_TITLE,
    API_VERSION,
    PANELS_DIR,
    BASE_DIR,
    UPLOADS_DIR,
    PANEL_DATA_DIR,
)
from services.logging_config import RequestLoggingMiddleware, configure_logging

configure_logging()

from services.bootstrap import check_runtime_status
from services.storage.image_service import ImageService
from routes.runtime import router as runtime_router
from routes.scanner import router as scanner_router
from routes.rabbithole import router as rabbithole_router
from routes.learning import router as learning_router

logger = logging.getLogger(__name__)

FRONTEND_DIR = BASE_DIR / "frontend"
THUMB_CACHE_DIR = BASE_DIR / "backend" / "data" / "thumbs"
THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    status = check_runtime_status()
    logger.info(
        'component=app status=startup ocr_ready=%s ollama_ready=%s warnings=%s msg="Backend startup check completed"',
        status.get("ocr", {}).get("ready"),
        status.get("ollama", {}).get("available"),
        len(status.get("warnings", []) or []),
    )
    yield


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Manga Language Learner - Learn Japanese through manga panels",
    lifespan=lifespan,
)

# GZip compression for all responses > 500 bytes
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(runtime_router)
app.include_router(scanner_router)
app.include_router(rabbithole_router)
app.include_router(learning_router)

# Serve frontend static files (CSS, JS)
app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_media_response(path: Path, root: Path, allowed_suffixes: set[str], cache_seconds: int = 86400):
    if path.suffix.lower() not in allowed_suffixes:
        raise HTTPException(status_code=404, detail="Unsupported media type")
    if not _is_relative_to(path, root) or not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(
        str(path),
        headers={"Cache-Control": f"public, max-age={cache_seconds}"},
    )


@app.get("/api/media/panel/{filename}")
async def get_panel_media(filename: str):
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")
    root = UPLOADS_DIR if _is_relative_to(panel_path, UPLOADS_DIR) else PANELS_DIR
    return _safe_media_response(panel_path, root, {".jpg", ".jpeg", ".png"})


@app.get("/api/media/rendered/{panel_id}/current.png")
async def get_rendered_panel(panel_id: str):
    if not ImageService._SAFE_NAME_RE.match(panel_id):
        raise HTTPException(status_code=404, detail="Media not found")
    rendered = PANEL_DATA_DIR / panel_id / "rendered" / "current.png"
    return _safe_media_response(rendered, PANEL_DATA_DIR, {".png"}, cache_seconds=0)


@app.get("/api/media/ocr-debug/{debug_path:path}")
async def get_ocr_debug_media(debug_path: str):
    if ".." in Path(debug_path).parts:
        raise HTTPException(status_code=404, detail="Media not found")
    path = PANEL_DATA_DIR / debug_path
    parts = path.parts
    if "ocr" not in parts or "debug" not in parts:
        raise HTTPException(status_code=404, detail="Media not found")
    return _safe_media_response(path, PANEL_DATA_DIR, {".png"}, cache_seconds=0)


@app.get("/api/thumb/{filename}")
async def get_thumbnail(filename: str, size: int = 160):
    """Serve a cached thumbnail of a panel image."""
    size = max(32, min(size, 400))
    panel_path = ImageService.get_panel_by_filename(filename)
    if not panel_path:
        raise HTTPException(status_code=404, detail="Panel not found")

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
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True, access_log=False, log_config=None)
