from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
from pathlib import Path
from config import API_HOST, API_PORT, API_TITLE, API_VERSION, PANELS_DIR, BASE_DIR
from routes.scanner import router as scanner_router
from routes.learning import router as learning_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Manga Language Learner - Learn Japanese through manga panels"
)

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


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True)
