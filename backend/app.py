from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import logging
from pathlib import Path
from config import API_HOST, API_PORT, API_TITLE, API_VERSION, PANELS_DIR
from routes import panels_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Manga Language Learner - Backend API"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(panels_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Manga Language Learner API",
        "version": API_VERSION,
        "status": "running"
    }


@app.get("/panels/{filename}")
async def get_panel(filename: str):
    """Serve a panel image file"""
    # Prevent path traversal attacks
    if ".." in filename or "/" in filename or "\\" in filename:
        return JSONResponse(status_code=400, content={"detail": "Invalid filename"})
    
    # Try to find the file
    panel_path = PANELS_DIR / filename
    upload_path = PANELS_DIR / "uploads" / filename
    
    if panel_path.exists() and panel_path.is_file():
        return FileResponse(panel_path, media_type="image/jpeg")
    elif upload_path.exists() and upload_path.is_file():
        return FileResponse(upload_path, media_type="image/jpeg")
    else:
        return JSONResponse(status_code=404, content={"detail": "Panel not found"})


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "manga-language-learner-api"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info(f"Server running on http://{API_HOST}:{API_PORT}")
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )
