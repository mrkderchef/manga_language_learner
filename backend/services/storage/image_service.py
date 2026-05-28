import os
import shutil
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
import io
import re
from config import PANELS_DIR, UPLOADS_DIR, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
import logging

logger = logging.getLogger(__name__)


class ImageService:
    _SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

    @staticmethod
    def _safe_filename(filename: str | None) -> str | None:
        name = Path(filename or "").name
        if not name or name in {".", ".."}:
            return None
        if name != (filename or ""):
            return None
        if not ImageService._SAFE_NAME_RE.match(name):
            return None
        if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            return None
        return name
    
    @staticmethod
    def get_all_panels() -> Dict:
        """Get all manga panels from the panels directory"""
        try:
            panels = []
            
            # Get panels from the main directory
            for file in PANELS_DIR.glob("*.jpg"):
                if file.is_file():
                        panels.append({
                            "filename": file.name,
                            "path": f"/api/media/panel/{file.name}",
                            "size": file.stat().st_size,
                            "type": "original"
                        })
            
            for file in PANELS_DIR.glob("*.jpeg"):
                if file.is_file():
                        panels.append({
                            "filename": file.name,
                            "path": f"/api/media/panel/{file.name}",
                            "size": file.stat().st_size,
                            "type": "original"
                        })
            
            for file in PANELS_DIR.glob("*.png"):
                if file.is_file():
                        panels.append({
                            "filename": file.name,
                            "path": f"/api/media/panel/{file.name}",
                            "size": file.stat().st_size,
                            "type": "original"
                        })
            
            # Get uploaded panels
            if UPLOADS_DIR.exists():
                for file in UPLOADS_DIR.glob("*.*"):
                    if file.is_file() and file.suffix.lower() in ALLOWED_EXTENSIONS:
                        panels.append({
                            "filename": file.name,
                            "path": f"/api/media/panel/{file.name}",
                            "size": file.stat().st_size,
                            "type": "uploaded"
                        })
            
            return {
                "success": True,
                "total": len(panels),
                "panels": sorted(panels, key=lambda x: x["filename"])
            }
        
        except Exception as e:
            logger.error(f"Failed to get panels: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "panels": []
            }
    
    @staticmethod
    def get_panel_by_filename(filename: str) -> Optional[Path]:
        """Get a specific panel file by filename"""
        try:
            safe_name = ImageService._safe_filename(filename)
            if not safe_name:
                return None
            # Check in main directory
            panel_path = PANELS_DIR / safe_name
            if panel_path.exists() and panel_path.is_file():
                return panel_path
            
            # Check in uploads directory
            upload_path = UPLOADS_DIR / safe_name
            if upload_path.exists() and upload_path.is_file():
                return upload_path
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to get panel: {str(e)}")
            return None
    
    @staticmethod
    def save_uploaded_panel(file_content: bytes, filename: str) -> Dict:
        """Save an uploaded panel file"""
        try:
            safe_name = ImageService._safe_filename(filename)
            if not safe_name:
                return {
                    "success": False,
                    "error": "Unsafe or unsupported filename"
                }
            if len(file_content) > MAX_FILE_SIZE:
                return {
                    "success": False,
                    "error": f"File exceeds maximum size of {MAX_FILE_SIZE} bytes"
                }
            # Validate file extension
            ext = Path(safe_name).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                return {
                    "success": False,
                    "error": f"File type {ext} not allowed. Allowed: {ALLOWED_EXTENSIONS}"
                }
            try:
                with Image.open(io.BytesIO(file_content)) as img:
                    img.verify()
            except Exception:
                return {
                    "success": False,
                    "error": "Uploaded file is not a valid image"
                }
            
            # Save file
            file_path = UPLOADS_DIR / safe_name
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            return {
                "success": True,
                "filename": safe_name,
                "path": f"/api/media/panel/{safe_name}",
                "size": len(file_content)
            }
        
        except Exception as e:
            logger.error(f"Failed to save panel: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def delete_panel(filename: str) -> Dict:
        """Delete an uploaded panel"""
        try:
            # Only allow deletion of uploaded panels, not original panels
            safe_name = ImageService._safe_filename(filename)
            if not safe_name:
                return {
                    "success": False,
                    "error": "Unsafe or unsupported filename"
                }
            file_path = UPLOADS_DIR / safe_name
            
            if not file_path.exists():
                return {
                    "success": False,
                    "error": "Panel not found"
                }
            
            os.remove(file_path)
            
            return {
                "success": True,
                "message": f"Panel {filename} deleted"
            }
        
        except Exception as e:
            logger.error(f"Failed to delete panel: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

__all__ = ['ImageService']
