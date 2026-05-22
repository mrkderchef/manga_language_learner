import os
import shutil
from pathlib import Path
from typing import Dict, Optional
from config import PANELS_DIR, UPLOADS_DIR, ALLOWED_EXTENSIONS
import logging

logger = logging.getLogger(__name__)


class ImageService:
    
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
                        "path": f"/panels/{file.name}",
                        "size": file.stat().st_size,
                        "type": "original"
                    })
            
            for file in PANELS_DIR.glob("*.jpeg"):
                if file.is_file():
                    panels.append({
                        "filename": file.name,
                        "path": f"/panels/{file.name}",
                        "size": file.stat().st_size,
                        "type": "original"
                    })
            
            for file in PANELS_DIR.glob("*.png"):
                if file.is_file():
                    panels.append({
                        "filename": file.name,
                        "path": f"/panels/{file.name}",
                        "size": file.stat().st_size,
                        "type": "original"
                    })
            
            # Get uploaded panels
            if UPLOADS_DIR.exists():
                for file in UPLOADS_DIR.glob("*.*"):
                    if file.is_file() and file.suffix.lower() in ALLOWED_EXTENSIONS:
                        panels.append({
                            "filename": file.name,
                            "path": f"/panels/uploads/{file.name}",
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
            # Check in main directory
            panel_path = PANELS_DIR / filename
            if panel_path.exists() and panel_path.is_file():
                return panel_path
            
            # Check in uploads directory
            upload_path = UPLOADS_DIR / filename
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
            # Validate file extension
            ext = Path(filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                return {
                    "success": False,
                    "error": f"File type {ext} not allowed. Allowed: {ALLOWED_EXTENSIONS}"
                }
            
            # Save file
            file_path = UPLOADS_DIR / filename
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            return {
                "success": True,
                "filename": filename,
                "path": str(file_path),
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
            file_path = UPLOADS_DIR / filename
            
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

