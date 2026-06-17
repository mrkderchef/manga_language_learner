from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter

from services.bootstrap import check_runtime_status, ensure_ocr_assets


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/runtime", tags=["runtime"])


@router.get("/status")
async def runtime_status():
    return check_runtime_status()


@router.post("/ocr-assets/download")
async def download_ocr_assets():
    logger.info('component=runtime action=download_ocr_assets status=queued msg="OCR asset setup queued"')
    return await asyncio.get_event_loop().run_in_executor(None, ensure_ocr_assets)
