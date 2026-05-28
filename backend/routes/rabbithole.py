from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from services.rabbithole import nlp as rabbithole_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rabbithole", tags=["rabbithole"])


@router.get("/lookup")
async def lookup_text(text: str):
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
    return rabbithole_service.lookup_text(text)


@router.get("/kanji/{character}")
async def lookup_kanji(character: str):
    if not character:
        raise HTTPException(status_code=400, detail="No kanji provided")
    return rabbithole_service.lookup_kanji(character)


@router.get("/word")
async def lookup_word(text: str):
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
    return rabbithole_service.lookup_word(text)


@router.get("/reading/{reading}")
async def lookup_reading(reading: str):
    if not reading:
        raise HTTPException(status_code=400, detail="No reading provided")
    return rabbithole_service.lookup_reading(reading)
