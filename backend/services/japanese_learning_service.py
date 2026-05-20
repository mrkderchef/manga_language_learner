"""Japanese tokenization, readings, and central lookup cache helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import requests

from config import BASE_DIR

logger = logging.getLogger(__name__)

LOOKUP_CACHE_DIR = BASE_DIR / "backend" / "data" / "lookup_cache"
for _name in ("kanji", "words", "readings", "lookup"):
    (LOOKUP_CACHE_DIR / _name).mkdir(parents=True, exist_ok=True)

KANJIAPI_BASE_URL = os.getenv("KANJIAPI_BASE_URL", "https://kanjiapi.dev/v1").rstrip("/")
KANJI_RE = re.compile(r"[\u3400-\u9fff]")
KANA_RE = re.compile(r"[\u3040-\u30ff]")

_tokenizer = None
_kakasi = None


def _cache_path(kind: str, key: str) -> Path:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return LOOKUP_CACHE_DIR / kind / f"{digest}.json"


def _has_cache(kind: str, key: str) -> bool:
    return _cache_path(kind, key).exists()


def _read_cache(kind: str, key: str) -> dict[str, Any] | None:
    path = _cache_path(kind, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(kind: str, key: str, data: dict[str, Any]) -> dict[str, Any]:
    path = _cache_path(kind, key)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _fetch_kanjiapi(path: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{KANJIAPI_BASE_URL}{path}", timeout=6)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.info("kanjiapi lookup skipped/failed for %s: %s", path, exc)
        return None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    try:
        from sudachipy import dictionary

        _tokenizer = dictionary.Dictionary().create()
    except Exception:
        _tokenizer = False
    return _tokenizer or None


def _get_kakasi():
    global _kakasi
    if _kakasi is not None:
        return _kakasi
    try:
        import pykakasi

        _kakasi = pykakasi.kakasi()
    except Exception:
        _kakasi = False
    return _kakasi or None


def kana_to_romaji(text: str) -> str:
    if not text:
        return ""
    kakasi = _get_kakasi()
    if not kakasi:
        return ""
    try:
        return " ".join(part.get("hepburn", "") for part in kakasi.convert(text)).strip()
    except Exception:
        return ""


def katakana_to_hiragana(text: str) -> str:
    chars = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            chars.append(chr(code - 0x60))
        else:
            chars.append(ch)
    return "".join(chars)


def tokenize_text(text: str) -> dict[str, Any]:
    """Return token/readings metadata. Falls back gracefully if Sudachi is absent."""
    normalized = (text or "").strip()
    if not normalized:
        return {"reading_kana": "", "reading_romaji": "", "tokens": [], "kanji_spans": []}

    tokenizer = _get_tokenizer()
    tokens: list[dict[str, Any]] = []
    reading_parts: list[str] = []
    if tokenizer:
        offset = 0
        for index, morpheme in enumerate(tokenizer.tokenize(normalized)):
            surface = morpheme.surface()
            start = normalized.find(surface, offset)
            if start < 0:
                start = offset
            end = start + len(surface)
            offset = end
            reading = morpheme.reading_form()
            if reading == "*":
                reading = surface
            reading_hira = katakana_to_hiragana(reading)
            reading_parts.append(reading_hira)
            tokens.append({
                "id": f"tok_{index:03d}",
                "surface": surface,
                "lemma": morpheme.dictionary_form(),
                "reading_kana": reading_hira,
                "reading_romaji": kana_to_romaji(reading_hira),
                "pos": [part for part in morpheme.part_of_speech() if part != "*"],
                "start": start,
                "end": end,
                "kanji": KANJI_RE.findall(surface),
            })
    else:
        reading_parts.append(normalized)
        tokens.append({
            "id": "tok_000",
            "surface": normalized,
            "lemma": normalized,
            "reading_kana": "",
            "reading_romaji": "",
            "pos": [],
            "start": 0,
            "end": len(normalized),
            "kanji": KANJI_RE.findall(normalized),
        })

    kanji_spans = [
        {"character": ch, "start": i, "end": i + 1}
        for i, ch in enumerate(normalized)
        if KANJI_RE.match(ch)
    ]
    reading_kana = "".join(reading_parts)
    return {
        "reading_kana": reading_kana,
        "reading_romaji": kana_to_romaji(reading_kana),
        "tokens": tokens,
        "kanji_spans": kanji_spans,
    }


def token_plausibility(text: str) -> dict[str, Any]:
    """Cheap semantic plausibility signal for OCR candidate ranking."""
    data = tokenize_text(text)
    tokens = data.get("tokens", [])
    if not text or not tokens:
        return {"score": 0.0, "known_ratio": 0.0, "token_count": 0}
    known = 0
    for token in tokens:
        if token.get("lemma") and token["lemma"] != token["surface"]:
            known += 1
        elif token.get("reading_kana") and KANA_RE.search(token["reading_kana"]):
            known += 1
    known_ratio = known / max(1, len(tokens))
    # Prefer candidates that tokenize into a few meaningful pieces, not one giant unknown.
    token_shape = min(len(tokens), 8) / 8
    return {
        "score": round((known_ratio * 16) + (token_shape * 4), 3),
        "known_ratio": round(known_ratio, 3),
        "token_count": len(tokens),
    }


def lookup_kanji(character: str) -> dict[str, Any]:
    ch = (character or "")[:1]
    cached = _read_cache("kanji", ch)
    if cached:
        return cached
    api_data = _fetch_kanjiapi(f"/kanji/{ch}") if ch else None
    if api_data:
        data = {
            "type": "kanji",
            "kanji": api_data.get("kanji") or ch,
            "meanings": api_data.get("meanings", []),
            "kun_readings": api_data.get("kun_readings", []),
            "on_readings": api_data.get("on_readings", []),
            "name_readings": api_data.get("name_readings", []),
            "stroke_count": api_data.get("stroke_count"),
            "grade": api_data.get("grade"),
            "jlpt": api_data.get("jlpt"),
            "source": "kanjiapi.dev",
        }
        return _write_cache("kanji", ch, data)
    data = {
        "type": "kanji",
        "kanji": ch,
        "meanings": [],
        "kun_readings": [],
        "on_readings": [],
        "name_readings": [],
        "stroke_count": None,
        "grade": None,
        "jlpt": None,
        "source": "central-cache-placeholder",
    }
    return _write_cache("kanji", ch, data)


def lookup_word(text: str) -> dict[str, Any]:
    key = (text or "").strip()
    cached = _read_cache("words", key)
    if cached:
        return cached
    api_data = _fetch_kanjiapi(f"/words/{key}") if key else None
    data = {
        "type": "word",
        "text": key,
        "tokens": tokenize_text(key).get("tokens", []),
        "entries": api_data.get("variants", []) if api_data else [],
        "source": "kanjiapi.dev+sudachi" if api_data else "sudachi",
    }
    return _write_cache("words", key, data)


def lookup_reading(reading: str) -> dict[str, Any]:
    key = (reading or "").strip()
    cached = _read_cache("readings", key)
    if cached:
        return cached
    api_data = _fetch_kanjiapi(f"/reading/{key}") if key else None
    data = {
        "type": "reading",
        "reading": key,
        "romaji": kana_to_romaji(key),
        "kanji": api_data.get("main_kanji", []) if api_data else [],
        "name_kanji": api_data.get("name_kanji", []) if api_data else [],
        "source": "kanjiapi.dev" if api_data else "local",
    }
    return _write_cache("readings", key, data)


def lookup_text(text: str) -> dict[str, Any]:
    key = (text or "").strip()
    cached = _read_cache("lookup", key)
    if cached:
        return cached
    data = {
        "type": "lookup",
        "text": key,
        **tokenize_text(key),
        "kanji": [lookup_kanji(ch) for ch in sorted(set(KANJI_RE.findall(key)))],
    }
    return _write_cache("lookup", key, data)


def build_panel_learning(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    """Build panel-specific learning metadata while reusing global lookup cache."""
    by_region: dict[str, dict[str, Any]] = {}
    lookup_hits = 0
    lookup_misses = 0

    for index, ann in enumerate(annotations, start=1):
        text = str(ann.get("text") or "")
        region_id = ann.get("region_id") or ann.get("id") or f"region_{index:04d}"
        tokenized = tokenize_text(text)
        kanji_chars = sorted(set(KANJI_RE.findall(text)))
        word_surfaces = [token.get("surface", "") for token in tokenized.get("tokens", []) if token.get("surface")]
        readings = [token.get("reading_kana", "") for token in tokenized.get("tokens", []) if token.get("reading_kana")]

        kanji = []
        for ch in kanji_chars:
            lookup_hits += int(_has_cache("kanji", ch))
            lookup_misses += int(not _has_cache("kanji", ch))
            kanji.append(lookup_kanji(ch))

        words = []
        for word in sorted(set(word_surfaces)):
            lookup_hits += int(_has_cache("words", word))
            lookup_misses += int(not _has_cache("words", word))
            words.append(lookup_word(word))

        reading_lookups = []
        for reading in sorted(set(readings)):
            lookup_hits += int(_has_cache("readings", reading))
            lookup_misses += int(not _has_cache("readings", reading))
            reading_lookups.append(lookup_reading(reading))

        by_region[region_id] = {
            **tokenized,
            "kanji": kanji,
            "words": words,
            "reading_lookups": reading_lookups,
        }

    return {
        "success": True,
        "by_region": by_region,
        "global_lookup_hits": lookup_hits,
        "global_lookup_misses": lookup_misses,
        "source": "sudachi+kanjiapi-cache",
    }
