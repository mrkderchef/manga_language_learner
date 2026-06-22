"""Japanese Rabbithole analysis, readings, glossing, and lookup cache helpers."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests

from config import BASE_DIR, KANJIAPI_BASE_URL, RABBITHOLE_GINZA_ENABLED
from services.rabbithole.reference_data import (
    FUNCTION_GLOSSARY,
    GRAMMAR_DETAILS,
    KANA_DIGRAPH_ROMAJI,
    KANA_ROMAJI,
    LEXICAL_GLOSSARY,
    POS_LABELS,
    SMALL_KANA_NOTES,
    SYMBOL_DETAILS,
)
from services.storage.json_store import write_json_atomic

logger = logging.getLogger(__name__)

LOOKUP_CACHE_DIR = BASE_DIR / "backend" / "data" / "lookup_cache"
for _name in ("kanji", "words", "readings", "lookup", "kanjivg", "wiktionary"):
    (LOOKUP_CACHE_DIR / _name).mkdir(parents=True, exist_ok=True)

CACHE_SCHEMA_VERSION = 4
RABBITHOLE_CONTRACT_VERSION = 2
KANJIVG_REVISION = "r20250816"
KANJIVG_RAW_URL = "https://raw.githubusercontent.com/KanjiVG/kanjivg/{revision}/kanji/{codepoint}.svg"
WIKTIONARY_API_URL = "https://en.wiktionary.org/w/api.php"
WIKTIONARY_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60
HTTP_USER_AGENT = "MangaLanguageLearner/1.0 (local educational app; provenance-first lookup)"

KANJI_RE = re.compile(r"[\u3400-\u9fff]")
KANA_RE = re.compile(r"[\u3040-\u30ff]")
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z\uFF21-\uFF3A\uFF41-\uFF5A]")
DIGIT_RE = re.compile(r"[0-9\uFF10-\uFF19]")

_tokenizer = None
_kakasi = None

LOCAL_GRAMMAR_SOURCE = "local:grammar"
LOCAL_KANA_SOURCE = "local:kana"
LOCAL_SYMBOL_SOURCE = "local:symbols"


def source_catalog() -> dict[str, dict[str, Any]]:
    """Stable source descriptions shared by panel and lazy inspector payloads."""
    catalog = {
        "kanjidic2": {
            "name": "KANJIDIC2 via KanjiAPI",
            "dataset": "KANJIDIC2",
            "intermediary": "kanjiapi.dev",
            "canonical_url": "https://www.edrdg.org/wiki/KANJIDIC_Project.html",
            "license": "EDRDG licence / acknowledgement required",
            "version": "KanjiAPI current export",
        },
        "jmdict": {
            "name": "JMdict via KanjiAPI",
            "dataset": "JMdict",
            "intermediary": "kanjiapi.dev",
            "canonical_url": "https://www.edrdg.org/jmdict/j_jmdict.html",
            "license": "EDRDG licence / acknowledgement required",
            "version": "KanjiAPI current export",
        },
        "sudachi": {
            "name": "SudachiPy",
            "dataset": "SudachiDict",
            "intermediary": "SudachiPy",
            "canonical_url": "https://github.com/WorksApplications/SudachiPy",
            "license": "Apache-2.0 (library); dictionary licence varies",
            "version": "installed runtime",
        },
        "pykakasi": {
            "name": "pykakasi",
            "dataset": "pykakasi transliteration tables",
            "intermediary": "local runtime",
            "canonical_url": "https://github.com/miurahr/pykakasi",
            "license": "GPL-3.0",
            "version": "installed runtime",
        },
        "local_romaji": {
            "name": "Local Hepburn fallback",
            "dataset": "application kana-to-romaji table",
            "intermediary": "local application",
            "canonical_url": "https://en.wikipedia.org/wiki/Hepburn_romanization",
            "license": "application source licence",
            "version": "current application revision",
        },
        "kanjivg": {
            "name": "KanjiVG",
            "dataset": "KanjiVG Japanese stroke-order SVG",
            "intermediary": "raw.githubusercontent.com",
            "canonical_url": "https://github.com/KanjiVG/kanjivg",
            "license": "CC BY-SA 3.0",
            "version": KANJIVG_REVISION,
        },
        "wiktionary": {
            "name": "English Wiktionary",
            "dataset": "Chinese → Glyph origin editorial section",
            "intermediary": "MediaWiki Action API",
            "canonical_url": "https://en.wiktionary.org/",
            "license": "CC BY-SA 4.0",
            "version": "page revision recorded per lookup",
        },
        "unicode": {
            "name": "Unicode",
            "dataset": "Unicode Character Database",
            "intermediary": "Python unicodedata",
            "canonical_url": "https://www.unicode.org/ucd/",
            "license": "Unicode data files licence",
            "version": unicodedata.unidata_version,
        },
        "app_editorial": {
            "name": "App editorial notes",
            "dataset": "Manga Language Learner grammar and usage notes",
            "intermediary": "local application",
            "canonical_url": "",
            "license": "application source licence",
            "version": "current application revision",
        },
        "mangaocr": {
            "name": "Manga OCR",
            "dataset": "OCR model output",
            "intermediary": "local runtime",
            "canonical_url": "https://github.com/kha-white/manga-ocr",
            "license": "Apache-2.0",
            "version": "installed runtime",
        },
        "translation_engine": {
            "name": "Translation engine",
            "dataset": "generated translation",
            "intermediary": "configured local/remote engine",
            "canonical_url": "",
            "license": "engine dependent",
            "version": "request metadata",
        },
    }
    retrieved_at = int(time.time())
    for record in catalog.values():
        record.setdefault("retrieved_at", retrieved_at)
    return catalog


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
    # Generated runtime data may be removed while the backend is still alive.
    # Recreate this cache bucket here instead of relying on module-import setup.
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["_cache_schema_version"] = CACHE_SCHEMA_VERSION
    payload.setdefault("retrieved_at", int(time.time()))
    write_json_atomic(path, payload)
    return payload


def _fetch_kanjiapi(path: str) -> Any | None:
    try:
        response = requests.get(f"{KANJIAPI_BASE_URL}{path}", timeout=6)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
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


def _fallback_kana_to_romaji(text: str) -> str:
    hira = katakana_to_hiragana(text)
    output: list[str] = []
    geminate = False
    index = 0
    while index < len(hira):
        ch = hira[index]
        if ch == "っ":
            geminate = True
            index += 1
            continue
        if ch == "ー":
            previous = output[-1] if output else ""
            vowel = next((value for value in reversed(previous) if value in "aeiou"), "")
            output.append(vowel)
            index += 1
            continue
        pair = hira[index:index + 2]
        roman = KANA_DIGRAPH_ROMAJI.get(pair)
        if roman:
            index += 2
        else:
            roman = KANA_ROMAJI.get(ch, ch)
            index += 1
        if geminate and roman and roman[0].isalpha() and roman != "n":
            roman = ("t" if roman.startswith("ch") else roman[0]) + roman
        geminate = False
        output.append(roman)
    return "".join(output)


def kana_to_romanji(text: str) -> str:
    if not text:
        return ""
    kakasi = _get_kakasi()
    if not kakasi:
        return _fallback_kana_to_romaji(text)
    try:
        return " ".join(part.get("hepburn", "") for part in kakasi.convert(text)).strip()
    except Exception:
        return _fallback_kana_to_romaji(text)


def _romanization_source_id() -> str:
    return "pykakasi" if _get_kakasi() else "local_romaji"


def katakana_to_hiragana(text: str) -> str:
    chars = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            chars.append(chr(code - 0x60))
        else:
            chars.append(ch)
    return "".join(chars)


def _resolve_sudachi_split_mode(name: str):
    try:
        from sudachipy import SplitMode

        return getattr(SplitMode, name)
    except Exception:
        pass
    try:
        from sudachipy import tokenizer

        return getattr(tokenizer.Tokenizer.SplitMode, name)
    except Exception:
        return None


def tokenize_text(text: str, split_mode: str = "C") -> dict[str, Any]:
    """Return token/readings metadata. Falls back gracefully if Sudachi is absent."""
    normalized = (text or "").strip()
    if not normalized:
        return {
            "reading_hiragana": "",
            "reading_romaji": "",
            "reading_romanji": "",
            "tokens": [],
            "kanji_spans": [],
        }

    tokenizer = _get_tokenizer()
    tokens: list[dict[str, Any]] = []
    reading_parts: list[str] = []
    if tokenizer:
        offset = 0
        mode = _resolve_sudachi_split_mode(split_mode)
        morphemes = tokenizer.tokenize(normalized, mode) if mode else tokenizer.tokenize(normalized)
        for index, morpheme in enumerate(morphemes):
            surface = morpheme.surface()
            start = normalized.find(surface, offset)
            if start < 0:
                start = offset
            end = start + len(surface)
            offset = end
            reading = morpheme.reading_form()
            pos = [part for part in morpheme.part_of_speech() if part != "*"]
            
            # Symbols and punctuation should not have readings
            is_symbol = any("補助記号" in str(part) for part in pos) or all(
                not KANJI_RE.match(ch) and not KANA_RE.match(ch) and not ch.isalnum() 
                for ch in surface
            )
            
            if is_symbol or reading == "*":
                reading_hira = ""
            else:
                reading_hira = katakana_to_hiragana(reading)
            
            reading_parts.append(reading_hira)
            token_romaji = kana_to_romanji(reading_hira)
            tokens.append({
                "id": f"tok_{index:03d}",
                "surface": surface,
                "lemma": morpheme.dictionary_form(),
                "reading_hiragana": reading_hira,
                "reading_romaji": token_romaji,
                "reading_romanji": token_romaji,
                "pos": pos,
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
            "reading_hiragana": "",
            "reading_romaji": "",
            "reading_romanji": "",
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
    reading_hiragana = "".join(reading_parts)
    reading_romaji = kana_to_romanji(reading_hiragana)
    return {
        "reading_hiragana": reading_hiragana,
        "reading_romaji": reading_romaji,
        "reading_romanji": reading_romaji,
        "tokens": tokens,
        "kanji_spans": kanji_spans,
    }


def token_plausibility(text: str) -> dict[str, Any]:
    """Cheap token plausibility signal for OCR candidate ranking.

    This is intentionally lightweight and local. It is not an embedding-based
    semantic judge; it estimates whether OCR output looks analyzable as
    Japanese rather than Latin/garbage noise.
    """
    data = tokenize_text(text)
    tokens = data.get("tokens", [])
    if not text or not tokens:
        return {
            "score": 0.0,
            "known_ratio": 0.0,
            "token_count": 0,
            "scored_token_count": 0,
            "japanese_token_ratio": 0.0,
            "latin_token_ratio": 0.0,
            "opaque_token_ratio": 0.0,
            "char_japanese_ratio": 0.0,
        }

    normalized = str(text).strip()
    significant_chars = [ch for ch in normalized if not ch.isspace()]
    japanese_char_count = sum(1 for ch in significant_chars if JAPANESE_RE.match(ch))
    latin_char_count = sum(1 for ch in significant_chars if LATIN_RE.match(ch))
    char_japanese_ratio = japanese_char_count / max(1, len(significant_chars))

    scored_tokens = [token for token in tokens if not _is_symbol_token(token)]
    japanese_tokens = 0
    known_tokens = 0
    latin_tokens = 0
    opaque_tokens = 0

    for token in scored_tokens:
        surface = str(token.get("surface") or "")
        lemma = str(token.get("lemma") or "")
        reading_hiragana = str(token.get("reading_hiragana") or "")
        pos = [str(part) for part in (token.get("pos") or []) if str(part)]

        has_japanese_surface = bool(JAPANESE_RE.search(surface))
        has_latin_surface = bool(LATIN_RE.search(surface))
        has_digit_surface = bool(DIGIT_RE.search(surface))
        has_function_hint = surface in FUNCTION_GLOSSARY or lemma in FUNCTION_GLOSSARY
        has_lexical_hint = surface in LEXICAL_GLOSSARY or lemma in LEXICAL_GLOSSARY
        has_analysis = bool(pos) or (lemma and lemma != surface) or bool(KANA_RE.search(reading_hiragana))

        if has_japanese_surface:
            japanese_tokens += 1
            if has_analysis or has_function_hint or has_lexical_hint:
                known_tokens += 1
            else:
                opaque_tokens += 1
            continue

        if has_latin_surface and not has_digit_surface:
            latin_tokens += 1
            opaque_tokens += 1
            continue

        if has_digit_surface:
            opaque_tokens += 1
            continue

        if has_function_hint or has_lexical_hint:
            known_tokens += 1
        else:
            opaque_tokens += 1

    scored_token_count = len(scored_tokens)
    known_ratio = known_tokens / max(1, scored_token_count)
    japanese_token_ratio = japanese_tokens / max(1, scored_token_count)
    latin_token_ratio = latin_tokens / max(1, scored_token_count)
    opaque_token_ratio = opaque_tokens / max(1, scored_token_count)
    token_shape = (min(scored_token_count, 8) / 8) if japanese_tokens else 0.0

    penalties: dict[str, float] = {}
    if latin_tokens and japanese_tokens == 0:
        penalties["latin_only_tokens"] = -12.0
    elif latin_tokens:
        penalties["mixed_latin_tokens"] = -min(6.0, latin_tokens * 2.0)
    if opaque_token_ratio >= 0.75:
        penalties["opaque_token_majority"] = -6.0
    if char_japanese_ratio < 0.35:
        penalties["low_japanese_char_ratio"] = -6.0

    score = (
        (known_ratio * 10.0)
        + (japanese_token_ratio * 8.0)
        + (token_shape * 4.0)
        + sum(penalties.values())
    )
    return {
        "score": round(max(-12.0, min(20.0, score)), 3),
        "known_ratio": round(known_ratio, 3),
        "token_count": len(tokens),
        "scored_token_count": scored_token_count,
        "japanese_token_ratio": round(japanese_token_ratio, 3),
        "latin_token_ratio": round(latin_token_ratio, 3),
        "opaque_token_ratio": round(opaque_token_ratio, 3),
        "char_japanese_ratio": round(char_japanese_ratio, 3),
        "penalties": penalties,
    }


def _source_subset(*source_ids: str) -> dict[str, dict[str, Any]]:
    catalog = source_catalog()
    return {source_id: catalog[source_id] for source_id in source_ids if source_id in catalog}


def _structured_readings(readings: list[Any]) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for value in readings:
        kana = str(value or "").strip()
        if not kana:
            continue
        romanized_parts = [
            part if part in {".", "-"} else kana_to_romanji(part).replace(" ", "")
            for part in re.split(r"([.-])", kana)
            if part
        ]
        results.append({"kana": kana, "romaji": "".join(romanized_parts)})
    return results


def _unicode_metadata(character: str) -> dict[str, Any]:
    if not character or len(character) != 1:
        return {}
    return {
        "character": character,
        "codepoint": f"U+{ord(character):04X}",
        "name": unicodedata.name(character, "Unassigned character"),
        "category": unicodedata.category(character),
        "script_note": "Unicode identifies the encoded character; it does not define a word meaning or etymology.",
    }


def _cache_is_fresh(payload: dict[str, Any] | None, ttl_seconds: int) -> bool:
    if not payload:
        return False
    retrieved_at = payload.get("retrieved_at")
    try:
        return (time.time() - float(retrieved_at)) < ttl_seconds
    except (TypeError, ValueError):
        return False


def _kanjivg_lookup(character: str) -> dict[str, Any]:
    cached = _read_cache("kanjivg", f"{KANJIVG_REVISION}:{character}")
    if cached and cached.get("revision") == KANJIVG_REVISION:
        return cached

    codepoint = f"{ord(character):05x}"
    url = KANJIVG_RAW_URL.format(revision=KANJIVG_REVISION, codepoint=codepoint)
    try:
        response = requests.get(url, headers={"User-Agent": HTTP_USER_AGENT}, timeout=6)
        if response.status_code == 404:
            raise LookupError("KanjiVG file is not available for this character")
        response.raise_for_status()
        if len(response.content) > 1_000_000:
            raise ValueError("KanjiVG document exceeds the safe size limit")
        root = ET.fromstring(response.content)
        paths: list[dict[str, str]] = []
        for index, element in enumerate(root.iter()):
            if element.tag.rsplit("}", 1)[-1] != "path":
                continue
            path_data = str(element.attrib.get("d") or "").strip()
            if path_data and len(path_data) <= 12_000 and re.fullmatch(r"[MmLlHhVvCcSsQqTtAaZzEe0-9.,+\-\s]+", path_data):
                paths.append({
                    "id": str(element.attrib.get("id") or f"stroke-{index + 1}"),
                    "d": path_data,
                })
            if len(paths) >= 64:
                break

        components: list[dict[str, str]] = []
        seen_components: set[tuple[str, str]] = set()
        for element in root.iter():
            component = next(
                (str(value) for key, value in element.attrib.items() if key.rsplit("}", 1)[-1] == "element"),
                "",
            ).strip()
            if not component or component == character:
                continue
            position = next(
                (str(value) for key, value in element.attrib.items() if key.rsplit("}", 1)[-1] == "position"),
                "",
            ).strip()
            key = (component, position)
            if key not in seen_components:
                seen_components.add(key)
                components.append({"element": component[:16], "position": position[:24]})
            if len(components) >= 64:
                break

        if not paths:
            raise ValueError("KanjiVG document contained no stroke paths")
        payload = {
            "available": True,
            "character": character,
            "revision": KANJIVG_REVISION,
            "source_url": url,
            "view_box": str(root.attrib.get("viewBox") or "0 0 109 109"),
            "paths": paths,
            "components": components,
            "source_id": "kanjivg",
        }
        return _write_cache("kanjivg", f"{KANJIVG_REVISION}:{character}", payload)
    except Exception as exc:
        logger.info("KanjiVG lookup skipped/failed for %s: %s", character, exc)
        if cached:
            return {**cached, "stale": True, "error": str(exc)}
        return {
            "available": False,
            "character": character,
            "revision": KANJIVG_REVISION,
            "source_url": url,
            "paths": [],
            "components": [],
            "source_id": "kanjivg",
            "error": str(exc),
        }


class _GlyphOriginHTMLParser(HTMLParser):
    """Extract readable paragraphs/list items; all markup is discarded."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._parts: list[str] = []
        self._block_depth = 0
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "table", "figure"}:
            self._ignored_depth += 1
        if not self._ignored_depth and tag in {"p", "li"}:
            if self._block_depth == 0:
                self._parts = []
            self._block_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "table", "figure"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if not self._ignored_depth and tag in {"p", "li"} and self._block_depth:
            self._block_depth -= 1
            if self._block_depth == 0:
                text = re.sub(r"\s+", " ", "".join(self._parts)).strip()
                text = re.sub(r"\[\d+\]", "", text)
                if text:
                    self.blocks.append(text)

    def handle_data(self, data: str) -> None:
        if self._block_depth and not self._ignored_depth:
            self._parts.append(data)


def _wiktionary_glyph_origin(character: str) -> dict[str, Any]:
    cache_key = f"en:{character}:glyph-origin"
    cached = _read_cache("wiktionary", cache_key)
    if _cache_is_fresh(cached, WIKTIONARY_CACHE_TTL_SECONDS):
        return cached or {}

    page_url = f"https://en.wiktionary.org/wiki/{requests.utils.quote(character)}"
    try:
        common_params = {"action": "parse", "page": character, "format": "json", "formatversion": "2"}
        section_response = requests.get(
            WIKTIONARY_API_URL,
            params={**common_params, "prop": "sections|revid"},
            headers={"User-Agent": HTTP_USER_AGENT},
            timeout=8,
        )
        section_response.raise_for_status()
        parse_data = section_response.json().get("parse", {})
        revision_id = parse_data.get("revid")
        glyph_section = None
        in_chinese = False
        for section in parse_data.get("sections", []):
            level = int(section.get("level") or 0)
            title = html.unescape(re.sub(r"<[^>]+>", "", str(section.get("line") or ""))).strip()
            if level == 2:
                in_chinese = title.casefold() == "chinese"
            elif in_chinese and title.casefold() in {"glyph origin", "glyph origins"}:
                glyph_section = str(section.get("index") or "")
                break

        text = ""
        if glyph_section:
            content_response = requests.get(
                WIKTIONARY_API_URL,
                params={**common_params, "prop": "text|revid", "section": glyph_section},
                headers={"User-Agent": HTTP_USER_AGENT},
                timeout=8,
            )
            content_response.raise_for_status()
            content_parse = content_response.json().get("parse", {})
            revision_id = content_parse.get("revid") or revision_id
            parser = _GlyphOriginHTMLParser()
            parser.feed(str(content_parse.get("text") or ""))
            text = "\n\n".join(parser.blocks[:5])[:3000]

        revision_url = (
            f"https://en.wiktionary.org/w/index.php?title={requests.utils.quote(character)}&oldid={revision_id}"
            if revision_id else page_url
        )
        payload = {
            "available": bool(text),
            "character": character,
            "text": text,
            "message": "" if text else "No glyph-origin note available",
            "revision_id": revision_id,
            "revision_url": revision_url,
            "page_url": page_url,
            "source_id": "wiktionary",
        }
        return _write_cache("wiktionary", cache_key, payload)
    except Exception as exc:
        logger.info("Wiktionary glyph-origin lookup skipped/failed for %s: %s", character, exc)
        if cached:
            return {**cached, "stale": True, "error": str(exc)}
        return {
            "available": False,
            "character": character,
            "text": "",
            "message": "No glyph-origin note available",
            "revision_id": None,
            "revision_url": page_url,
            "page_url": page_url,
            "source_id": "wiktionary",
            "error": str(exc),
        }


def _normalize_kanji_payload(character: str, raw: dict[str, Any], *, source: str) -> dict[str, Any]:
    kun = list(raw.get("kun_readings") or [])
    on = list(raw.get("on_readings") or [])
    nanori = list(raw.get("name_readings") or [])
    romaji_source = _romanization_source_id()
    return {
        "type": "kanji",
        "kanji": raw.get("kanji") or character,
        "meanings": list(raw.get("meanings") or []),
        "kun_readings": kun,
        "on_readings": on,
        "name_readings": nanori,
        "structured_readings": {
            "kun": _structured_readings(kun),
            "on": _structured_readings(on),
            "nanori": _structured_readings(nanori),
        },
        "stroke_count": raw.get("stroke_count"),
        "grade": raw.get("grade"),
        "jlpt": raw.get("jlpt"),
        "unicode": raw.get("unicode") or f"{ord(character):04X}",
        "unicode_metadata": _unicode_metadata(character),
        "heisig_en": raw.get("heisig_en"),
        "freq_mainichi_shinbun": raw.get("freq_mainichi_shinbun"),
        "notes": list(raw.get("notes") or []),
        "source": source,
        "sources": _source_subset("kanjidic2", romaji_source, "unicode"),
        "field_sources": {
            "meanings": ["kanjidic2"],
            "kun_readings": ["kanjidic2"],
            "on_readings": ["kanjidic2"],
            "name_readings": ["kanjidic2"],
            "structured_readings": ["kanjidic2", romaji_source],
            "stroke_count": ["kanjidic2"],
            "grade": ["kanjidic2"],
            "jlpt": ["kanjidic2"],
            "heisig_en": ["kanjidic2"],
            "freq_mainichi_shinbun": ["kanjidic2"],
            "unicode": ["unicode"],
            "unicode_metadata": ["unicode"],
        },
    }


def lookup_kanji(character: str, enrich: bool = False) -> dict[str, Any]:
    ch = (character or "")[:1]
    if not ch:
        return _normalize_kanji_payload("?", {}, source="unavailable")
    cached = _read_cache("kanji", ch)
    cached_is_valid = bool(cached and cached.get("kanji") and (
        cached.get("meanings")
        or cached.get("kun_readings")
        or cached.get("on_readings")
        or cached.get("stroke_count") is not None
    ))
    current_schema = bool(cached and cached.get("_cache_schema_version") == CACHE_SCHEMA_VERSION)
    api_data = None if cached_is_valid and current_schema else _fetch_kanjiapi(f"/kanji/{ch}")
    if isinstance(api_data, dict):
        data = _write_cache("kanji", ch, _normalize_kanji_payload(ch, api_data, source="KANJIDIC2 via KanjiAPI"))
    elif cached_is_valid:
        data = cached if current_schema else _write_cache(
            "kanji", ch, _normalize_kanji_payload(ch, cached or {}, source="KANJIDIC2 via KanjiAPI (cached)")
        )
    else:
        data = _normalize_kanji_payload(ch, {}, source="unavailable")

    if enrich:
        stroke_order = _kanjivg_lookup(ch)
        glyph_origin = _wiktionary_glyph_origin(ch)
        rich_sources = _source_subset("kanjivg", "wiktionary")
        if "kanjivg" in rich_sources:
            rich_sources["kanjivg"]["retrieved_at"] = stroke_order.get("retrieved_at")
        if "wiktionary" in rich_sources:
            rich_sources["wiktionary"]["retrieved_at"] = glyph_origin.get("retrieved_at")
            rich_sources["wiktionary"]["version"] = (
                f"revision {glyph_origin.get('revision_id')}" if glyph_origin.get("revision_id") else "revision unavailable"
            )
        data = {
            **data,
            "stroke_order": stroke_order,
            "glyph_origin": glyph_origin,
            "components": stroke_order.get("components", []),
            "component_source": "kanjivg",
            "sources": {
                **(data.get("sources") or {}),
                **rich_sources,
            },
            "field_sources": {
                **(data.get("field_sources") or {}),
                "stroke_order": ["kanjivg"],
                "components": ["kanjivg"],
                "glyph_origin": ["wiktionary"],
            },
        }
    return data


def _word_lookup_cache_key(text: str, reading_hiragana: str = "") -> str:
    reading = (reading_hiragana or "").strip()
    return f"{text.strip()}::{reading}" if reading else text.strip()


def lookup_word(text: str, reading_hiragana: str = "") -> dict[str, Any]:
    key = (text or "").strip()
    contextual_reading = (reading_hiragana or "").strip()
    cache_key = _word_lookup_cache_key(key, contextual_reading)
    cached = _read_cache("words", cache_key)
    if cached and cached.get("text") and cached.get("entries") and cached.get("_cache_schema_version") == CACHE_SCHEMA_VERSION:
        return cached
    tokenized = tokenize_text(key)
    resolved_reading = contextual_reading or str(tokenized.get("reading_hiragana") or "")
    entries, candidate_count = _fetch_dictionary_entries(key, resolved_reading)
    if not entries and cached and cached.get("entries"):
        entries = list(cached.get("entries") or [])
        candidate_count = int(cached.get("candidate_count") or len(entries))
    reading_romaji = kana_to_romanji(resolved_reading) if contextual_reading else str(
        tokenized.get("reading_romaji") or tokenized.get("reading_romanji") or ""
    )
    romaji_source = _romanization_source_id()
    data = {
        "type": "word",
        "text": key,
        "reading_hiragana": resolved_reading,
        "reading_romaji": reading_romaji,
        "reading_romanji": reading_romaji,
        "tokens": tokenized.get("tokens", []),
        "entries": entries,
        "candidate_count": candidate_count,
        "source": "jmdict" if entries else "sudachi",
        "sources": _source_subset("jmdict", "sudachi", romaji_source),
        "field_sources": {
            "reading_hiragana": ["sudachi"],
            "reading_romaji": ["sudachi", romaji_source],
            "reading_romanji": ["sudachi", romaji_source],
            "tokens": ["sudachi"],
            "entries": ["jmdict"],
            "candidate_count": ["jmdict"],
        },
    }
    if JAPANESE_RE.search(key) and not entries and candidate_count == 0:
        return data
    return _write_cache("words", cache_key, data)


def lookup_reading(reading: str) -> dict[str, Any]:
    key = (reading or "").strip()
    cached = _read_cache("readings", key)
    if cached and cached.get("_cache_schema_version") == CACHE_SCHEMA_VERSION:
        return cached
    api_data = _fetch_kanjiapi(f"/reading/{key}") if key else None
    api_dict = api_data if isinstance(api_data, dict) else None
    romaji = kana_to_romanji(key)
    cached_dict = cached if isinstance(cached, dict) else {}
    romaji_source = _romanization_source_id()
    data = {
        "type": "reading",
        "reading": key,
        "romaji": romaji,
        "romanji": romaji,
        "kanji": api_dict.get("main_kanji", []) if api_dict else cached_dict.get("kanji", []),
        "name_kanji": api_dict.get("name_kanji", []) if api_dict else cached_dict.get("name_kanji", []),
        "source": "jmdict" if api_dict or cached_dict.get("kanji") else romaji_source,
        "sources": _source_subset("jmdict", romaji_source),
        "field_sources": {
            "romaji": [romaji_source],
            "romanji": [romaji_source],
            "kanji": ["jmdict"],
            "name_kanji": ["jmdict"],
        },
    }
    return _write_cache("readings", key, data)


def lookup_text(text: str) -> dict[str, Any]:
    key = (text or "").strip()
    cached = _read_cache("lookup", key)
    if cached and cached.get("_cache_schema_version") == CACHE_SCHEMA_VERSION:
        return cached
    data = {
        "type": "lookup",
        "text": key,
        **tokenize_text(key),
        "kanji": [lookup_kanji(ch) for ch in sorted(set(KANJI_RE.findall(key)))],
    }
    return _write_cache("lookup", key, data)


def _flatten_meanings(value: Any) -> list[str]:
    results: list[str] = []
    if value is None:
        return results
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            results.append(cleaned)
        return results
    if isinstance(value, list):
        for item in value:
            results.extend(_flatten_meanings(item))
        return results
    if isinstance(value, dict):
        for key in ("meanings", "meaning", "gloss", "glosses", "definition", "definitions"):
            if key in value:
                results.extend(_flatten_meanings(value.get(key)))
        return results
    return results


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _priority_weight(priority: str) -> int:
    value = str(priority or "")
    if value.startswith("ichi"):
        return 100
    if value.startswith("news"):
        return 90
    if value.startswith("spec"):
        return 70
    if value.startswith("gai"):
        return 60
    if value.startswith("nf"):
        match = re.search(r"(\d+)", value)
        if match:
            return max(1, 80 - int(match.group(1)))
        return 50
    return 0


def _priority_label(priority: str) -> str:
    value = str(priority or "")
    if value == "news1":
        return "Mainichi newspaper priority, first 12,000 words"
    if value == "news2":
        return "Mainichi newspaper priority, second 12,000 words"
    if value == "ichi1":
        return "Ichimango common-word list, higher-priority subset"
    if value == "ichi2":
        return "Ichimango common-word list, lower-priority subset"
    if value == "spec1":
        return "special common-word priority, first group"
    if value == "spec2":
        return "special common-word priority, second group"
    if value == "gai1":
        return "common loanword list, higher-priority subset"
    if value == "gai2":
        return "common loanword list, lower-priority subset"
    nf_match = re.fullmatch(r"nf(\d{2})", value)
    if nf_match:
        band = int(nf_match.group(1))
        start = ((band - 1) * 500) + 1
        return f"Mainichi frequency band {band:02d} (approximately ranks {start}–{band * 500})"
    return value


def _pos_labels(pos: list[Any]) -> list[str]:
    labels: list[str] = []
    for part in pos:
        label = POS_LABELS.get(str(part))
        if label and label not in labels:
            labels.append(label)
    return labels


def _dictionary_search_keys(text: str) -> list[str]:
    kanji = _dedupe_strings(KANJI_RE.findall(text))
    return _dedupe_strings([text, *kanji]) if text else []


def _variant_matches(variant: dict[str, Any], text: str, reading_hiragana: str) -> bool:
    written = str(variant.get("written") or "")
    pronounced = katakana_to_hiragana(str(variant.get("pronounced") or ""))
    return written == text or bool(reading_hiragana and pronounced == reading_hiragana)


def _entry_match_score(entry: dict[str, Any], text: str, reading_hiragana: str) -> int:
    scores: list[int] = []
    for variant in entry.get("variants", []) if isinstance(entry.get("variants"), list) else []:
        if not isinstance(variant, dict):
            continue
        score = 0
        written = str(variant.get("written") or "")
        pronounced = katakana_to_hiragana(str(variant.get("pronounced") or ""))
        priorities = [str(item) for item in variant.get("priorities", []) if str(item)]
        if written == text:
            score += 1000
        if reading_hiragana and pronounced == reading_hiragana:
            score += 300
        score += max([_priority_weight(priority) for priority in priorities] or [0])
        scores.append(score)
    return max(scores or [0])


def _normalize_dictionary_entry(entry: dict[str, Any], text: str, reading_hiragana: str) -> dict[str, Any] | None:
    variants = entry.get("variants", [])
    meanings = entry.get("meanings", [])
    if not isinstance(variants, list) or not isinstance(meanings, list):
        return None

    matching_variants = [variant for variant in variants if isinstance(variant, dict) and _variant_matches(variant, text, reading_hiragana)]
    selected_variants = matching_variants or [variant for variant in variants if isinstance(variant, dict)]
    priorities = _dedupe_strings([
        str(priority)
        for variant in selected_variants
        for priority in (variant.get("priorities") or [])
        if str(priority)
    ])
    senses = [
        {"glosses": _dedupe_strings(_flatten_meanings(meaning.get("glosses", [])))}
        for meaning in meanings
        if isinstance(meaning, dict)
    ]
    senses = [sense for sense in senses if sense["glosses"]]
    if not senses:
        return None

    return {
        "source": "jmdict",
        "match": "exact" if matching_variants else "candidate",
        "score": _entry_match_score(entry, text, reading_hiragana),
        "variants": [
            {
                "written": str(variant.get("written") or ""),
                "reading_hiragana": katakana_to_hiragana(str(variant.get("pronounced") or "")),
                "priorities": [str(priority) for priority in (variant.get("priorities") or []) if str(priority)],
            }
            for variant in selected_variants[:6]
        ],
        "priority_tags": priorities,
        "priority_labels": _dedupe_strings([_priority_label(priority) for priority in priorities]),
        "senses": senses[:6],
        "glosses": _dedupe_strings([gloss for sense in senses for gloss in sense["glosses"]]),
    }


def _fetch_dictionary_entries(text: str, reading_hiragana: str) -> tuple[list[dict[str, Any]], int]:
    candidates_by_key: dict[str, dict[str, Any]] = {}
    for search_key in _dictionary_search_keys(text):
        api_data = _fetch_kanjiapi(f"/words/{search_key}")
        if not isinstance(api_data, list):
            continue
        for entry in api_data:
            if not isinstance(entry, dict):
                continue
            normalized = _normalize_dictionary_entry(entry, text, reading_hiragana)
            if not normalized:
                continue
            key = json.dumps(normalized.get("variants", []), ensure_ascii=False, sort_keys=True)
            existing = candidates_by_key.get(key)
            if not existing or int(normalized.get("score", 0)) > int(existing.get("score", 0)):
                candidates_by_key[key] = normalized

    entries = sorted(candidates_by_key.values(), key=lambda item: int(item.get("score", 0)), reverse=True)
    exact = [entry for entry in entries if entry.get("match") == "exact"]
    return exact[:8], len(entries)


def _unit_kind(token: dict[str, Any]) -> str:
    pos = token.get("pos") or []
    joined = "/".join(str(part) for part in pos)
    head = str(pos[0]) if pos else ""
    if "補助記号" in joined:
        return "token"
    if "助詞" in joined:
        return "particle"
    if "助動詞" in joined:
        return "aux"
    if "接尾" in joined or "接尾辞" in joined:
        return "suffix"
    if head:
        return "word"
    return "token"


def _unit_label(kind: str) -> str:
    return {
        "word": "segment",
        "particle": "particle",
        "aux": "auxiliary",
        "suffix": "suffix",
        "kanji": "kanji",
        "whole": "full text",
    }.get(kind, kind or "unit")


def _word_meanings(word_lookup: dict[str, Any]) -> list[str]:
    entries = word_lookup.get("entries", [])
    collected: list[str] = []
    for entry in entries if isinstance(entries, list) else []:
        collected.extend(_flatten_meanings(entry))
    return _dedupe_strings(collected)


def _kanji_meanings(kanji_lookup: dict[str, Any]) -> list[str]:
    return _dedupe_strings(_flatten_meanings(kanji_lookup.get("meanings", [])))


def _is_symbol_token(token: dict[str, Any]) -> bool:
    pos = token.get("pos") or []
    if any("補助記号" in str(part) for part in pos):
        return True
    surface = str(token.get("surface") or "")
    if not surface:
        return False
    return all(not KANJI_RE.match(ch) and not KANA_RE.match(ch) and not ch.isalnum() for ch in surface)


def _is_displayable_breakdown_token(token: dict[str, Any]) -> bool:
    return bool(str(token.get("surface") or "").strip())


def _fallback_gloss(token: dict[str, Any], prefer_reading: bool = False) -> str:
    if prefer_reading:
        reading_romanji = str(token.get("reading_romanji") or "").strip()
        if reading_romanji:
            return reading_romanji
        reading_hiragana = str(token.get("reading_hiragana") or "").strip()
        if reading_hiragana:
            return reading_hiragana
    lemma = str(token.get("lemma") or "").strip()
    surface = str(token.get("surface") or "").strip()
    if lemma and lemma != "*":
        return lemma
    return surface


def _dedupe_entries_by_variant(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = json.dumps(entry.get("variants", []), ensure_ascii=False, sort_keys=True)
        existing = deduped.get(key)
        if not existing or int(entry.get("score", 0)) > int(existing.get("score", 0)):
            deduped[key] = entry
    return list(deduped.values())


def _source_concept(source: str) -> str:
    value = str(source or "").strip()
    if not value:
        return ""
    if value == "jmdict" or value.startswith("kanjiapi.dev:/words"):
        return "jmdict"
    if value == "kanjidic2" or value.startswith("kanjiapi.dev:/kanji") or value == "kanjiapi.dev":
        return "kanjidic2"
    if value.startswith("local:") and "extended" in value:
        return "app_editorial"
    if value.startswith("local:"):
        return "app_editorial"
    if value.startswith("SudachiPy"):
        return "sudachi"
    return value


def _clean_dictionary_entry_source(entry: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(entry)
    cleaned["source"] = _source_concept(str(cleaned.get("source") or ""))
    return cleaned


def _hiragana_character_metadata(ch: str) -> dict[str, str] | None:
    if not ch or len(ch) != 1:
        return None
    if not re.fullmatch(r"[\u3041-\u3096]", ch):
        return None
    romaji = kana_to_romanji(ch)
    note = SMALL_KANA_NOTES.get(ch)
    kind = "small hiragana" if note else "hiragana syllable"
    return {
        "character": ch,
        "romaji": romaji or "",
        "kind": kind,
        "note": note or "",
    }


def _build_symbol_entry(surface: str) -> dict[str, Any] | None:
    detail = SYMBOL_DETAILS.get(surface)
    if not detail:
        return None
    label = str(detail.get("label") or "symbol")
    glosses = _dedupe_strings([label, *_flatten_meanings(detail.get("glosses", []))])
    tags = _dedupe_strings([str(tag) for tag in detail.get("tags", []) if str(tag)])
    return {
        "source": LOCAL_SYMBOL_SOURCE,
        "match": "exact",
        "score": 720,
        "variants": [{"written": surface, "reading_hiragana": "", "priorities": []}],
        "priority_tags": tags,
        "priority_labels": ["extended dictionary", "symbol reference"],
        "senses": [{"glosses": glosses}],
        "glosses": glosses,
    }


def _build_grammar_entry(surface: str, lemma: str, reading_hiragana: str) -> dict[str, Any] | None:
    key_candidates = _dedupe_strings([surface, lemma, reading_hiragana])
    detail = next((GRAMMAR_DETAILS.get(candidate) for candidate in key_candidates if GRAMMAR_DETAILS.get(candidate)), None)
    if not detail:
        return None

    label = str(detail.get("label") or "")
    glosses = _dedupe_strings([label, *_flatten_meanings(detail.get("glosses", []))])
    tags = _dedupe_strings([str(tag) for tag in detail.get("tags", []) if str(tag)])
    return {
        "source": LOCAL_GRAMMAR_SOURCE,
        "match": "exact",
        "score": 780,
        "variants": [{
            "written": surface,
            "reading_hiragana": reading_hiragana or surface if KANA_RE.search(surface) else "",
            "priorities": [],
        }],
        "priority_tags": tags,
        "priority_labels": ["extended dictionary", "grammar reference"],
        "senses": [{"glosses": glosses}],
        "glosses": glosses,
    }


def _build_kana_reference_entry(surface: str) -> dict[str, Any] | None:
    if not surface or len(surface) > 4:
        return None
    if not all(re.fullmatch(r"[\u3041-\u3096]", ch) for ch in surface):
        return None

    char_profiles = [_hiragana_character_metadata(ch) for ch in surface]
    char_profiles = [profile for profile in char_profiles if profile]
    if not char_profiles:
        return None

    senses: list[dict[str, Any]] = []
    for profile in char_profiles:
        line = f"{profile['character']}: {profile['kind']}"
        if profile.get("romaji"):
            line += f" ({profile['romaji']})"
        if profile.get("note"):
            line += f"; {profile['note']}"
        senses.append({"glosses": [line]})

    combined_romaji = kana_to_romanji(surface)
    summary = f"hiragana sequence ({combined_romaji})" if combined_romaji else "hiragana sequence"
    return {
        "source": LOCAL_KANA_SOURCE,
        "match": "exact",
        "score": 660,
        "variants": [{"written": surface, "reading_hiragana": surface, "priorities": []}],
        "priority_tags": ["kana", "hiragana", "orthography"],
        "priority_labels": ["extended dictionary", "kana reference"],
        "senses": [{"glosses": [summary]}, *senses[:5]],
        "glosses": [summary],
    }


def _extended_dictionary_entries_for_token(token: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    surface = str(token.get("surface") or "").strip()
    lemma = str(token.get("lemma") or surface).strip()
    reading_hiragana = str(token.get("reading_hiragana") or "").strip()
    pos_joined = "/".join(str(part) for part in (token.get("pos") or []))
    entries: list[dict[str, Any]] = []

    if kind in {"particle", "aux", "suffix"} or "助詞" in pos_joined or "助動詞" in pos_joined:
        grammar_entry = _build_grammar_entry(surface, lemma, reading_hiragana)
        if grammar_entry:
            entries.append(grammar_entry)

    symbol_entry = _build_symbol_entry(surface)
    if symbol_entry:
        entries.append(symbol_entry)

    if (kind in {"particle", "aux", "suffix", "token"} or len(surface) <= 2) and KANA_RE.search(surface):
        kana_entry = _build_kana_reference_entry(surface)
        if kana_entry:
            entries.append(kana_entry)

    return _dedupe_entries_by_variant(entries)


def _merge_dictionary_entries(remote_entries: list[dict[str, Any]], local_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = _dedupe_entries_by_variant([
        _clean_dictionary_entry_source(entry)
        for entry in [*(remote_entries or []), *(local_entries or [])]
        if isinstance(entry, dict)
    ])
    return sorted(merged, key=lambda item: int(item.get("score", 0)), reverse=True)


def _dictionary_source_summary(word_lookup: dict[str, Any], entries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    sources = _dedupe_strings([
        _source_concept(str(word_lookup.get("source") or "")),
        *[_source_concept(str(entry.get("source") or "")) for entry in entries],
    ])
    if not sources:
        return "", []
    catalog = source_catalog()
    labels = [str(catalog.get(source, {}).get("name") or source) for source in sources]
    return " + ".join(labels), sources


def _primary_meaning(token: dict[str, Any], kind: str, word_meanings: list[str], kanji_details: list[dict[str, Any]]) -> tuple[str, list[str]]:
    surface = str(token.get("surface") or "")
    lemma = str(token.get("lemma") or surface)
    pos = token.get("pos") or []
    joined = "/".join(str(part) for part in pos)
    glossary_keys = [surface, lemma]

    # Prefer dictionary-derived meanings. The compact editorial tables below
    # exist only as an offline fallback, not as a replacement for JMdict data.
    if word_meanings:
        return word_meanings[0], word_meanings[1:]

    for key in glossary_keys:
        if key in LEXICAL_GLOSSARY:
            primary = LEXICAL_GLOSSARY[key]
            alternates = [meaning for meaning in word_meanings if meaning != primary]
            return primary, alternates
        if key in FUNCTION_GLOSSARY and kind in {"particle", "aux", "suffix"}:
            primary = FUNCTION_GLOSSARY[key]
            alternates = [meaning for meaning in word_meanings if meaning != primary]
            return primary, alternates

    if _is_symbol_token(token):
        return surface, []

    if kind in {"particle", "aux", "suffix"}:
        for key in glossary_keys:
            if key in FUNCTION_GLOSSARY:
                return FUNCTION_GLOSSARY[key], []

    if "固有名詞" in joined:
        return _fallback_gloss(token, prefer_reading=True), []

    if len(kanji_details) == 1 and len(surface) == 1:
        kanji_meanings = _kanji_meanings(kanji_details[0])
        if kanji_meanings:
            return kanji_meanings[0], kanji_meanings[1:]

    if len(surface) <= 2:
        flattened_kanji = _dedupe_strings([
            meaning
            for kanji_detail in kanji_details
            for meaning in _kanji_meanings(kanji_detail)
        ])
        if flattened_kanji:
            return flattened_kanji[0], flattened_kanji[1:]

    return _fallback_gloss(token, prefer_reading=bool(kanji_details)), []


def _grammar_detail_for_token(token: dict[str, Any]) -> dict[str, Any]:
    surface = str(token.get("surface") or "")
    lemma = str(token.get("lemma") or surface)
    reading_hiragana = str(token.get("reading_hiragana") or "")
    key_candidates = _dedupe_strings([surface, lemma, reading_hiragana])
    detail = next((GRAMMAR_DETAILS.get(candidate) for candidate in key_candidates if GRAMMAR_DETAILS.get(candidate)), None)
    if not detail:
        return {}
    return {
        "label": detail.get("label"),
        "notes": detail.get("glosses", []),
        "tags": detail.get("tags", []),
    }


def _unit_feature_groups(unit: dict[str, Any]) -> dict[str, Any]:
    dictionary_entries = unit.get("dictionary_entries", [])
    return {
        "reading": {
            "hiragana": unit.get("reading_hiragana", ""),
            "romaji": unit.get("reading_romaji") or unit.get("reading_romanji", ""),
        },
        "meaning": {
            "glossary": unit.get("primary_meaning", ""),
            "alternate_glossary": unit.get("alternate_meanings", []),
        },
        "grammar": {
            "segment_type": unit.get("kind", ""),
            "unit_label": unit.get("unit_label", ""),
            "part_of_speech": unit.get("part_of_speech_labels", []),
            "analysis_tags": unit.get("pos", []),
            "function": unit.get("grammar_detail", {}),
        },
        "kanji": unit.get("kanji_details", []) if isinstance(unit.get("kanji_details"), list) else unit.get("kanji_details", {}),
        "dictionary": {
            "entries": dictionary_entries,
            "source": unit.get("dictionary_source", ""),
            "sources": unit.get("dictionary_sources", []),
            "candidate_count": unit.get("dictionary_candidate_count", 0),
        },
        "translation_context": {},
    }


def _kanji_unit(unit_id: str, character: str, start: int, end: int, kanji_lookup: dict[str, Any]) -> dict[str, Any]:
    meanings = _kanji_meanings(kanji_lookup)
    word_lookup = lookup_word(character)
    remote_entries = list(word_lookup.get("entries") or [])
    merged_entries = _merge_dictionary_entries(remote_entries, [])
    dictionary_source, dictionary_sources = _dictionary_source_summary(word_lookup, merged_entries)
    romaji_source = _romanization_source_id()
    unit = {
        "id": unit_id,
        "kind": "kanji",
        "unit_label": "kanji",
        "text": character,
        "start": start,
        "end": end,
        "lemma": character,
        "reading_hiragana": "",
        "reading_romaji": "",
        "reading_romanji": "",
        "pos": [],
        "primary_meaning": meanings[0] if meanings else character,
        "alternate_meanings": meanings[1:],
        "dictionary_entries": merged_entries[:4],
        "dictionary_source": dictionary_source,
        "dictionary_sources": dictionary_sources,
        "dictionary_candidate_count": word_lookup.get("candidate_count", 0),
        "kanji_details": {
            "kanji": kanji_lookup.get("kanji") or character,
            "meanings": kanji_lookup.get("meanings", []),
            "kun_readings": kanji_lookup.get("kun_readings", []),
            "on_readings": kanji_lookup.get("on_readings", []),
            "name_readings": kanji_lookup.get("name_readings", []),
            "structured_readings": kanji_lookup.get("structured_readings", {}),
            "stroke_count": kanji_lookup.get("stroke_count"),
            "grade": kanji_lookup.get("grade"),
            "jlpt": kanji_lookup.get("jlpt"),
            "unicode": kanji_lookup.get("unicode"),
            "heisig_en": kanji_lookup.get("heisig_en"),
            "freq_mainichi_shinbun": kanji_lookup.get("freq_mainichi_shinbun"),
            "notes": kanji_lookup.get("notes", []),
            "unicode_metadata": kanji_lookup.get("unicode_metadata", {}),
            "components": [],
            "component_source": "",
            "sources": kanji_lookup.get("sources", {}),
            "field_sources": kanji_lookup.get("field_sources", {}),
        },
        "sources": {
            **_source_subset("kanjidic2", "jmdict", romaji_source, "unicode"),
            **(word_lookup.get("sources") or {}),
        },
        "field_sources": {
            "text": ["kanjidic2"],
            "dictionary_entries": ["jmdict"],
            "kanji_details": ["kanjidic2", romaji_source, "unicode"],
        },
        "children": [],
    }
    unit["feature_groups"] = _unit_feature_groups(unit)
    return unit


def _whole_unit(text: str, reading_hiragana: str, reading_romaji: str, glossary: str, token_count: int, kanji_count: int) -> dict[str, Any]:
    romaji_source = _romanization_source_id()
    unit = {
        "id": "whole_000",
        "kind": "whole",
        "unit_label": "full text",
        "text": text,
        "start": 0,
        "end": len(text),
        "lemma": text,
        "reading_hiragana": reading_hiragana,
        "reading_romaji": reading_romaji,
        "reading_romanji": reading_romaji,
        "pos": [],
        "primary_meaning": glossary,
        "alternate_meanings": [],
        "children": [],
        "sources": _source_subset("mangaocr", "sudachi", romaji_source, "jmdict", "app_editorial", "translation_engine"),
        "field_sources": {
            "text": ["mangaocr"],
            "reading_hiragana": ["sudachi"],
            "reading_romaji": ["sudachi", romaji_source],
            "reading_romanji": ["sudachi", romaji_source],
            "primary_meaning": ["jmdict", "app_editorial"],
            "translation": ["translation_engine"],
        },
        "summary": {
            "token_count": token_count,
            "kanji_count": kanji_count,
        },
    }
    unit["feature_groups"] = _unit_feature_groups(unit)
    return unit


def _segment_from_unit(unit: dict[str, Any], segment_id: str) -> dict[str, Any]:
    return {
        "segment_id": segment_id,
        "unit_id": unit["id"],
        "kind": unit.get("kind", ""),
        "unit_label": unit.get("unit_label", ""),
        "text": unit.get("text", ""),
        "hiragana": unit.get("reading_hiragana", ""),
        "romaji": unit.get("reading_romaji") or unit.get("reading_romanji", ""),
        "romanji": unit.get("reading_romanji", ""),
        "gloss": unit.get("primary_meaning", ""),
        "start": unit.get("start", 0),
        "end": unit.get("end", 0),
        "children": unit.get("children", []),
        "feature_groups": unit.get("feature_groups", {}),
        "field_sources": unit.get("field_sources", {}),
    }


def _ginza_enrichment(text: str) -> dict[str, Any]:
    if not RABBITHOLE_GINZA_ENABLED:
        return {"enabled": False, "available": False}
    try:
        import spacy  # type: ignore

        nlp = spacy.load("ja_ginza")
        doc = nlp(text)
        return {
            "enabled": True,
            "available": True,
            "tokens": [
                {
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "dependency": token.dep_,
                    "head": token.head.text,
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                }
                for token in doc
            ],
            "entities": [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
                for ent in doc.ents
            ],
        }
    except Exception as exc:
        return {
            "enabled": True,
            "available": False,
            "error": str(exc),
        }


def build_panel_rabbithole(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    """Build per-region Rabbithole analysis while reusing the global lookup cache."""
    by_region: dict[str, dict[str, Any]] = {}
    global_lookup_hits = 0
    global_lookup_misses = 0
    romaji_source = _romanization_source_id()

    for index, ann in enumerate(annotations, start=1):
        text = str(ann.get("text") or "")
        region_id = ann.get("region_id") or ann.get("id") or f"region_{index:04d}"
        word_tokenized = tokenize_text(text, "C")
        units_by_id: dict[str, dict[str, Any]] = {}
        breakdowns: dict[str, dict[str, Any]] = {}
        region_lookup_hits = 0
        region_lookup_misses = 0

        def build_breakdown(layer_id: str, tokens: list[dict[str, Any]]) -> dict[str, Any]:
            nonlocal region_lookup_hits, region_lookup_misses
            layer_segments: list[dict[str, Any]] = []
            kanji_count = 0

            for token_index, token in enumerate(tokens):
                surface = str(token.get("surface") or "")
                if not _is_displayable_breakdown_token(token):
                    continue
                lemma = str(token.get("lemma") or surface)
                reading_hiragana = str(token.get("reading_hiragana") or "")
                kind = _unit_kind(token)
                word_keys = [surface]
                if lemma and lemma != surface:
                    word_keys.append(lemma)

                word_lookup = None
                for key in word_keys:
                    lookup_reading_hint = reading_hiragana if key == surface else ""
                    cache_key = _word_lookup_cache_key(key, lookup_reading_hint)
                    region_lookup_hits += int(_has_cache("words", cache_key))
                    region_lookup_misses += int(not _has_cache("words", cache_key))
                    lookup_data = lookup_word(key, lookup_reading_hint)
                    if lookup_data.get("entries"):
                        word_lookup = lookup_data
                        break
                    if word_lookup is None:
                        word_lookup = lookup_data

                kanji_details: list[dict[str, Any]] = []
                child_ids: list[str] = []
                for char_index, char in enumerate(surface):
                    if not KANJI_RE.match(char):
                        continue
                    region_lookup_hits += int(_has_cache("kanji", char))
                    region_lookup_misses += int(not _has_cache("kanji", char))
                    kanji_lookup = lookup_kanji(char)
                    kanji_details.append(kanji_lookup)
                    kanji_count += 1
                    child_id = f"{layer_id}_{token_index:03d}_kanji_{char_index:02d}"
                    child_ids.append(child_id)
                    if child_id not in units_by_id:
                        units_by_id[child_id] = _kanji_unit(
                            child_id,
                            char,
                            int(token.get("start", 0)) + char_index,
                            int(token.get("start", 0)) + char_index + 1,
                            kanji_lookup,
                        )

                if reading_hiragana:
                    region_lookup_hits += int(_has_cache("readings", reading_hiragana))
                    region_lookup_misses += int(not _has_cache("readings", reading_hiragana))
                    lookup_reading(reading_hiragana)

                word_meanings = _word_meanings(word_lookup or {})
                primary_meaning, alternate_meanings = _primary_meaning(token, kind, word_meanings, kanji_details)
                remote_entries = list((word_lookup or {}).get("entries") or [])
                local_entries = _extended_dictionary_entries_for_token(token, kind)
                dictionary_entries = _merge_dictionary_entries(remote_entries, local_entries)
                dictionary_source, dictionary_sources = _dictionary_source_summary(word_lookup or {}, dictionary_entries)
                unit_id = f"{layer_id}_{token_index:03d}"
                reading_romaji = str(token.get("reading_romaji") or token.get("reading_romanji") or "")
                grammar_detail = _grammar_detail_for_token(token)
                editorial_dictionary = any(entry.get("source") == "app_editorial" for entry in dictionary_entries)
                unit_source_ids = ["sudachi", romaji_source]
                if remote_entries:
                    unit_source_ids.append("jmdict")
                if grammar_detail or editorial_dictionary:
                    unit_source_ids.append("app_editorial")
                if len(surface) == 1:
                    unit_source_ids.append("unicode")
                unit = {
                    "id": unit_id,
                    "kind": kind,
                    "unit_label": _unit_label(kind),
                    "text": surface,
                    "start": int(token.get("start", 0)),
                    "end": int(token.get("end", 0)),
                    "lemma": lemma,
                    "reading_hiragana": reading_hiragana,
                    "reading_romaji": reading_romaji,
                    "reading_romanji": reading_romaji,
                    "pos": list(token.get("pos") or []),
                    "part_of_speech_labels": _pos_labels(list(token.get("pos") or [])),
                    "primary_meaning": primary_meaning,
                    "alternate_meanings": alternate_meanings,
                    "dictionary_entries": dictionary_entries[:4],
                    "dictionary_source": dictionary_source,
                    "dictionary_sources": dictionary_sources,
                    "dictionary_candidate_count": (word_lookup or {}).get("candidate_count", 0),
                    "grammar_detail": grammar_detail,
                    "unicode_metadata": _unicode_metadata(surface) if len(surface) == 1 else {},
                    "kanji_details": [
                        units_by_id[child_id].get("kanji_details", {})
                        for child_id in child_ids
                        if child_id in units_by_id
                    ],
                    "sources": _source_subset(*unit_source_ids),
                    "field_sources": {
                        "text": ["sudachi"],
                        "lemma": ["sudachi"],
                        "pos": ["sudachi"],
                        "part_of_speech_labels": ["sudachi", "app_editorial"],
                        "reading_hiragana": ["sudachi"],
                        "reading_romaji": ["sudachi", romaji_source],
                        "reading_romanji": ["sudachi", romaji_source],
                        "dictionary_entries": dictionary_sources,
                        "primary_meaning": dictionary_sources or ["app_editorial"],
                        "grammar_detail": ["app_editorial"] if grammar_detail else [],
                        "unicode_metadata": ["unicode"] if len(surface) == 1 else [],
                    },
                    "children": child_ids,
                }
                unit["feature_groups"] = _unit_feature_groups(unit)
                units_by_id[unit_id] = unit
                layer_segments.append(_segment_from_unit(unit, f"{layer_id}_seg_{token_index:03d}"))

            meaningful = [
                segment
                for segment in layer_segments
                if segment.get("gloss") and not _is_symbol_token(units_by_id.get(segment.get("unit_id"), {}))
            ]
            return {
                "segments": layer_segments,
                "glossary": " ".join(segment["gloss"] for segment in meaningful).strip(),
                "summary": {
                    "token_count": len(layer_segments),
                    "kanji_count": kanji_count,
                },
            }

        breakdowns["words"] = build_breakdown("words", word_tokenized.get("tokens", []))

        whole_glossary = breakdowns["words"]["glossary"]
        kanji_count = len(KANJI_RE.findall(text))
        whole_unit = _whole_unit(
            text,
            word_tokenized.get("reading_hiragana", ""),
            word_tokenized.get("reading_romaji") or word_tokenized.get("reading_romanji", ""),
            whole_glossary,
            breakdowns["words"]["summary"]["token_count"],
            kanji_count,
        )
        units_by_id[whole_unit["id"]] = whole_unit
        breakdowns["whole"] = {
            "segments": [_segment_from_unit(whole_unit, "whole_seg_000")] if text else [],
            "glossary": whole_glossary,
            "summary": whole_unit["summary"],
        }

        region_kanji_details = [
            unit.get("kanji_details", {})
            for unit in units_by_id.values()
            if unit.get("kind") == "kanji" and unit.get("kanji_details")
        ]
        seen_kanji = set()
        region_kanji_details = [
            detail
            for detail in region_kanji_details
            if not (detail.get("kanji") in seen_kanji or seen_kanji.add(detail.get("kanji")))
        ]

        global_lookup_hits += region_lookup_hits
        global_lookup_misses += region_lookup_misses
        by_region[region_id] = {
            "reading_hiragana": word_tokenized.get("reading_hiragana", ""),
            "reading_romaji": word_tokenized.get("reading_romaji") or word_tokenized.get("reading_romanji", ""),
            "reading_romanji": word_tokenized.get("reading_romanji", ""),
            "glossary": whole_glossary,
            "segments": breakdowns["words"]["segments"],
            "lexical_segments": breakdowns["words"]["segments"],
            "kanji_details": region_kanji_details,
            "enrichment": {
                "ginza": _ginza_enrichment(text),
            },
            "breakdowns": breakdowns,
            "units_by_id": units_by_id,
            "summary": {
                "token_count": breakdowns["words"]["summary"]["token_count"],
                "kanji_count": kanji_count,
                "lookup_hits": region_lookup_hits,
                "lookup_misses": region_lookup_misses,
            },
            "source_catalog": source_catalog(),
            "field_sources": {
                "reading_hiragana": ["sudachi"],
                "reading_romaji": ["sudachi", romaji_source],
                "reading_romanji": ["sudachi", romaji_source],
                "glossary": ["jmdict", "app_editorial"],
            },
        }

    return {
        "success": True,
        "by_region": by_region,
        "global_lookup_hits": global_lookup_hits,
        "global_lookup_misses": global_lookup_misses,
        "source": "SudachiPy + JMdict/KANJIDIC2 via KanjiAPI",
        "source_catalog": source_catalog(),
    }


__all__ = [
    "build_panel_rabbithole",
    "kana_to_romanji",
    "lookup_kanji",
    "lookup_reading",
    "lookup_text",
    "lookup_word",
    "token_plausibility",
    "tokenize_text",
]
