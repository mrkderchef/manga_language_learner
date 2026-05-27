"""Japanese Rabbithole analysis, readings, glossing, and lookup cache helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import requests

from config import BASE_DIR, KANJIAPI_BASE_URL, RABBITHOLE_GINZA_ENABLED

logger = logging.getLogger(__name__)

LOOKUP_CACHE_DIR = BASE_DIR / "backend" / "data" / "lookup_cache"
for _name in ("kanji", "words", "readings", "lookup"):
    (LOOKUP_CACHE_DIR / _name).mkdir(parents=True, exist_ok=True)

KANJI_RE = re.compile(r"[\u3400-\u9fff]")
KANA_RE = re.compile(r"[\u3040-\u30ff]")
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z\uFF21-\uFF3A\uFF41-\uFF5A]")
DIGIT_RE = re.compile(r"[0-9\uFF10-\uFF19]")

_tokenizer = None
_kakasi = None

RABBITHOLE_VERSION = "rabbithole-v6"
KANJI_LOOKUP_VERSION = "kanji-lookup-v2"
WORD_LOOKUP_VERSION = "word-lookup-v8"
LOCAL_EXTENDED_DICTIONARY_VERSION = "local-extended-dictionary-v1"

FUNCTION_GLOSSARY = {
    "は": "topic marker",
    "が": "subject marker",
    "を": "object marker",
    "に": "to / at / in",
    "へ": "toward",
    "で": "at / by / with",
    "と": "and / with / quote",
    "も": "also",
    "の": "of / possessive",
    "から": "from / because",
    "まで": "until / up to",
    "より": "than / from",
    "や": "and / such as",
    "か": "question marker",
    "ね": "seeking agreement",
    "よ": "emphasis",
    "な": "soft emphasis",
    "ぞ": "emphatic assertion",
    "さ": "casual emphasis",
    "て": "and then / -te form",
    "で": "and then / by means of",
    "た": "past",
    "だ": "copula",
    "です": "polite copula",
    "ます": "polite verb ending",
    "ない": "not",
    "ん": "explanatory / contraction",
    "のだ": "explanatory",
}

LEXICAL_GLOSSARY = {
    "夢": "dream",
    "果て": "the end",
    "さん": "honorific",
    "ちゃん": "affectionate suffix",
    "くん": "familiar honorific",
    "ガタッ": "clatter / thud",
    "ドキドキ": "heartbeat / nervous excitement",
}

PRIORITY_LABELS = {
    "ichi": "common term",
    "news": "news frequency",
    "nf": "frequency rank",
    "spec": "specialized/common",
    "gai": "common loanword",
}

POS_LABELS = {
    "名詞": "noun",
    "動詞": "verb",
    "形容詞": "adjective",
    "副詞": "adverb",
    "連体詞": "pre-noun adjective",
    "接続詞": "conjunction",
    "感動詞": "interjection",
    "助詞": "particle",
    "助動詞": "auxiliary",
    "接頭辞": "prefix",
    "接尾辞": "suffix",
    "補助記号": "symbol / punctuation",
}

GRAMMAR_DETAILS = {
    "は": {
        "label": "topic marker",
        "glosses": [
            "marks the sentence topic",
            "often adds contrast depending on context",
            "written は but pronounced わ in this particle use",
        ],
        "tags": ["grammar", "particle", "JLPT N5 core"],
    },
    "が": {
        "label": "subject marker",
        "glosses": [
            "marks grammatical subject",
            "often introduces new or focused information",
        ],
        "tags": ["grammar", "particle", "JLPT N5 core"],
    },
    "を": {
        "label": "object marker",
        "glosses": [
            "marks direct object",
            "pronounced お in particle use",
        ],
        "tags": ["grammar", "particle", "JLPT N5 core"],
    },
    "に": {
        "label": "target / time marker",
        "glosses": [
            "marks destination, indirect object, or point in time",
            "can also mark purpose with movement verbs",
        ],
        "tags": ["grammar", "particle"],
    },
    "で": {
        "label": "location / means marker",
        "glosses": [
            "marks location of action",
            "marks method, instrument, or material",
        ],
        "tags": ["grammar", "particle"],
    },
    "へ": {
        "label": "direction marker",
        "glosses": [
            "marks direction toward a destination",
            "written へ but pronounced え in particle use",
        ],
        "tags": ["grammar", "particle"],
    },
    "の": {
        "label": "genitive linker",
        "glosses": [
            "links nouns (roughly \"of\")",
            "can nominalize clauses in some constructions",
        ],
        "tags": ["grammar", "particle", "JLPT N5 core"],
    },
    "と": {
        "label": "and / with / quotation marker",
        "glosses": [
            "joins nouns as \"and\"",
            "marks quoted speech or thought",
            "can mark companion as \"with\"",
        ],
        "tags": ["grammar", "particle"],
    },
    "も": {
        "label": "also marker",
        "glosses": [
            "adds meaning like \"also / too\"",
            "can replace は, が, or を depending on sentence role",
        ],
        "tags": ["grammar", "particle"],
    },
    "か": {
        "label": "question marker",
        "glosses": [
            "marks a direct question",
            "can also create alternatives or indefinites",
        ],
        "tags": ["grammar", "particle"],
    },
    "から": {
        "label": "from / because marker",
        "glosses": [
            "marks origin or starting point",
            "can mark reason (\"because\")",
        ],
        "tags": ["grammar", "particle"],
    },
    "まで": {
        "label": "until / up to marker",
        "glosses": [
            "marks end point in time or space",
            "often pairs with から",
        ],
        "tags": ["grammar", "particle"],
    },
    "より": {
        "label": "comparison marker",
        "glosses": [
            "marks comparison baseline (\"than\")",
            "can mark source in formal/literary style",
        ],
        "tags": ["grammar", "particle"],
    },
    "や": {
        "label": "non-exhaustive listing marker",
        "glosses": [
            "lists representative items (\"A, B, and so on\")",
        ],
        "tags": ["grammar", "particle"],
    },
    "ね": {
        "label": "agreement-seeking ending",
        "glosses": [
            "softly seeks listener agreement",
        ],
        "tags": ["grammar", "sentence ending"],
    },
    "よ": {
        "label": "assertive ending",
        "glosses": [
            "adds emphasis or new information for the listener",
        ],
        "tags": ["grammar", "sentence ending"],
    },
    "だ": {
        "label": "plain copula",
        "glosses": [
            "plain assertive copula",
            "used in casual/plain style",
        ],
        "tags": ["grammar", "auxiliary", "copula"],
    },
    "です": {
        "label": "polite copula",
        "glosses": [
            "polite copula",
            "used to raise formality",
        ],
        "tags": ["grammar", "auxiliary", "copula"],
    },
    "ます": {
        "label": "polite verb ending",
        "glosses": [
            "politeness marker attached to verb stem",
            "appears in non-past affirmative polite forms",
        ],
        "tags": ["grammar", "auxiliary", "politeness"],
    },
    "ない": {
        "label": "negative auxiliary",
        "glosses": [
            "marks negation for verbs/adjectival predicates",
        ],
        "tags": ["grammar", "auxiliary", "negation"],
    },
    "た": {
        "label": "past/perfect auxiliary",
        "glosses": [
            "marks past or completed action/state",
        ],
        "tags": ["grammar", "auxiliary"],
    },
    "て": {
        "label": "te-form linker",
        "glosses": [
            "links actions/clauses",
            "can support requests, progressive forms, and many fixed constructions",
        ],
        "tags": ["grammar", "verb form"],
    },
}

SYMBOL_DETAILS = {
    "、": {
        "label": "Japanese comma",
        "glosses": ["pause separator in Japanese writing"],
        "tags": ["symbol", "punctuation"],
    },
    "。": {
        "label": "Japanese period",
        "glosses": ["sentence terminator in Japanese writing"],
        "tags": ["symbol", "punctuation"],
    },
    "「": {
        "label": "opening quote",
        "glosses": ["opens Japanese quotation marks"],
        "tags": ["symbol", "punctuation"],
    },
    "」": {
        "label": "closing quote",
        "glosses": ["closes Japanese quotation marks"],
        "tags": ["symbol", "punctuation"],
    },
    "『": {
        "label": "opening inner quote",
        "glosses": ["opens nested/emphatic Japanese quotes"],
        "tags": ["symbol", "punctuation"],
    },
    "』": {
        "label": "closing inner quote",
        "glosses": ["closes nested/emphatic Japanese quotes"],
        "tags": ["symbol", "punctuation"],
    },
    "・": {
        "label": "middle dot",
        "glosses": ["separator in names or borrowed compounds"],
        "tags": ["symbol", "punctuation"],
    },
    "ー": {
        "label": "long vowel mark",
        "glosses": ["extends the previous vowel sound"],
        "tags": ["symbol", "kana mark"],
    },
    "…": {
        "label": "ellipsis",
        "glosses": ["pause, trailing thought, or silence"],
        "tags": ["symbol", "punctuation"],
    },
    "？": {
        "label": "question mark",
        "glosses": ["question punctuation"],
        "tags": ["symbol", "punctuation"],
    },
    "！": {
        "label": "exclamation mark",
        "glosses": ["exclamatory punctuation"],
        "tags": ["symbol", "punctuation"],
    },
}

SMALL_KANA_NOTES = {
    "ぁ": "small a kana; used in stylistic spellings/phonetic effects",
    "ぃ": "small i kana; used in stylistic spellings/phonetic effects",
    "ぅ": "small u kana; used in stylistic spellings/phonetic effects",
    "ぇ": "small e kana; used in stylistic spellings/phonetic effects",
    "ぉ": "small o kana; used in stylistic spellings/phonetic effects",
    "ゃ": "small ya kana; combines with i-row kana for contracted sounds",
    "ゅ": "small yu kana; combines with i-row kana for contracted sounds",
    "ょ": "small yo kana; combines with i-row kana for contracted sounds",
    "ゎ": "small wa kana; rare in modern standard text",
    "っ": "small tsu (sokuon); marks consonant gemination",
}


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


def kana_to_romanji(text: str) -> str:
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
            tokens.append({
                "id": f"tok_{index:03d}",
                "surface": surface,
                "lemma": morpheme.dictionary_form(),
                "reading_hiragana": reading_hira,
                "reading_romanji": kana_to_romanji(reading_hira),
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
    return {
        "reading_hiragana": reading_hiragana,
        "reading_romanji": kana_to_romanji(reading_hiragana),
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


def lookup_kanji(character: str) -> dict[str, Any]:
    ch = (character or "")[:1]
    cached = _read_cache("kanji", ch)
    if cached and cached.get("schema_version") == KANJI_LOOKUP_VERSION:
        return cached
    api_data = _fetch_kanjiapi(f"/kanji/{ch}") if ch else None
    if isinstance(api_data, dict):
        data = {
            "type": "kanji",
            "schema_version": KANJI_LOOKUP_VERSION,
            "kanji": api_data.get("kanji") or ch,
            "meanings": api_data.get("meanings", []),
            "kun_readings": api_data.get("kun_readings", []),
            "on_readings": api_data.get("on_readings", []),
            "name_readings": api_data.get("name_readings", []),
            "stroke_count": api_data.get("stroke_count"),
            "grade": api_data.get("grade"),
            "jlpt": api_data.get("jlpt"),
            "unicode": api_data.get("unicode"),
            "heisig_en": api_data.get("heisig_en"),
            "freq_mainichi_shinbun": api_data.get("freq_mainichi_shinbun"),
            "notes": api_data.get("notes", []),
            "source": "kanjiapi.dev",
        }
        return _write_cache("kanji", ch, data)
    if cached:
        data = {
            "type": "kanji",
            "schema_version": KANJI_LOOKUP_VERSION,
            "kanji": cached.get("kanji") or ch,
            "meanings": cached.get("meanings", []),
            "kun_readings": cached.get("kun_readings", []),
            "on_readings": cached.get("on_readings", []),
            "name_readings": cached.get("name_readings", []),
            "stroke_count": cached.get("stroke_count"),
            "grade": cached.get("grade"),
            "jlpt": cached.get("jlpt"),
            "unicode": cached.get("unicode"),
            "heisig_en": cached.get("heisig_en"),
            "freq_mainichi_shinbun": cached.get("freq_mainichi_shinbun"),
            "notes": cached.get("notes", []),
            "source": cached.get("source") or "kanjiapi.dev-cache",
        }
        return _write_cache("kanji", ch, data)
    data = {
        "type": "kanji",
        "schema_version": KANJI_LOOKUP_VERSION,
        "kanji": ch,
        "meanings": [],
        "kun_readings": [],
        "on_readings": [],
        "name_readings": [],
        "stroke_count": None,
        "grade": None,
        "jlpt": None,
        "unicode": None,
        "heisig_en": None,
        "freq_mainichi_shinbun": None,
        "notes": [],
        "source": "central-cache-placeholder",
    }
    return _write_cache("kanji", ch, data)


def _word_lookup_cache_key(text: str, reading_hiragana: str = "") -> str:
    reading = (reading_hiragana or "").strip()
    return f"{text.strip()}::{reading}" if reading else text.strip()


def lookup_word(text: str, reading_hiragana: str = "") -> dict[str, Any]:
    key = (text or "").strip()
    contextual_reading = (reading_hiragana or "").strip()
    cache_key = _word_lookup_cache_key(key, contextual_reading)
    cached = _read_cache("words", cache_key)
    if cached and cached.get("schema_version") == WORD_LOOKUP_VERSION:
        return cached
    tokenized = tokenize_text(key)
    resolved_reading = contextual_reading or str(tokenized.get("reading_hiragana") or "")
    entries, candidate_count = _fetch_dictionary_entries(key, resolved_reading)
    data = {
        "type": "word",
        "schema_version": WORD_LOOKUP_VERSION,
        "text": key,
        "reading_hiragana": resolved_reading,
        "reading_romanji": kana_to_romanji(resolved_reading) if contextual_reading else tokenized.get("reading_romanji", ""),
        "tokens": tokenized.get("tokens", []),
        "entries": entries,
        "candidate_count": candidate_count,
        "source": "kanjiapi.dev:/words (JMdict-backed) + SudachiPy" if entries else "SudachiPy (no remote match)",
    }
    if JAPANESE_RE.search(key) and not entries and candidate_count == 0:
        return data
    return _write_cache("words", cache_key, data)


def lookup_reading(reading: str) -> dict[str, Any]:
    key = (reading or "").strip()
    cached = _read_cache("readings", key)
    if cached:
        return cached
    api_data = _fetch_kanjiapi(f"/reading/{key}") if key else None
    api_dict = api_data if isinstance(api_data, dict) else None
    data = {
        "type": "reading",
        "reading": key,
        "romanji": kana_to_romanji(key),
        "kanji": api_dict.get("main_kanji", []) if api_dict else [],
        "name_kanji": api_dict.get("name_kanji", []) if api_dict else [],
        "source": "kanjiapi.dev" if api_dict else "local",
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
    for prefix, label in PRIORITY_LABELS.items():
        if value.startswith(prefix):
            return label
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
    if kanji:
        return kanji
    return [text] if text else []


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
        "source": "kanjiapi.dev:/words (JMdict-backed)",
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
    return not _is_symbol_token(token)


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
        "source": f"local:{LOCAL_EXTENDED_DICTIONARY_VERSION}",
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
        "source": f"local:{LOCAL_EXTENDED_DICTIONARY_VERSION}",
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
        "source": f"local:{LOCAL_EXTENDED_DICTIONARY_VERSION}",
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
    merged = _dedupe_entries_by_variant([*(remote_entries or []), *(local_entries or [])])
    return sorted(merged, key=lambda item: int(item.get("score", 0)), reverse=True)


def _dictionary_source_summary(word_lookup: dict[str, Any], entries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    sources = _dedupe_strings([
        str(word_lookup.get("source") or "").strip(),
        *[str(entry.get("source") or "").strip() for entry in entries],
    ])
    if not sources:
        return "", []
    return " + ".join(sources), sources


def _primary_meaning(token: dict[str, Any], kind: str, word_meanings: list[str], kanji_details: list[dict[str, Any]]) -> tuple[str, list[str]]:
    surface = str(token.get("surface") or "")
    lemma = str(token.get("lemma") or surface)
    pos = token.get("pos") or []
    joined = "/".join(str(part) for part in pos)
    glossary_keys = [surface, lemma]
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

    if word_meanings:
        return word_meanings[0], word_meanings[1:]

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
            "romaji": unit.get("reading_romanji", ""),
        },
        "meaning": {
            "glossary": unit.get("primary_meaning", ""),
            "alternate_glossary": unit.get("alternate_meanings", []),
        },
        "grammar": {
            "segment_type": unit.get("kind", ""),
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
    unit = {
        "id": unit_id,
        "kind": "kanji",
        "text": character,
        "start": start,
        "end": end,
        "lemma": character,
        "reading_hiragana": "",
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
            "schema_version": kanji_lookup.get("schema_version"),
            "meanings": kanji_lookup.get("meanings", []),
            "kun_readings": kanji_lookup.get("kun_readings", []),
            "on_readings": kanji_lookup.get("on_readings", []),
            "name_readings": kanji_lookup.get("name_readings", []),
            "stroke_count": kanji_lookup.get("stroke_count"),
            "grade": kanji_lookup.get("grade"),
            "jlpt": kanji_lookup.get("jlpt"),
            "unicode": kanji_lookup.get("unicode"),
            "heisig_en": kanji_lookup.get("heisig_en"),
            "freq_mainichi_shinbun": kanji_lookup.get("freq_mainichi_shinbun"),
            "notes": kanji_lookup.get("notes", []),
        },
        "children": [],
    }
    unit["feature_groups"] = _unit_feature_groups(unit)
    return unit


def _whole_unit(text: str, reading_hiragana: str, reading_romanji: str, glossary: str, token_count: int, kanji_count: int) -> dict[str, Any]:
    unit = {
        "id": "whole_000",
        "kind": "whole",
        "text": text,
        "start": 0,
        "end": len(text),
        "lemma": text,
        "reading_hiragana": reading_hiragana,
        "reading_romanji": reading_romanji,
        "pos": [],
        "primary_meaning": glossary,
        "alternate_meanings": [],
        "children": [],
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
        "text": unit.get("text", ""),
        "hiragana": unit.get("reading_hiragana", ""),
        "romanji": unit.get("reading_romanji", ""),
        "gloss": unit.get("primary_meaning", ""),
        "start": unit.get("start", 0),
        "end": unit.get("end", 0),
        "children": unit.get("children", []),
        "feature_groups": unit.get("feature_groups", {}),
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

    for index, ann in enumerate(annotations, start=1):
        text = str(ann.get("text") or "")
        region_id = ann.get("region_id") or ann.get("id") or f"region_{index:04d}"
        word_tokenized = tokenize_text(text, "C")
        morpheme_tokenized = tokenize_text(text, "A")
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
                unit = {
                    "id": unit_id,
                    "kind": kind,
                    "text": surface,
                    "start": int(token.get("start", 0)),
                    "end": int(token.get("end", 0)),
                    "lemma": lemma,
                    "reading_hiragana": reading_hiragana,
                    "reading_romanji": str(token.get("reading_romanji") or ""),
                    "pos": list(token.get("pos") or []),
                    "part_of_speech_labels": _pos_labels(list(token.get("pos") or [])),
                    "primary_meaning": primary_meaning,
                    "alternate_meanings": alternate_meanings,
                    "dictionary_entries": dictionary_entries[:4],
                    "dictionary_source": dictionary_source,
                    "dictionary_sources": dictionary_sources,
                    "dictionary_candidate_count": (word_lookup or {}).get("candidate_count", 0),
                    "grammar_detail": _grammar_detail_for_token(token),
                    "kanji_details": [
                        units_by_id[child_id].get("kanji_details", {})
                        for child_id in child_ids
                        if child_id in units_by_id
                    ],
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
        breakdowns["morphemes"] = build_breakdown("morphemes", morpheme_tokenized.get("tokens", []))

        whole_glossary = breakdowns["words"]["glossary"] or breakdowns["morphemes"]["glossary"]
        kanji_count = len(KANJI_RE.findall(text))
        whole_unit = _whole_unit(
            text,
            word_tokenized.get("reading_hiragana", ""),
            word_tokenized.get("reading_romanji", ""),
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
            "reading_romanji": word_tokenized.get("reading_romanji", ""),
            "glossary": whole_glossary,
            "segments": breakdowns["words"]["segments"],
            "lexical_segments": breakdowns["words"]["segments"],
            "morpheme_segments": breakdowns["morphemes"]["segments"],
            "kanji_details": region_kanji_details,
            "enrichment": {
                "ginza": _ginza_enrichment(text),
            },
            "breakdowns": breakdowns,
            "units_by_id": units_by_id,
            "summary": {
                "token_count": breakdowns["words"]["summary"]["token_count"],
                "morpheme_count": breakdowns["morphemes"]["summary"]["token_count"],
                "kanji_count": kanji_count,
                "lookup_hits": region_lookup_hits,
                "lookup_misses": region_lookup_misses,
            },
        }

    return {
        "success": True,
        "by_region": by_region,
        "global_lookup_hits": global_lookup_hits,
        "global_lookup_misses": global_lookup_misses,
        "source": f"sudachi+kanjiapi-cache:{RABBITHOLE_VERSION}",
    }


__all__ = [
    "RABBITHOLE_VERSION",
    "build_panel_rabbithole",
    "kana_to_romanji",
    "lookup_kanji",
    "lookup_reading",
    "lookup_text",
    "lookup_word",
    "token_plausibility",
    "tokenize_text",
]
