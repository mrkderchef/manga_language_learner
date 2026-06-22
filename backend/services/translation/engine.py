"""Ollama translation for manga dialogue."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests

from config import OLLAMA_BASE_URL, OLLAMA_TEXT_MODEL, TRANSLATION_TARGET_LANGUAGE

logger = logging.getLogger(__name__)

PROMPT_VERSION = "manga-dialogue-v4"
OLLAMA_TAGS_TIMEOUT_SECONDS = 2
OLLAMA_GENERATE_TIMEOUT_SECONDS = 180
MANGA_CONTEXT_LINE_LIMIT = 10
MANGA_TARGET_CHUNK_SIZE = 8
MANGA_DIALOGUE_STRATEGY = "manga-dialogue-context-window-v1"
_OLLAMA_MODELS_CACHE: dict[str, Any] = {"ts": 0.0, "models": [], "last_error": None}

MANGA_LOCALIZER_SYSTEM_PROMPT = """You are a professional Japanese manga localizer.
Translate Japanese dialogue naturally while preserving character voice, emotion, register, and panel-to-panel context.
Use context-only lines only to resolve meaning. Do not translate or output context-only lines unless they are also targets."""

OLLAMA_JSON_FORMAT = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "translation": {"type": "string"},
                },
                "required": ["id", "translation"],
            },
        }
    },
    "required": ["translations"],
}


def lang_name(code: str) -> str:
    return {"en": "English", "de": "German", "fr": "French", "es": "Spanish"}.get(code, code)


def list_ollama_models(force: bool = False) -> list[str]:
    now = time.monotonic()
    if not force and now - float(_OLLAMA_MODELS_CACHE.get("ts", 0.0)) < 30:
        return list(_OLLAMA_MODELS_CACHE.get("models", []))
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=OLLAMA_TAGS_TIMEOUT_SECONDS)
        response.raise_for_status()
        models = [m.get("name", "") for m in response.json().get("models", []) if m.get("name")]
        _OLLAMA_MODELS_CACHE.update({"ts": now, "models": models, "last_error": None})
        return models
    except Exception:
        _OLLAMA_MODELS_CACHE.update({
            "ts": now,
            "models": [],
            "last_error": "Could not reach /api/tags; Ollama model discovery is unavailable",
        })
        return []


def preferred_ollama_text_model() -> str:
    models = list_ollama_models()
    if OLLAMA_TEXT_MODEL and OLLAMA_TEXT_MODEL in models:
        return OLLAMA_TEXT_MODEL
    if not models:
        return OLLAMA_TEXT_MODEL
    return models[0]


def ollama_model_discovery_status(force: bool = False) -> dict[str, Any]:
    models = list_ollama_models(force=force)
    return {
        "models": models,
        "preferred_model": preferred_ollama_text_model(),
        "discovery_available": bool(models),
        "discovery_error": _OLLAMA_MODELS_CACHE.get("last_error"),
        "can_attempt_generation": bool(models or OLLAMA_TEXT_MODEL),
        "base_url_configured": bool(OLLAMA_BASE_URL),
    }


def ollama_available(model: str | None = None) -> bool:
    models = list_ollama_models()
    if not models:
        return False
    return model in models if model else True


def _ollama_options(temperature: float) -> dict[str, Any]:
    return {
        "temperature": temperature,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
        "num_predict": 4096,
    }


def _resolve_ollama_model(model: str | None = None) -> str:
    installed = list_ollama_models()
    if model:
        if installed and model not in installed:
            raise ValueError(f"Ollama model is not installed: {model}")
        return model

    use_model = preferred_ollama_text_model()
    if use_model:
        return use_model
    raise ValueError("Could not discover installed Ollama models; translation is unavailable")


def call_ollama(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.1,
    system: str | None = None,
    response_format: dict[str, Any] | str | None = None,
) -> str:
    use_model = model or _resolve_ollama_model()
    payload: dict[str, Any] = {
        "model": use_model,
        "prompt": prompt,
        "stream": False,
        "options": _ollama_options(temperature),
    }
    if system:
        payload["system"] = system
    if response_format:
        payload["format"] = response_format
    response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=OLLAMA_GENERATE_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def _parse_json_value(text: str) -> Any | None:
    stripped = (text or "").strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start:end + 1])
            except Exception:
                continue
    return None


def _parse_json_array(text: str) -> list[str] | None:
    value = _parse_json_value(text)
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict) and isinstance(value.get("translations"), list):
        entries = value["translations"]
        if all(not isinstance(item, dict) for item in entries):
            return [str(item) for item in entries]
    return None


def _parse_translation_map(text: str) -> dict[str, str] | None:
    value = _parse_json_value(text)
    if isinstance(value, dict):
        entries = value.get("translations")
        if isinstance(entries, list):
            mapped: dict[str, str] = {}
            for item in entries:
                if not isinstance(item, dict):
                    continue
                region_id = item.get("id") or item.get("region_id")
                translation = item.get("translation") or item.get("translated") or item.get("text")
                if region_id is not None and translation is not None:
                    mapped[str(region_id)] = str(translation)
            return mapped
        mapped = {
            str(key): str(value)
            for key, value in value.items()
            if key not in {"translations"} and isinstance(value, str)
        }
        return mapped or None
    return None


def _parse_numbered_list(text: str) -> list[str]:
    items = []
    for line in text.strip().splitlines():
        match = re.match(r"^\s*\d+[\.\):]\s*(.+?)\s*$", line)
        if match:
            items.append(match.group(1))
    return items


def _style_instruction(style: str) -> str:
    if style == "literal":
        return "Translate accurately and stay close to the Japanese wording."
    if style == "learner":
        return "Translate naturally, but keep wording helpful for a Japanese learner."
    return "Translate naturally as manga dialogue, preserving emotion and tone."


def _normalize_context_units(texts: list[str], context_units: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    units = []
    for index, text in enumerate(texts):
        source = context_units[index] if context_units and index < len(context_units) and isinstance(context_units[index], dict) else {}
        region_id = source.get("region_id") or source.get("id") or f"line_{index + 1}"
        unit = {
            "index": index,
            "id": str(region_id),
            "text": str(source.get("text") or text),
            "reading_order": int(source.get("reading_order") or index + 1),
            "orientation": source.get("orientation") or source.get("recognized_orientation") or ("vertical" if source.get("vertical") else "horizontal"),
        }
        box = source.get("box") or source.get("bbox")
        if box:
            unit["box"] = box
        units.append(unit)
    return units


def _translation_windows(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    count = len(units)
    if count <= MANGA_CONTEXT_LINE_LIMIT:
        indices = list(range(count))
        return [{
            "target_indices": indices,
            "context_indices": [],
            "line_indices": indices,
        }]

    windows = []
    for start in range(0, count, MANGA_TARGET_CHUNK_SIZE):
        end = min(start + MANGA_TARGET_CHUNK_SIZE, count)
        target_indices = list(range(start, end))
        context_indices: list[int] = []
        slots = max(0, MANGA_CONTEXT_LINE_LIMIT - len(target_indices))
        before = start - 1
        after = end
        while slots > 0 and (before >= 0 or after < count):
            if before >= 0:
                context_indices.insert(0, before)
                before -= 1
                slots -= 1
            if slots > 0 and after < count:
                context_indices.append(after)
                after += 1
                slots -= 1
        line_indices = sorted(set(target_indices + context_indices))
        windows.append({
            "target_indices": target_indices,
            "context_indices": context_indices,
            "line_indices": line_indices,
        })
    return windows


def _prompt_line(unit: dict[str, Any], role: str) -> dict[str, Any]:
    line = {
        "id": unit["id"],
        "role": role,
        "order": unit["reading_order"],
        "orientation": unit.get("orientation"),
        "text": unit["text"],
    }
    if unit.get("box"):
        line["box"] = unit["box"]
    return line


def _build_manga_dialogue_prompt(
    units: list[dict[str, Any]],
    window: dict[str, Any],
    target_lang: str,
    style: str,
) -> str:
    target_ids = [units[index]["id"] for index in window["target_indices"]]
    target_set = set(window["target_indices"])
    lines = [
        _prompt_line(units[index], "target" if index in target_set else "context")
        for index in window["line_indices"]
    ]
    return f"""Translate Japanese manga dialogue to {lang_name(target_lang)}.
{_style_instruction(style)}
Translate only lines where role is "target". Use role "context" lines only for surrounding meaning.
Return only JSON in this shape: {{"translations":[{{"id":"<target id>","translation":"<translation>"}}]}}
Output exactly these target ids in order: {json.dumps(target_ids, ensure_ascii=False)}

Lines:
{json.dumps(lines, ensure_ascii=False, separators=(",", ":"))}"""


def _translations_from_response(raw: str, target_units: list[dict[str, Any]]) -> list[str]:
    mapped = _parse_translation_map(raw)
    if mapped is not None:
        return [mapped.get(unit["id"], "") for unit in target_units]
    legacy = _parse_json_array(raw) or _parse_numbered_list(raw)
    translations = list(legacy)
    while len(translations) < len(target_units):
        translations.append("")
    return translations[:len(target_units)]


def _chunk_debug(units: list[dict[str, Any]], windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "line_count": len(window["line_indices"]),
            "target_ids": [units[index]["id"] for index in window["target_indices"]],
            "context_ids": [units[index]["id"] for index in window["context_indices"]],
        }
        for window in windows
    ]


def translate_batch(
    texts: list[str],
    target_lang: str | None = None,
    engine: str = "ollama",
    model: str | None = None,
    style: str = "natural",
    temperature: float = 0.1,
    context_units: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Translate OCR text through Ollama with explicit manga context windows."""
    target_lang = target_lang or TRANSLATION_TARGET_LANGUAGE
    requested_engine = engine
    if not texts:
        return _result([], requested_engine, "ollama", model, target_lang, style, False)

    if engine != "ollama":
        raise ValueError(f"Unsupported translation engine: {engine}. Only Ollama translation is supported.")

    use_model = _resolve_ollama_model(model)
    units = _normalize_context_units(texts, context_units)
    windows = _translation_windows(units)
    translations = [""] * len(units)

    for window in windows:
        target_units = [units[index] for index in window["target_indices"]]
        prompt = _build_manga_dialogue_prompt(units, window, target_lang, style)
        raw = call_ollama(
            prompt,
            model=use_model,
            temperature=temperature,
            system=MANGA_LOCALIZER_SYSTEM_PROMPT,
            response_format=OLLAMA_JSON_FORMAT,
        )
        chunk_translations = _translations_from_response(raw, target_units)
        for index, translated in zip(window["target_indices"], chunk_translations):
            translations[index] = translated

    result = _result(translations, requested_engine, "ollama", use_model, target_lang, style, False)
    result["translation_prompt_payload"] = {
        "engine": "ollama",
        "model": use_model,
        "target_lang": target_lang,
        "style": style,
        "temperature": temperature,
        "strategy": "single_panel" if len(windows) == 1 and len(units) <= MANGA_CONTEXT_LINE_LIMIT else MANGA_DIALOGUE_STRATEGY,
        "text_blocks": len(units),
        "target_chunk_size": MANGA_TARGET_CHUNK_SIZE,
        "max_context_lines": MANGA_CONTEXT_LINE_LIMIT,
        "chunk_count": len(windows),
        "chunks": _chunk_debug(units, windows),
    }
    return result


def translate_text(text: str, **options) -> dict[str, Any]:
    result = translate_batch([text], **options)
    return {
        "success": True,
        "source": text,
        "translated": result["translations"][0] if result["translations"] else "",
        **{k: v for k, v in result.items() if k != "translations"},
    }


def _result(translations, requested, used, model, target_lang, style, fallback_used):
    return {
        "success": True,
        "translations": translations,
        "translation_engine_requested": requested,
        "translation_engine_used": used,
        "translation_model": model,
        "translation_target_lang": target_lang,
        "translation_style": style,
        "translation_prompt_version": PROMPT_VERSION,
        "fallback_used": fallback_used,
    }


def engine_status() -> list[dict[str, Any]]:
    models = list_ollama_models()
    return [{
        "id": "ollama",
        "label": "Ollama",
        "available": bool(models),
        "discovery_available": bool(models),
        "discovery_error": _OLLAMA_MODELS_CACHE.get("last_error"),
        "default": True,
        "models": models,
        "preferred_model": preferred_ollama_text_model(),
    }]
