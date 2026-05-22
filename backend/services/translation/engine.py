"""Transparent, selectable translation engines.

Provides lower-level translation helpers used by the app. Supports Ollama
and Gemini backends and exposes a small discovery API.
"""

from __future__ import annotations

import json
import logging
import re
import time
import importlib
from typing import Any

import requests

from config import GEMINI_API_KEY, OLLAMA_BASE_URL, OLLAMA_TEXT_MODEL, TRANSLATION_TARGET_LANGUAGE

logger = logging.getLogger(__name__)

PROMPT_VERSION = "manga-dialogue-v2"
_OLLAMA_MODELS_CACHE: dict[str, Any] = {"ts": 0.0, "models": [], "last_error": None}


def lang_name(code: str) -> str:
	return {"en": "English", "de": "German", "fr": "French", "es": "Spanish"}.get(code, code)


def list_ollama_models(force: bool = False) -> list[str]:
	now = time.monotonic()
	if not force and now - float(_OLLAMA_MODELS_CACHE.get("ts", 0.0)) < 30:
		return list(_OLLAMA_MODELS_CACHE.get("models", []))
	try:
		response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
		response.raise_for_status()
		models = [m.get("name", "") for m in response.json().get("models", []) if m.get("name")]
		_OLLAMA_MODELS_CACHE.update({"ts": now, "models": models, "last_error": None})
		return models
	except Exception:
		_OLLAMA_MODELS_CACHE.update({"ts": now, "models": [], "last_error": "Could not reach /api/tags; generation may still work with the configured default model"})
		return []


def ollama_model_discovery_status() -> dict[str, Any]:
	models = list_ollama_models()
	return {
		"models": models,
		"preferred_model": preferred_ollama_text_model(),
		"discovery_available": bool(models),
		"discovery_error": _OLLAMA_MODELS_CACHE.get("last_error"),
		"can_attempt_generation": True,
		"base_url_configured": bool(OLLAMA_BASE_URL),
	}


def preferred_ollama_text_model() -> str:
	models = list_ollama_models()
	if OLLAMA_TEXT_MODEL and OLLAMA_TEXT_MODEL in models:
		return OLLAMA_TEXT_MODEL
	for pref in ("qwen2.5", "llama3.1", "llama3", "mistral", "gemma"):
		for model in models:
			if pref in model:
				return model
	return models[0] if models else OLLAMA_TEXT_MODEL


def ollama_available(model: str | None = None) -> bool:
	models = list_ollama_models()
	if not models:
		return False
	return not model or model in models


def call_ollama(prompt: str, model: str | None = None, temperature: float = 0.1) -> str:
	use_model = model or preferred_ollama_text_model()
	payload = {
		"model": use_model,
		"prompt": prompt,
		"stream": False,
		"options": {"temperature": temperature, "num_predict": 4096},
	}
	response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=None)
	response.raise_for_status()
	return response.json().get("response", "").strip()


def _parse_json_array(text: str) -> list[str] | None:
	match = re.search(r"\[.*\]", text, re.DOTALL)
	if not match:
		return None
	try:
		value = json.loads(match.group())
	except Exception:
		return None
	if isinstance(value, list):
		return [str(item) for item in value]
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


def translate_batch(
	texts: list[str],
	target_lang: str | None = None,
	engine: str = "ollama",
	model: str | None = None,
	style: str = "natural",
	temperature: float = 0.1,
) -> dict[str, Any]:
	"""Translate text with no silent fallback. Returns metadata for transparency."""
	target_lang = target_lang or TRANSLATION_TARGET_LANGUAGE
	if engine != "ollama":
		model = None
	if not texts:
		return _result([], engine, engine, model, target_lang, style, False)

	if engine == "none":
		return _result([""] * len(texts), engine, "none", None, target_lang, style, False)

	if engine == "gemini":
		return _translate_gemini(texts, target_lang, model, style, temperature)

	if engine != "ollama":
		raise ValueError(f"Unsupported translation engine: {engine}")

	installed = list_ollama_models()
	if model:
		if not installed:
			raise ValueError("Cannot use an explicit Ollama model because model discovery is unavailable")
		if model not in installed:
			raise ValueError(f"Ollama model is not installed: {model}")
	use_model = model or preferred_ollama_text_model()
	# If /api/tags is unavailable or slow, still try /api/generate with the
	# configured default model. Explicit user-selected models require discovery
	# so arbitrary custom model names are not accepted.

	numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(texts))
	prompt = f"""You are translating Japanese manga dialogue to {lang_name(target_lang)}.
{_style_instruction(style)}
Return ONLY a JSON array of translations in the same order.

Japanese texts:
{numbered}"""
	raw = call_ollama(prompt, model=use_model, temperature=temperature)
	translations = _parse_json_array(raw) or _parse_numbered_list(raw)
	while len(translations) < len(texts):
		translations.append("")
	return _result(translations[:len(texts)], engine, "ollama", use_model, target_lang, style, False)


def translate_text(text: str, **options) -> dict[str, Any]:
	result = translate_batch([text], **options)
	return {
		"success": True,
		"source": text,
		"translated": result["translations"][0] if result["translations"] else "",
		**{k: v for k, v in result.items() if k != "translations"},
	}


def _translate_gemini(texts: list[str], target_lang: str, model: str | None, style: str, temperature: float) -> dict[str, Any]:
	if not GEMINI_API_KEY:
		raise ValueError("Gemini is not configured")
	try:
		genai = importlib.import_module("google.genai")
		types = importlib.import_module("google.genai.types")
	except Exception as exc:
		raise ValueError(f"Gemini dependencies are unavailable: {exc}") from exc

	client = genai.Client(api_key=GEMINI_API_KEY)
	use_model = model or "gemini-2.0-flash"
	numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(texts))
	prompt = f"""Translate these Japanese manga text blocks to {lang_name(target_lang)}.
{_style_instruction(style)}
Return ONLY a JSON array of translations in the same order.

{numbered}"""
	response = client.models.generate_content(
		model=use_model,
		contents=prompt,
		config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=4096),
	)
	translations = _parse_json_array(response.text.strip()) or _parse_numbered_list(response.text)
	while len(translations) < len(texts):
		translations.append("")
	return _result(translations[:len(texts)], "gemini", "gemini", use_model, target_lang, style, False)


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
	return [
		{
			"id": "ollama",
			"label": "Ollama",
			"available": True,
			"discovery_available": bool(models),
			"discovery_error": _OLLAMA_MODELS_CACHE.get("last_error"),
			"default": True,
			"models": models,
			"preferred_model": preferred_ollama_text_model(),
		},
		{
			"id": "gemini",
			"label": "Gemini",
			"available": bool(GEMINI_API_KEY),
			"default": False,
			"models": ["gemini-2.0-flash"],
		},
		{"id": "none", "label": "No translation", "available": True, "default": False, "models": []},
	]
