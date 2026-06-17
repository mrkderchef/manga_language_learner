from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
import asyncio
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config import panel_data_dir  # noqa: E402
from services.storage.image_service import ImageService  # noqa: E402
from routes import scanner  # noqa: E402
from routes import runtime as runtime_routes  # noqa: E402
from services import bootstrap  # noqa: E402
from services import logging_config  # noqa: E402
from services.translation import engine as translation_engine  # noqa: E402

SUGOI_14B = "hf.co/sugoitoolkit/Sugoi-14B-Ultra-GGUF:Q4_K_M"
SUGOI_32B = "hf.co/sugoitoolkit/Sugoi-32B-Ultra-GGUF:Q4_K_M"


def _fake_panel() -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="reader-contract-"))
    panel = temp_dir / "panel.png"
    Image.new("RGB", (120, 120), "white").save(panel)
    return panel


def _cleanup_panel(panel: Path) -> None:
    shutil.rmtree(panel.parent, ignore_errors=True)
    shutil.rmtree(panel_data_dir(panel), ignore_errors=True)


def _fake_region() -> dict:
    return {
        "x": 10,
        "y": 12,
        "width": 30,
        "height": 40,
        "vertical": True,
        "font_size": 18,
        "angle": 0,
        "lines": [],
    }


def _fake_ocr_result(*_args, **_kwargs) -> dict:
    return {
        "success": True,
        "text": "猫",
        "annotations": [{
            "id": "ann_0001",
            "region_id": scanner._region_id(_fake_region()),
            "text": "猫",
            "confidence": 0.91,
            "bbox": [[10, 12], [40, 12], [40, 52], [10, 52]],
            "char_count": 1,
            "vertical": True,
            "ocr_variant": "raw_upscaled",
            "recognized_orientation": "vertical",
            "reading_order": 1,
        }],
        "image_width": 120,
        "image_height": 120,
        "ocr_engine_used": "mangaocr",
    }


class PipelineContractTests(unittest.TestCase):
    def test_ocr_stage_does_not_translate(self):
        panel = _fake_panel()
        try:
            with patch.object(scanner, "detect_text_regions", return_value=[_fake_region()]), \
                    patch("services.recognition.ocr_provider.run_ocr", side_effect=_fake_ocr_result), \
                    patch.object(scanner.translation_engine, "translate_batch") as translate_batch:
                result = scanner._run_ocr(panel, {"use_cache": False})

            self.assertTrue(result["success"])
            self.assertEqual(result["annotations"][0]["text"], "猫")
            self.assertNotIn("translated", result["annotations"][0])
            translate_batch.assert_not_called()
        finally:
            _cleanup_panel(panel)

    def test_rabbithole_requires_existing_ocr_and_uses_cached_state(self):
        panel = _fake_panel()
        try:
            missing = scanner._run_rabbithole_existing(panel, {})
            self.assertFalse(missing["success"])
            self.assertIn("Run Scan before Rabbithole", missing["error"])

            with patch.object(scanner, "detect_text_regions", return_value=[_fake_region()]), \
                    patch("services.recognition.ocr_provider.run_ocr", side_effect=_fake_ocr_result):
                scanner._run_ocr(panel, {"use_cache": False})

            rabbit_payload = {
                "success": True,
                "by_region": {
                    scanner._region_id(_fake_region()): {
                        "summary": {"token_count": 1, "kanji_count": 1},
                        "reading_hiragana": "ねこ",
                    }
                },
                "global_lookup_hits": 1,
                "global_lookup_misses": 0,
            }
            with patch("services.recognition.ocr_provider.run_ocr") as run_ocr, \
                    patch.object(scanner.rabbithole_service, "build_panel_rabbithole", return_value=rabbit_payload):
                result = scanner._run_rabbithole_existing(panel, {"use_cache": False})

            self.assertTrue(result["success"])
            self.assertEqual(result["annotations"][0]["rabbithole"]["reading_hiragana"], "ねこ")
            run_ocr.assert_not_called()
        finally:
            _cleanup_panel(panel)

    def test_translation_requires_ocr_and_does_not_require_rabbithole(self):
        panel = _fake_panel()
        try:
            missing = scanner._run_translate_existing(panel, {})
            self.assertFalse(missing["success"])
            self.assertIn("Run Scan before Translate", missing["error"])

            with patch.object(scanner, "detect_text_regions", return_value=[_fake_region()]), \
                    patch("services.recognition.ocr_provider.run_ocr", side_effect=_fake_ocr_result):
                scanner._run_ocr(panel, {"use_cache": False})

            translation_payload = {
                "success": True,
                "translations": ["cat"],
                "translation_engine_requested": "ollama",
                "translation_engine_used": "ollama",
                "translation_model": "test-model",
            }
            with patch.object(scanner.translation_engine, "translate_batch", return_value=translation_payload) as translate_batch, \
                    patch.object(scanner.translation_engine, "preferred_ollama_text_model", return_value=SUGOI_14B), \
                    patch.object(scanner.rabbithole_service, "build_panel_rabbithole") as rabbithole, \
                    patch.object(scanner, "render_translated_panel", return_value={"translated_image_url": None, "render_warnings": []}):
                result = scanner._run_translate_existing(panel, {"use_cache": False})

            self.assertTrue(result["success"])
            self.assertEqual(result["annotations"][0]["translated"], "cat")
            call_kwargs = translate_batch.call_args.kwargs
            self.assertEqual(call_kwargs["context_units"][0]["region_id"], scanner._region_id(_fake_region()))
            self.assertEqual(call_kwargs["context_units"][0]["orientation"], "vertical")
            rabbithole.assert_not_called()
        finally:
            _cleanup_panel(panel)


class TranslationEngineTests(unittest.TestCase):
    def test_ollama_translates_small_panel_in_one_window_with_structured_json(self):
        texts = ["猫", "行くぞ", "えっ"]
        units = [
            {"region_id": f"r{i}", "text": text, "reading_order": i + 1, "vertical": True}
            for i, text in enumerate(texts)
        ]
        raw = '{"translations":[{"id":"r0","translation":"cat"},{"id":"r1","translation":"let us go"},{"id":"r2","translation":"huh"}]}'

        with patch.object(translation_engine, "list_ollama_models", return_value=[SUGOI_14B]), \
                patch.object(translation_engine, "call_ollama", return_value=raw) as call_ollama:
            result = translation_engine.translate_batch(texts, context_units=units)

        self.assertEqual(result["translations"], ["cat", "let us go", "huh"])
        self.assertEqual(result["translation_engine_used"], "ollama")
        self.assertEqual(result["translation_prompt_version"], "manga-dialogue-v4")
        self.assertEqual(result["translation_prompt_payload"]["strategy"], "single_panel")
        self.assertEqual(result["translation_prompt_payload"]["chunk_count"], 1)
        call_ollama.assert_called_once()
        self.assertIn('"role":"target"', call_ollama.call_args.args[0])

    def test_ollama_chunks_large_panel_with_context_only_lines(self):
        texts = [f"台詞{i}" for i in range(11)]
        units = [
            {"region_id": f"r{i}", "text": text, "reading_order": i + 1, "vertical": i % 2 == 0}
            for i, text in enumerate(texts)
        ]
        responses = [
            '{"translations":[' + ",".join(f'{{"id":"r{i}","translation":"t{i}"}}' for i in range(8)) + "]}",
            '{"translations":[' + ",".join(f'{{"id":"r{i}","translation":"t{i}"}}' for i in range(8, 11)) + "]}",
        ]

        with patch.object(translation_engine, "list_ollama_models", return_value=[SUGOI_14B]), \
                patch.object(translation_engine, "call_ollama", side_effect=responses) as call_ollama:
            result = translation_engine.translate_batch(texts, context_units=units)

        self.assertEqual(result["translations"], [f"t{i}" for i in range(11)])
        payload = result["translation_prompt_payload"]
        self.assertEqual(payload["strategy"], translation_engine.MANGA_DIALOGUE_STRATEGY)
        self.assertEqual(payload["chunk_count"], 2)
        self.assertTrue(all(chunk["line_count"] <= 10 for chunk in payload["chunks"]))
        self.assertIn("r8", payload["chunks"][0]["context_ids"])
        self.assertIn("r7", payload["chunks"][1]["context_ids"])
        self.assertEqual(call_ollama.call_count, 2)
        first_prompt = call_ollama.call_args_list[0].args[0]
        self.assertIn('"id":"r8","role":"context"', first_prompt)

    def test_ollama_keeps_legacy_json_array_fallback(self):
        with patch.object(translation_engine, "list_ollama_models", return_value=[SUGOI_14B]), \
                patch.object(translation_engine, "call_ollama", return_value='["cat","dog"]'):
            result = translation_engine.translate_batch(["猫", "犬"])

        self.assertEqual(result["translations"], ["cat", "dog"])

    def test_model_discovery_exposes_all_ollama_models_and_prefers_configured_then_first(self):
        models = ["other-local-model:1b", SUGOI_32B, SUGOI_14B]
        with patch.object(translation_engine, "list_ollama_models", return_value=models):
            status = translation_engine.ollama_model_discovery_status()
            engines = translation_engine.engine_status()

        self.assertEqual(status["models"], models)
        self.assertEqual(status["preferred_model"], SUGOI_14B)
        self.assertEqual([engine["id"] for engine in engines], ["ollama"])
        self.assertEqual(engines[0]["label"], "Ollama")
        self.assertEqual(engines[0]["models"], models)

        with patch.object(translation_engine, "OLLAMA_TEXT_MODEL", "missing-model"), \
                patch.object(translation_engine, "list_ollama_models", return_value=models):
            self.assertEqual(translation_engine.preferred_ollama_text_model(), "other-local-model:1b")


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as backend_app

        cls.app_module = backend_app
        cls.route_keys = {
            (method, route.path)
            for route in backend_app.app.routes
            for method in getattr(route, "methods", set())
        }

    def test_legacy_endpoints_are_removed(self):
        checks = [
            ("POST", "/api/scanner/{filename}/scan-translate"),
            ("GET", "/api/scanner/ocr-engines"),
            ("POST", "/api/scanner/translate"),
            ("GET", "/api/rabbithole/panels"),
            ("GET", "/api/rabbithole/progress"),
            ("GET", "/api/rabbithole/{filename}/vocab"),
            ("POST", "/api/rabbithole/{filename}/answer"),
        ]
        for method, path in checks:
            with self.subTest(path=path):
                self.assertNotIn((method, path), self.route_keys)

    def test_media_rejects_path_traversal(self):
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(self.app_module.get_ocr_debug_media("../config.py"))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_upload_rejects_non_images(self):
        result = ImageService.save_uploaded_panel(b"not an image", "bad.png")
        self.assertFalse(result["success"])


class RuntimeHealthTests(unittest.TestCase):
    def test_runtime_routes_are_present(self):
        import app as backend_app

        route_keys = {
            (method, route.path)
            for route in backend_app.app.routes
            for method in getattr(route, "methods", set())
        }
        self.assertIn(("GET", "/api/runtime/status"), route_keys)
        self.assertIn(("POST", "/api/runtime/ocr-assets/download"), route_keys)

    def test_startup_checks_status_without_downloading(self):
        import app as backend_app

        fake_status = {"ocr": {"ready": False}, "ollama": {"available": False}, "warnings": ["missing"]}

        async def run_lifespan():
            async with backend_app.lifespan(backend_app.app):
                return True

        with patch.object(backend_app, "check_runtime_status", return_value=fake_status) as check_status, \
                patch.object(bootstrap, "ensure_ocr_assets") as ensure_ocr_assets:
            self.assertTrue(asyncio.run(run_lifespan()))

        check_status.assert_called_once()
        ensure_ocr_assets.assert_not_called()

    def test_missing_mangaocr_package_status_does_not_crash(self):
        with patch.object(bootstrap, "_manga_ocr_package_available", return_value=False), \
                patch.object(bootstrap, "_detector_status", return_value={"available": True, "status": "ready"}), \
                patch.object(bootstrap, "_ollama_status", return_value={"available": False, "status": "missing"}):
            status = bootstrap.check_runtime_status()

        self.assertFalse(status["ocr"]["ready"])
        self.assertEqual(status["ocr"]["package"]["status"], "missing")
        self.assertEqual(status["ocr"]["mangaocr_cache"]["status"], "blocked")

    def test_missing_mangaocr_cache_status_does_not_crash(self):
        with patch.object(bootstrap, "_manga_ocr_package_available", return_value=True), \
                patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("cache missing")), \
                patch.object(bootstrap, "_detector_status", return_value={"available": True, "status": "ready"}), \
                patch.object(bootstrap, "_ollama_status", return_value={"available": True, "status": "ready"}):
            status = bootstrap.check_runtime_status()

        self.assertFalse(status["ocr"]["ready"])
        self.assertEqual(status["ocr"]["mangaocr_cache"]["status"], "missing")
        self.assertIn("cache missing", status["ocr"]["mangaocr_cache"]["error"])

    def test_missing_detector_status_does_not_crash(self):
        missing_model = Path(tempfile.mkdtemp(prefix="missing-detector-")) / "missing.onnx"
        try:
            with patch.object(bootstrap, "TEXT_REGION_MODEL_PATH", missing_model), \
                    patch.object(bootstrap, "_manga_ocr_package_available", return_value=True), \
                    patch.object(bootstrap, "_manga_ocr_cache_status", return_value={"available": True, "status": "ready"}), \
                    patch.object(bootstrap, "_ollama_status", return_value={"available": True, "status": "ready"}):
                status = bootstrap.check_runtime_status()

            self.assertFalse(status["ocr"]["ready"])
            self.assertEqual(status["ocr"]["detector"]["status"], "missing")
        finally:
            shutil.rmtree(missing_model.parent, ignore_errors=True)

    def test_download_endpoint_uses_ocr_asset_setup_only(self):
        fake_status = {"success": True, "ocr": {"ready": True}, "setup": {"success": True}}
        with patch.object(runtime_routes, "ensure_ocr_assets", return_value=fake_status) as ensure_ocr_assets:
            result = asyncio.run(runtime_routes.download_ocr_assets())

        self.assertEqual(result, fake_status)
        ensure_ocr_assets.assert_called_once()

    def test_ocr_endpoint_returns_clear_503_when_unavailable(self):
        panel = _fake_panel()
        try:
            with patch.object(scanner.ImageService, "get_panel_by_filename", return_value=panel), \
                    patch.object(scanner, "_run_ocr", return_value={"success": False, "error": "MangaOCR model cache is missing"}):
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(scanner.scan_panel("panel.png", None))

            self.assertEqual(ctx.exception.status_code, 503)
            self.assertIn("download OCR assets", ctx.exception.detail)
        finally:
            _cleanup_panel(panel)

    def test_ollama_unavailable_is_reflected_in_runtime_status(self):
        with patch.object(translation_engine.requests, "get", side_effect=RuntimeError("offline")), \
                patch.object(bootstrap, "_manga_ocr_package_available", return_value=True), \
                patch.object(bootstrap, "_manga_ocr_cache_status", return_value={"available": True, "status": "ready"}), \
                patch.object(bootstrap, "_detector_status", return_value={"available": True, "status": "ready"}):
            status = bootstrap.check_runtime_status()

        self.assertFalse(status["ollama"]["reachable"])
        self.assertFalse(status["ollama"]["available"])

    def test_ollama_ready_accepts_any_installed_model(self):
        with patch.object(translation_engine, "ollama_model_discovery_status", return_value={
            "models": ["other-local-model:1b"],
            "preferred_model": "other-local-model:1b",
            "discovery_available": True,
            "discovery_error": None,
        }), patch.object(bootstrap, "OLLAMA_TEXT_MODEL", SUGOI_14B):
            status = bootstrap._ollama_status()

        self.assertTrue(status["available"])
        self.assertTrue(status["reachable"])
        self.assertFalse(status["model_installed"])
        self.assertEqual(status["models"], ["other-local-model:1b"])

    def test_translation_engines_are_ollama_only_and_expose_all_models(self):
        with patch.object(translation_engine, "list_ollama_models", return_value=["other-local-model:1b", SUGOI_14B]):
            engines = translation_engine.engine_status()

        self.assertEqual([engine["id"] for engine in engines], ["ollama"])
        self.assertEqual(engines[0]["label"], "Ollama")
        self.assertEqual(engines[0]["models"], ["other-local-model:1b", SUGOI_14B])

    def test_request_logging_filters_noisy_successes(self):
        self.assertFalse(logging_config._is_important_success("GET", "/api/media/panel/best.jpg"))
        self.assertFalse(logging_config._is_important_success("GET", "/api/scanner/best.jpg/regions"))
        self.assertFalse(logging_config._is_important_success("GET", "/api/scanner/best.jpg/cache-status"))
        self.assertTrue(logging_config._is_important_success("POST", "/api/scanner/best.jpg/ocr"))
        self.assertTrue(logging_config._is_important_success("POST", "/api/runtime/ocr-assets/download"))


if __name__ == "__main__":
    unittest.main()
