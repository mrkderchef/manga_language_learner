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
            with patch.object(scanner.translation_engine, "translate_batch", return_value=translation_payload), \
                    patch.object(scanner.rabbithole_service, "build_panel_rabbithole") as rabbithole, \
                    patch.object(scanner, "render_translated_panel", return_value={"translated_image_url": None, "render_warnings": []}):
                result = scanner._run_translate_existing(panel, {"use_cache": False})

            self.assertTrue(result["success"])
            self.assertEqual(result["annotations"][0]["translated"], "cat")
            rabbithole.assert_not_called()
        finally:
            _cleanup_panel(panel)


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


if __name__ == "__main__":
    unittest.main()
