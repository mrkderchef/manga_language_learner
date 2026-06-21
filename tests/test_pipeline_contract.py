from __future__ import annotations

import shutil
import sys
import tempfile
import time
import unittest
import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
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
from services import model_assets  # noqa: E402
from services import logging_config  # noqa: E402
from services.recognition import mangaocr as mangaocr_service  # noqa: E402
from services.rabbithole import nlp as rabbithole_nlp  # noqa: E402
from services.translation import engine as translation_engine  # noqa: E402
from services.rendering import panel_renderer  # noqa: E402
from services.vision.bubble_allocator import associate_model_bubbles, estimate_allocation_space  # noqa: E402
from services.vision.bubble_segmenter import BubblePrediction, _refine_mask  # noqa: E402

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

    def test_rabbithole_endpoint_queues_background_job_and_poll_returns_result(self):
        panel = _fake_panel()
        try:
            with scanner._RABBITHOLE_JOBS_LOCK:
                scanner._RABBITHOLE_JOBS.clear()

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
            with patch.object(scanner.ImageService, "get_panel_by_filename", return_value=panel), \
                    patch.object(scanner.rabbithole_service, "build_panel_rabbithole", return_value=rabbit_payload):
                queued = asyncio.run(scanner.build_rabbithole("panel.png", scanner.ScanOptions(use_cache=False)))
                self.assertTrue(queued["rabbithole_job"])
                self.assertIn(queued["status"], {"queued", "running"})

                status = queued
                for _ in range(100):
                    status = asyncio.run(scanner.get_rabbithole_job("panel.png", queued["job_id"]))
                    if status["status"] == "done":
                        break
                    time.sleep(0.02)

            self.assertEqual(status["status"], "done")
            result = status["result"]
            self.assertTrue(result["success"])
            self.assertEqual(result["annotations"][0]["rabbithole"]["reading_hiragana"], "ねこ")
        finally:
            with scanner._RABBITHOLE_JOBS_LOCK:
                scanner._RABBITHOLE_JOBS.clear()
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
            self.assertEqual(
                result["translations_by_region"][scanner._region_id(_fake_region())],
                "cat",
            )
            call_kwargs = translate_batch.call_args.kwargs
            self.assertEqual(call_kwargs["context_units"][0]["region_id"], scanner._region_id(_fake_region()))
            self.assertEqual(call_kwargs["context_units"][0]["orientation"], "vertical")
            rabbithole.assert_not_called()
        finally:
            _cleanup_panel(panel)

    def test_translation_merge_uses_region_ids_instead_of_array_position(self):
        base = {
            "success": True,
            "annotations": [
                {"region_id": "right", "text": "愛してくれて"},
                {"region_id": "left", "text": "ありがとう"},
            ],
        }
        translation = {
            "success": True,
            "translations": ["wrong left", "wrong right"],
            "translations_by_region": {
                "left": "Thank you!",
                "right": "Thank you for loving me",
            },
        }

        result = scanner._merge_translation_result(base, translation, {})

        self.assertEqual(result["annotations"][0]["translated"], "Thank you for loving me")
        self.assertEqual(result["annotations"][1]["translated"], "Thank you!")

    def test_rendered_image_url_is_versioned_by_render_content(self):
        first = panel_renderer._rendered_image_url("panel-id", "panel_aaa111.png")
        second = panel_renderer._rendered_image_url("panel-id", "panel_bbb222.png")

        self.assertEqual(first, "/api/media/rendered/panel-id/current.png?v=aaa111")
        self.assertNotEqual(first, second)

    def test_cached_stage_hydration_joins_annotations_by_region_id(self):
        source = (ROOT / "frontend" / "js" / "scanner.js").read_text(encoding="utf-8")
        hydration = source.split("function mergeHydratedScanResult", 1)[1].split(
            "function annotationHydrationSignature", 1
        )[0]

        self.assertIn("stageAnnotationsByRegion", hydration)
        self.assertIn("getAnnotationRegionId(ann, index)", hydration)
        self.assertNotIn("stageAnnotations[index]", hydration)

    def test_translation_storage_migrates_to_and_keeps_only_current_result(self):
        panel = _fake_panel()
        try:
            cache_dir = scanner._stage_cache_dir(panel, "translation")
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "older.json").write_text(json.dumps({
                "cache_key": "older",
                "saved_at": 10,
                "result": {"translations": ["old"], "translation_model": "old-model"},
            }), encoding="utf-8")
            (cache_dir / "newer.json").write_text(json.dumps({
                "cache_key": "newer",
                "saved_at": 20,
                "result": {"translations": ["new"], "translation_model": "new-model"},
            }), encoding="utf-8")

            migrated = scanner._load_persisted_stage(panel, "translation", "ignored-model-key")

            self.assertEqual(migrated["translations"], ["new"])
            self.assertTrue(scanner._translation_current_path(panel).exists())
            self.assertFalse(cache_dir.exists())
            self.assertEqual(scanner._panel_cache_status({}, panel)["buckets"]["translation"]["entries"], 1)

            scanner._save_persisted_stage(
                panel,
                "translation",
                "replacement",
                {"translations": ["replacement"], "translation_model": "replacement-model"},
            )
            current = scanner._load_persisted_stage(panel, "translation", "any-key")
            self.assertEqual(current["translations"], ["replacement"])
            self.assertEqual(
                [path.name for path in scanner._stage_root_dir(panel, "translation").glob("*.json")],
                ["current.json"],
            )
        finally:
            _cleanup_panel(panel)

    def test_normal_translate_never_reuses_the_current_translation(self):
        panel = _fake_panel()
        try:
            scanner._save_persisted_stage(panel, "translation", "old", {"translations": ["old"]})
            with patch.object(scanner, "_load_persisted_stage") as load_current, \
                    patch.object(scanner.translation_engine, "preferred_ollama_text_model", return_value=SUGOI_14B), \
                    patch.object(scanner.translation_engine, "translate_batch", return_value={
                        "success": True,
                        "translations": ["new"],
                        "translation_engine_used": "ollama",
                    }) as translate_batch:
                result, _key, should_save = scanner._run_translation_stage(
                    _fake_ocr_result(),
                    {},
                    {"panel_path": panel},
                    [],
                    "test",
                )

            load_current.assert_not_called()
            translate_batch.assert_called_once()
            self.assertEqual(result["translations"], ["new"])
            self.assertTrue(should_save)
        finally:
            _cleanup_panel(panel)

    def test_frontend_has_one_current_translation_without_model_cache_sync(self):
        scanner_source = (ROOT / "frontend" / "js" / "scanner.js").read_text(encoding="utf-8")
        api_source = (ROOT / "frontend" / "js" / "api.js").read_text(encoding="utf-8")

        self.assertIn("getCurrentTranslation", scanner_source)
        self.assertIn("getCurrentTranslation", api_source)
        self.assertNotIn("getCachedTranslation", scanner_source + api_source)
        self.assertNotIn("syncTranslatedPanelForCurrentSelection", scanner_source)

    def test_rabbithole_help_popover_uses_unclipped_viewport_overlay(self):
        source = (ROOT / "frontend" / "js" / "scanner.js").read_text(encoding="utf-8")
        css = (ROOT / "frontend" / "css" / "scanner.css").read_text(encoding="utf-8")
        help_code = source.split("function ensureHelpPopoverOverlay", 1)[1].split(
            "function getSourceRecord", 1
        )[0]
        help_css = css.split(".rabbithole-help-content {", 1)[1].split("}", 1)[0]

        self.assertIn("document.body.appendChild(overlay)", help_code)
        self.assertIn("positionHelpPopover", help_code)
        self.assertIn("position: fixed", help_css)
        self.assertIn("z-index: 10000", help_css)


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


class HybridBubbleTests(unittest.TestCase):
    def test_model_mask_can_be_shared_by_multiple_text_regions(self):
        gray = np.full((100, 100), 245, dtype=np.uint8)
        mask = np.zeros_like(gray)
        mask[10:90, 10:90] = 255
        entries = [
            {"region": {"x": 25, "y": 25, "width": 12, "height": 18}, "ocr_meta": {}},
            {"region": {"x": 58, "y": 52, "width": 13, "height": 16}, "ocr_meta": {}},
        ]
        prediction = BubblePrediction("bubble_001", 0.93, (10, 10, 90, 90), mask)

        matched = associate_model_bubbles(gray, entries, [prediction], {"model_min_containment": 0.6})

        self.assertEqual(matched, {0, 1})
        self.assertEqual(entries[0]["ocr_meta"]["vision"]["bubble_id"], "bubble_001")
        self.assertEqual(entries[1]["ocr_meta"]["vision"]["bubble_id"], "bubble_001")
        self.assertEqual(entries[0]["ocr_meta"]["vision"]["source"], "model_instance")
        self.assertGreaterEqual(entries[0]["ocr_meta"]["vision"]["text_containment"], 0.6)

    def test_model_mask_refinement_fills_holes_and_drops_detached_noise(self):
        mask = np.zeros((80, 80), dtype=np.float32)
        mask[10:65, 10:65] = 1.0
        mask[30:40, 30:40] = 0.0
        mask[72:76, 72:76] = 1.0

        refined = _refine_mask(mask)

        self.assertEqual(int(refined[35, 35]), 255)
        self.assertEqual(int(refined[73, 73]), 0)

    def test_unenclosed_text_uses_compact_allocation_not_global_brightness(self):
        gray = np.full((100, 100), 255, dtype=np.uint8)
        region = {"x": 43, "y": 43, "width": 14, "height": 14, "lines": []}

        allocation = estimate_allocation_space(gray, region, {"wand_enabled": False})

        self.assertEqual(allocation.debug["source"], "compact_text_seed")
        self.assertIn(allocation.debug["fallback_reason"], {"no_enclosing_topology", "disabled_by_settings"})
        self.assertLess(allocation.debug["bubble_box"][2] * allocation.debug["bubble_box"][3], 100 * 100)

    def test_model_association_rejects_text_outside_mask(self):
        gray = np.full((80, 80), 245, dtype=np.uint8)
        mask = np.zeros_like(gray)
        mask[5:25, 5:25] = 255
        entry = {"region": {"x": 50, "y": 50, "width": 12, "height": 12}, "ocr_meta": {}}
        prediction = BubblePrediction("bubble_001", 0.99, (5, 5, 25, 25), mask)

        matched = associate_model_bubbles(gray, [entry], [prediction], {"model_min_containment": 0.6})

        self.assertEqual(matched, set())
        self.assertNotIn("vision", entry["ocr_meta"])


class RabbitholeProvenanceTests(unittest.TestCase):
    def test_kanji_readings_have_corrected_romaji_and_field_provenance(self):
        fixture = {
            "kanji": "夢",
            "meanings": ["dream", "illusion", "vision"],
            "kun_readings": ["くら.い", "ゆめ", "ゆめ.みる"],
            "on_readings": ["ボウ", "ム"],
            "name_readings": ["くら"],
            "stroke_count": 13,
            "grade": 5,
            "jlpt": 3,
            "unicode": "5922",
        }

        with patch.object(rabbithole_nlp, "_read_cache", return_value=None), \
                patch.object(rabbithole_nlp, "_fetch_kanjiapi", return_value=fixture), \
                patch.object(rabbithole_nlp, "kana_to_romanji", side_effect=lambda value: f"r:{value}"), \
                patch.object(rabbithole_nlp, "_get_kakasi", return_value=object()), \
                patch.object(rabbithole_nlp, "_write_cache", side_effect=lambda _kind, _key, data: data):
            result = rabbithole_nlp.lookup_kanji("夢")

        for group in result["structured_readings"].values():
            self.assertTrue(all(reading["romaji"] for reading in group))
        self.assertEqual(result["structured_readings"]["kun"][0]["kana"], "くら.い")
        self.assertEqual(result["field_sources"]["structured_readings"], ["kanjidic2", "pykakasi"])
        self.assertEqual(result["sources"]["kanjidic2"]["dataset"], "KANJIDIC2")

    def test_panel_kanji_analysis_does_not_fetch_rich_remote_sources(self):
        tokenized = {
            "reading_hiragana": "ゆめ",
            "reading_romaji": "yume",
            "reading_romanji": "yume",
            "tokens": [{
                "surface": "夢", "lemma": "夢", "reading_hiragana": "ゆめ",
                "reading_romaji": "yume", "reading_romanji": "yume",
                "pos": ["名詞"], "start": 0, "end": 1, "kanji": ["夢"],
            }],
            "kanji_spans": [{"character": "夢", "start": 0, "end": 1}],
        }
        kanji = rabbithole_nlp._normalize_kanji_payload("夢", {
            "kanji": "夢", "meanings": ["dream"], "kun_readings": ["ゆめ"],
        }, source="fixture")
        word = {
            "entries": [{
                "source": "jmdict", "score": 1000,
                "variants": [{"written": "夢", "reading_hiragana": "ゆめ", "priorities": ["news1"]}],
                "priority_tags": ["news1"], "priority_labels": ["Mainichi newspaper priority, first 12,000 words"],
                "senses": [{"glosses": ["dream"]}], "glosses": ["dream"],
            }],
            "candidate_count": 1, "source": "jmdict", "sources": {},
        }
        with patch.object(rabbithole_nlp, "tokenize_text", return_value=tokenized), \
                patch.object(rabbithole_nlp, "lookup_word", return_value=word), \
                patch.object(rabbithole_nlp, "lookup_kanji", return_value=kanji), \
                patch.object(rabbithole_nlp, "lookup_reading", return_value={}), \
                patch.object(rabbithole_nlp, "_has_cache", return_value=True), \
                patch.object(rabbithole_nlp, "_kanjivg_lookup") as kanjivg, \
                patch.object(rabbithole_nlp, "_wiktionary_glyph_origin") as wiktionary:
            result = rabbithole_nlp.build_panel_rabbithole([{"region_id": "r1", "text": "夢"}])

        kanjivg.assert_not_called()
        wiktionary.assert_not_called()
        self.assertIn("source_catalog", result)
        region = result["by_region"]["r1"]
        self.assertEqual(region["reading_romaji"], "yume")
        self.assertNotIn("morphemes", region["breakdowns"])
        self.assertNotIn("morpheme_segments", region)
        self.assertNotIn("morpheme_count", region["summary"])

    def test_stale_wiktionary_cache_survives_network_failure(self):
        stale = {
            "available": True,
            "text": "Ideographic description from a cached revision.",
            "revision_id": 123,
            "revision_url": "https://en.wiktionary.org/w/index.php?oldid=123",
            "retrieved_at": 1,
        }
        with patch.object(rabbithole_nlp, "_read_cache", return_value=stale), \
                patch.object(rabbithole_nlp.requests, "get", side_effect=RuntimeError("offline")):
            result = rabbithole_nlp._wiktionary_glyph_origin("夢")

        self.assertTrue(result["available"])
        self.assertTrue(result["stale"])
        self.assertEqual(result["revision_id"], 123)

    def test_kanjivg_returns_only_sanitized_structured_paths(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" xmlns:kvg="http://kanjivg.tagaini.net"
            viewBox="0 0 109 109"><script>alert(1)</script>
            <g kvg:element="夢"><g kvg:element="夕" kvg:position="bottom">
            <path id="s1" d="M10 10 L20 20" onclick="alert(2)"/></g></g></svg>'''.encode("utf-8")

        class Response:
            status_code = 200
            content = svg

            @staticmethod
            def raise_for_status():
                return None

        with patch.object(rabbithole_nlp, "_read_cache", return_value=None), \
                patch.object(rabbithole_nlp, "_write_cache", side_effect=lambda _kind, _key, data: data), \
                patch.object(rabbithole_nlp.requests, "get", return_value=Response()):
            result = rabbithole_nlp._kanjivg_lookup("夢")

        self.assertEqual(result["paths"], [{"id": "s1", "d": "M10 10 L20 20"}])
        self.assertEqual(result["components"], [{"element": "夕", "position": "bottom"}])
        self.assertNotIn("svg", result)
        self.assertNotIn("onclick", str(result))

    def test_kanji_inspector_never_renders_primary_meaning_above_dictionary(self):
        source = (ROOT / "frontend" / "js" / "scanner.js").read_text(encoding="utf-8")
        kanji_renderer = source.split("function createKanjiInspectorContent", 1)[1].split(
            "function createGeneralInspectorContent", 1
        )[0]
        self.assertNotIn("primary_meaning", kanji_renderer)
        self.assertEqual(kanji_renderer.count("details.meanings"), 1)


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

    def test_rabbithole_job_polling_route_is_present(self):
        self.assertIn(("GET", "/api/scanner/{filename}/rabbithole/jobs/{job_id}"), self.route_keys)

    def test_learning_page_endpoints_are_present(self):
        checks = [
            ("GET", "/api/learning/panels"),
            ("GET", "/api/learning/{filename}/vocab"),
            ("POST", "/api/learning/{filename}/answer"),
            ("GET", "/api/learning/progress"),
        ]
        for method, path in checks:
            with self.subTest(path=path):
                self.assertIn((method, path), self.route_keys)

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
        self.assertIn(("POST", "/api/runtime/bubble-assets/download"), route_keys)

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
        self.assertEqual(status["ocr"]["mangaocr_model"]["status"], "blocked")

    def test_missing_mangaocr_model_status_does_not_crash(self):
        with patch.object(bootstrap, "_manga_ocr_package_available", return_value=True), \
                patch.object(bootstrap, "missing_manga_ocr_files", return_value=["config.json"]), \
                patch.object(bootstrap, "_detector_status", return_value={"available": True, "status": "ready"}), \
                patch.object(bootstrap, "_ollama_status", return_value={"available": True, "status": "ready"}):
            status = bootstrap.check_runtime_status()

        self.assertFalse(status["ocr"]["ready"])
        self.assertEqual(status["ocr"]["mangaocr_model"]["status"], "missing")
        self.assertIn("config.json", status["ocr"]["mangaocr_model"]["error"])

    def test_mangaocr_download_targets_backend_model_directory(self):
        model_dir = Path(tempfile.mkdtemp(prefix="mangaocr-model-"))
        try:
            with patch.object(
                model_assets,
                "missing_manga_ocr_files",
                side_effect=[["config.json"], []],
            ), patch("huggingface_hub.snapshot_download", return_value=str(model_dir)) as download:
                result = model_assets.download_manga_ocr_model(model_dir)

            self.assertEqual(result, model_dir)
            download.assert_called_once_with(
                repo_id=model_assets.MANGA_OCR_REPO_ID,
                revision=model_assets.MANGA_OCR_REVISION,
                local_dir=str(model_dir),
                allow_patterns=list(model_assets.MANGA_OCR_REQUIRED_FILES),
            )
        finally:
            shutil.rmtree(model_dir, ignore_errors=True)

    def test_mangaocr_loader_never_downloads_missing_assets(self):
        previous = mangaocr_service._mocr
        mangaocr_service._mocr = None
        try:
            with patch.object(mangaocr_service, "missing_manga_ocr_files", return_value=["pytorch_model.bin"]), \
                    patch("huggingface_hub.snapshot_download") as download:
                with self.assertRaisesRegex(RuntimeError, "Runtime settings"):
                    mangaocr_service._get_mocr()
            download.assert_not_called()
        finally:
            mangaocr_service._mocr = previous

    def test_missing_detector_status_does_not_crash(self):
        missing_model = Path(tempfile.mkdtemp(prefix="missing-detector-")) / "missing.onnx"
        try:
            with patch.object(bootstrap, "TEXT_REGION_MODEL_PATH", missing_model), \
                    patch.object(bootstrap, "_manga_ocr_package_available", return_value=True), \
                    patch.object(bootstrap, "_manga_ocr_model_status", return_value={"available": True, "status": "ready"}), \
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

    def test_bubble_download_endpoint_uses_pinned_asset_setup(self):
        fake_status = {"success": True, "bubble_segmentation": {"available": True}}
        with patch.object(runtime_routes, "ensure_bubble_assets", return_value=fake_status) as ensure_bubble_assets:
            result = asyncio.run(runtime_routes.download_bubble_assets())

        self.assertEqual(result, fake_status)
        ensure_bubble_assets.assert_called_once()

    def test_bubble_checkpoint_manifest_is_revision_and_checksum_pinned(self):
        self.assertEqual(model_assets.BUBBLE_MODEL_REVISION, "3a860269ee0beb43ce9f31d82c7851441eb178ae")
        self.assertEqual(model_assets.BUBBLE_MODEL_SHA256, "0b4376e426fa96af3976afa6a2602421dacf2dec96ef87b4a44f5e8d4971cb6f")

    def test_ocr_endpoint_returns_clear_503_when_unavailable(self):
        panel = _fake_panel()
        try:
            with patch.object(scanner.ImageService, "get_panel_by_filename", return_value=panel), \
                    patch.object(scanner, "_run_ocr", return_value={"success": False, "error": "MangaOCR model is missing"}):
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(scanner.scan_panel("panel.png", None))

            self.assertEqual(ctx.exception.status_code, 503)
            self.assertIn("download OCR assets", ctx.exception.detail)
        finally:
            _cleanup_panel(panel)

    def test_ollama_unavailable_is_reflected_in_runtime_status(self):
        with patch.object(translation_engine.requests, "get", side_effect=RuntimeError("offline")), \
                patch.object(bootstrap, "_manga_ocr_package_available", return_value=True), \
                patch.object(bootstrap, "_manga_ocr_model_status", return_value={"available": True, "status": "ready"}), \
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
