# Manga Language Learner Roadmap And Progress

This is the single canonical roadmap/progress file. The scanner remains the core product: detect manga text, OCR it transparently, translate it with a selectable engine, render the translated panel, and expose learning metadata directly in the reading UI.

## Current Architecture

```text
panel image
-> text detection
-> panel overrides
-> OCR candidates
-> OCR candidate scoring
-> explicit translation engine
-> tokenization/readings
-> scanner annotations
-> translated panel renderer
-> hover/debug/learning UI
```

## Active Goals

- [x] Native translated panel renderer with JP/EN slider.
- [x] OCR debug panel with candidate previews.
- [x] Explicit OCR engine metadata in responses.
- [x] Explicit translation engine abstraction with Ollama as default.
- [x] Scan options dropdown for cache, OCR, scoring, and translation controls.
- [x] Cache indicator and cache-clearing action.
- [x] Central lookup cache scaffold under `backend/data/lookup_cache/`.
- [x] Token/readings metadata scaffold with SudachiPy and romaji conversion.
- [x] Click-to-pin OCR hover cards and orientation recompute endpoint.
- [x] Full drag/resize UI for mutable OCR boxes.
- [ ] Rich kanji/word learning inspector inspired by kai.kanjiapi.dev.
- [ ] Optional embedding/LLM semantic reranker for close OCR candidates.
- [ ] Furigana alignment and reading correction.

## Implementation Notes

- OCR services are retained and selectable. MangaOCR is the default; Ollama Vision and Gemini Vision can be exposed when available.
- Translation is separate from OCR. Ollama text translation is the default; Gemini translation can be selectable if configured. Google/MyMemory translation should not be used silently.
- Per-panel state belongs under `backend/data/ocr_state/` and stores detection, overrides, derived OCR/translation/render outputs, and panel-specific token spans.
- Lookup facts are global, not per-panel. Kanji, word, and reading lookups are cached centrally under `backend/data/lookup_cache/`.
- Manual overrides survive cache deletion. Cache deletion means “clear derived OCR/translation/render outputs for this panel.”

## Next Milestones

1. Finish mutable boxes: add, remove, drag, resize, keyboard nudging, and visual edit mode.
2. Replace placeholder kanji lookup with copied/reused kanjiapi.dev data loading suitable for this non-commercial university project.
3. Build the learning inspector: word spans, kanji spans, readings, meanings, and rabbit-hole navigation.
4. Calibrate OCR scoring with real examples, especially rotated vertical text failures.
5. Add optional semantic reranking only for suspicious or close OCR candidates.

## Acceptance Tests

- MangaOCR remains default and no OCR fallback is hidden.
- The selected OCR/translation engines are visible in response/debug UI.
- User-selected Ollama translation model is sent to the backend.
- Missing models produce a clear error instead of falling back silently.
- Trash clears derived panel cache but preserves manual overrides.
- Correct unrotated OCR beats rotated hallucinations when the score margin is close.
- Kana and romaji readings appear in hover/debug/learning views.
- Word and kanji spans can be looked up through central cache endpoints.
