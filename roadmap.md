# Manga Language Learner Roadmap

This roadmap turns the current wish-list into an implementation path that fits the existing architecture. The main idea is simple:

The scanner should stay the core product. New learning features should attach to the scanner output as structured metadata and UI layers, not replace the OCR/translation pipeline.

## Current Product Shape

The app currently has two visible modes:

- Scanner mode: select/upload a manga panel, detect text boxes, OCR Japanese text, translate it, and show hover overlays.
- Learning mode: a separate early mode that turns OCR annotations into simple vocabulary cards.

The current scanner pipeline is:

```text
image
-> text region detection
-> crop each region
-> OCR preprocessing
-> MangaOCR
-> OCR cleanup
-> reading order sort
-> batch translation
-> scanner annotations
-> frontend overlay boxes
```

The key response object today is roughly:

```json
{
  "text": "full recognized text",
  "annotations": [
    {
      "text": "それがおれの",
      "translated": "That's my",
      "bbox": [[1055, 48], [1234, 48], [1234, 283], [1055, 283]],
      "confidence": 0.95,
      "vertical": true,
      "ocr_variant": "preprocessed"
    }
  ],
  "image_width": 1304,
  "image_height": 1377
}
```

The roadmap should extend this object, not throw it away.

## Product Direction

The tool should become an interactive manga reading tutor:

- The panel remains visible as the main object.
- The user can compare the original and translated view.
- Hovering or clicking a text region opens a richer reading card.
- Japanese text inside that card is token-aware.
- Clicking a kanji, word, or phrase reveals learner information.
- External data sources enrich the experience only when needed.
- The separate learning mode can later reuse scanner metadata.

## Architecture Principles

1. Preserve the current scanner flow.

The existing detection, OCR, translation, and overlay flow is already valuable. Improvements should be additive unless a part is clearly broken.

2. Prepare cheap deterministic data during scan.

During `Scan & Translate`, it is reasonable to compute:

- OCR text
- translations
- bounding boxes
- reading order
- OCR/debug metadata
- Japanese tokenization
- candidate token spans

It is not reasonable to call multiple external APIs for every kanji during scan.

3. Fetch enrichment lazily.

Kanji details, stroke order, etymology, examples, and audio should load when the user clicks/selects something. This keeps scanning fast and avoids wasting API calls.

4. Cache every lookup.

Kanji and word lookups should use a small backend cache. The frontend may also keep a short-lived in-memory cache for snappy repeated clicks.

5. Keep optional sources optional.

The scanner should work even if Kanji Alive, KanjiPortraits, or an embedding model is unavailable.

6. Treat Japanese text as controlled interactive content.

Selection and click behavior should only trigger inside known Japanese text containers rendered by the app. Random UI selection should not trigger learning lookups.

## Target Architecture

```text
Backend

image
-> region detector
-> OCR crops
-> MangaOCR
-> OCR cleanup
-> reading order
-> batch translation
-> Japanese tokenizer
-> annotation metadata
-> JSON response

Frontend

scanner response
-> original image layer
-> OCR region overlay
-> optional translated overlay layer
-> hover/inspect card
-> clickable Japanese token renderer
-> lazy lookup calls

Lookup services

selected kanji/word/phrase
-> backend lookup endpoint
-> cache
-> kanjiapi.dev
-> optional Kanji Alive
-> optional KanjiPortraits
-> optional phrase/context model
-> normalized learning object
```

## Target Annotation Contract

Keep all existing fields and add structured fields gradually.

```json
{
  "id": "ann_0001",
  "text": "私は毎日学校へ歩いて行きます。",
  "translated": "I walk to school every day.",
  "bbox": [[0, 0], [100, 0], [100, 200], [0, 200]],
  "confidence": 0.95,
  "vertical": true,
  "ocr_variant": "preprocessed_rot90_ccw",
  "reading_order": 1,
  "tokens": {
    "short": [
      {
        "id": "ann_0001_s_000",
        "surface": "私",
        "lemma": "私",
        "reading": "ワタシ",
        "pos": ["noun"],
        "start": 0,
        "end": 1,
        "kanji": ["私"]
      }
    ],
    "middle": [],
    "long": []
  },
  "phrase_candidates": [
    {
      "id": "ann_0001_p_000",
      "text": "学校へ",
      "start": 4,
      "end": 7,
      "token_ids": ["ann_0001_s_003", "ann_0001_s_004"],
      "source": "token_rules"
    }
  ]
}
```

This lets the frontend render Japanese text safely and lets later features reuse the same data.

## Phase 1: Better Hoverbox / Inspector

Goal: make the existing hover interaction useful and pleasant before adding complex language data.

Current state:

- `.ocr-box` has a small CSS-only tooltip.
- Tooltip shows only Japanese OCR text and English translation.
- Longer text does not have much room.
- The tooltip cannot easily support loading states or interactive token clicks.

Implementation:

- Replace the tiny tooltip with a richer hover/inspect card.
- Keep hover behavior for quick reading.
- Add click-to-pin behavior so the user can interact with Japanese text.
- Show Japanese text, English translation, OCR confidence/debug hints, and room for learning sections.
- Ensure the card handles long text with wrapping and max-height scrolling.
- Keep overlay boxes visible on the image.

Suggested frontend pieces:

- `frontend/js/scanner.js`
  - Store latest scan annotations.
  - Render box hover state.
  - Render an inspector card for active annotation.
- `frontend/css/scanner.css`
  - Larger card layout.
  - Active/pinned box state.
  - Loading and empty states.

Why this comes first:

It gives us a stable UI container for all later features.

## Phase 2: Tokenize Japanese Text During Scan

Goal: attach token metadata to every OCR annotation.

Recommended tokenizer:

- SudachiPy first.
- Use MeCab only if SudachiPy causes installation or quality problems.

Why SudachiPy:

- It supports multiple split modes.
- It is well suited for short, ambiguous Japanese text.
- It gives us a clean path to character, word, and phrase-like units.

Sudachi split modes:

```text
Mode A -> shorter units, good for kanji/word study
Mode B -> medium units, good for compounds
Mode C -> longer units, useful for phrase candidates
```

Implementation:

- Add a backend service such as `backend/services/japanese_text_service.py`.
- Tokenize each annotation text after OCR and before returning the scan response.
- Add token fields to annotations.
- Keep scanner behavior unchanged if tokenization fails.

Dependencies:

- Add `sudachipy`.
- Add a Sudachi dictionary package, likely `sudachidict_core`.

Important:

Tokenization should not block the whole scan if it errors. Return the old annotation shape plus an empty token list and log the issue.

## Phase 3: Controlled Japanese Text Rendering

Goal: make Japanese text in the inspector clickable/selectable without accidental UI triggers.

Implementation:

- Render Japanese text from token metadata rather than as one raw text node.
- Each token gets `data-token-id`, `data-start`, `data-end`, and `data-annotation-id`.
- Clicking a token selects that token.
- Clicking a single kanji inside a token selects that kanji if practical.
- Text selection can be added after click behavior is stable.

Recommended interaction split:

- Click: best for kanji and individual tokens.
- Drag/select: later, best for phrase spans.

First version should favor clicking. It is simpler, more mobile-friendly, and avoids fighting browser selection behavior.

## Phase 4: Lazy Kanji and Word Lookups

Goal: when the user clicks a kanji or token, show real learning information.

First backend endpoints:

```text
GET /api/learning/kanji/{character}
GET /api/learning/word?text=...
```

Possible later endpoint:

```text
POST /api/learning/selection/explain
```

First source:

- `kanjiapi.dev`

Useful kanjiapi data:

- meanings
- on readings
- kun readings
- name readings
- stroke count
- JLPT level
- school grade
- words containing the kanji
- reading lookup

Cache keys:

```text
kanji:猫
words:猫
reading:ネコ
word:学校
```

Cache location:

- Backend file cache under `backend/data/lookup_cache/`.
- Optional frontend memory cache in `API`.

Frontend behavior:

- User clicks kanji/token.
- Inspector shows a loading spinner in the learning section.
- Backend lookup returns normalized data.
- Inspector renders meanings, readings, stroke count, JLPT/grade, and example words.

## Phase 5: Word and Phrase Explanations

Goal: go beyond single kanji and explain words or phrase chunks.

Base approach:

- Use Sudachi tokens and split modes for candidate units.
- Generate phrase candidates from adjacent tokens.
- Use current sentence translation as context.
- Ask the existing local/Ollama translation layer for short explanations when needed.

Example:

```text
私 / は / 毎日 / 学校 / へ / 歩いて / 行きます

学校へ -> to school
歩いて行きます -> go by walking / walk
学校へ歩いて行きます -> walk to school
```

First implementation:

- Rules generate phrase candidates from neighboring tokens.
- The UI shows phrase chips for likely spans.
- Clicking a phrase chip asks the backend for a concise contextual explanation.

Later implementation:

- Add a context model to rank candidate spans.

## Phase 6: Context Model for Ambiguous Segmentation

Goal: improve phrase grouping and context-sensitive meaning.

This is useful, but it should not be the first implementation.

The tokenizer answers:

```text
Where are possible units?
```

The dictionary/API layer answers:

```text
What can this kanji or word mean in isolation?
```

The context model helps answer:

```text
Which neighboring tokens belong together here?
Which reading or meaning is likely in this sentence?
Does this selected span form a meaningful phrase?
```

Possible model choices:

- Japanese Sentence-BERT.
- Japanese BERT with embeddings.
- A multilingual sentence-transformers model with strong Japanese support.
- Existing local LLM/Ollama model for explanation and ranking if latency is acceptable.

Recommended first use:

- Do not use embeddings to tokenize.
- Use embeddings or an LLM only to score/rank phrase candidates produced by SudachiPy and simple rules.

This keeps the architecture clean:

```text
SudachiPy -> candidates
dictionaries/APIs -> factual language data
context model -> ranking and contextual explanation
```

## Phase 7: Generated Translated Panel + Slider

Goal: create a translated panel image that looks like it belongs in the manga, then let the user compare original and translated versions with a JP/EN slider.

Important architectural note:

We keep our existing detection, OCR and translation pipeline. The new feature starts after translations are available:

```text
scan annotations
-> text-removal mask
-> OpenCV cleanup/inpaint
-> fitted English text rendering
-> translated image cache
-> JP/EN image comparison slider
```

The downloaded `manga-image-translator` project is used as inspiration for architecture and quality goals only. We should reimplement a small native renderer in this project rather than copying GPL code into the app.

### Phase 7A: Native Backend Renderer

First version:

- Build a backend renderer that consumes the existing scan annotations.
- Generate a real translated image under `backend/data/rendered_panels/`.
- Return `translated_image_url`, `render_method` and `render_warnings` in the scan response.
- Cache rendered images by panel identity plus annotation/translation content.

Renderer input data:

```text
annotation.bbox
annotation.lines
annotation.text
annotation.translated
annotation.vertical
annotation.font_size
annotation.angle
image_width
image_height
```

Renderer strategy:

1. Build a text mask from `annotation.lines` when available, falling back to `bbox`.
2. Dilate/blur the mask enough to cover glyph strokes.
3. Clean the original text with OpenCV inpainting; use light/white fill as a fallback for speech bubbles.
4. Estimate a writing area by expanding from the OCR box into nearby bubble-like whitespace.
5. Render English horizontally with Pillow, using a manga/comic-friendly font if available.
6. Fit text by wrapping lines, reducing font size, and using a subtle stroke for readability.
7. Paste rendered text onto the cleaned image.

Placement rules:

- Vertical Japanese speech text usually becomes horizontal English text centered in the detected/expanded bubble area.
- Angled text keeps the source angle when it is meaningful and not too noisy.
- Very small or very narrow regions get a minimum writing area.
- Long translations reduce font size first, then wrap more tightly.
- If a translation still cannot fit, render the best readable version and keep the full text available in the hover/inspector UI.

Quality target:

The translated image should be inspectable on its own. It should not depend on semi-transparent UI boxes to look translated.

### Phase 7B: JP/EN Image Slider

Frontend layer structure:

```text
panel wrapper
|-- original image
|-- generated translated image
|-- OCR hover boxes
|-- slider control
|-- JP/EN button
```

Behavior:

- Slider at 0%: original panel.
- Slider at 100%: generated translated panel visible.
- JP button animates slider to 0%.
- EN button animates slider to 100%.
- User can stop anywhere in between.
- The button does not swap images directly. It only moves the slider.
- The slider is the single source of truth for how much translated image is visible.

The OCR hover/inspect overlay should remain aligned above both images so language-learning features still work.

### Phase 7C: Later Quality Upgrades

Possible later upgrades:

- Better speech bubble interior detection.
- Collision-aware text placement across nearby boxes.
- Model-based inpainting such as LaMa/AOT as an optional renderer mode.
- Text color extraction from the original region.
- Separate style profiles for dialogue, narration boxes and sound effects.

These should not block the first generated-image version.

### Acceptance Criteria

- After `Scan & Translate`, the backend produces a translated image URL when annotations have translations.
- The translated image removes or softens Japanese text before drawing English.
- English is fitted into the panel in a readable way without visible UI boxes.
- A slider appears above or near the panel.
- JP moves the slider to original view.
- EN moves the slider to translated view.
- Moving the slider reveals the generated translated image over the original panel.
- The normal OCR hover/inspect behavior still works.

This phase is important because it makes the scanner feel like a manga translation tool, not only a list of OCR results.

## Phase 8: Rich Kanji Sources

Goal: add visual and historical learning information after the basic lookup path works.

Source priority:

1. Local tokenization result.
2. Current annotation translation.
3. `kanjiapi.dev`.
4. Kanji Alive API.
5. KanjiPortraits.
6. Context model / embeddings.

Kanji Alive can add:

- stroke order poster
- stroke animation
- radical image
- radical position
- radical meaning
- example words
- example audio

KanjiPortraits can add:

- etymology links
- related kanji families
- historical images
- short excerpts

KanjiPortraits limitation:

It is a website, not a clean JSON API. Treat it as optional enrichment. Do not dump article HTML into the app.

If used, extract only:

- article title
- article URL
- related kanji from title/tags
- main image URLs
- short excerpt

## Phase 9: Scanner-Aware Learning Mode

Goal: make the learning mode reuse scanner metadata instead of treating whole OCR boxes as naive vocabulary.

Current learning mode:

- Reads OCR annotations.
- Makes one card per detected text block.

Better learning mode:

- Use tokenized scanner results.
- Create cards from kanji, words, and phrases.
- Track progress per normalized item.
- Link every card back to its source panel and text box.

Possible card types:

- Kanji card: character, meaning, readings, stroke count.
- Word card: surface form, reading, meaning, source sentence.
- Phrase card: selected span, contextual translation.
- Sentence card: OCR text, translation, panel context.

Progress keys:

```text
kanji:学
word:学校
phrase:学校へ
sentence:<hash>
```

This should remain separate from scanner mode visually, but it should reuse the scanner data model.

## Phase 10: OCR Debug and Quality Improvements

Goal: improve reliability of the foundation.

Useful debug fields:

- detection box
- crop bounds
- OCR variant
- OCR internal score
- vertical/horizontal guess
- reading order
- translation index
- tokenization status

Possible UI:

- Debug toggle in scanner.
- Show confidence color on boxes.
- Inspect OCR variant and score in the hover card.

Quality improvements:

- Furigana detection/handling.
- Better OCR cleanup before translation.
- Panel-aware processing.
- Better reading order for full pages.

These are not separate from the learning features. Better OCR makes every downstream learning feature better.

## Suggested Implementation Order

1. Create richer hover/inspect card.
2. Add click-to-pin active OCR box behavior.
3. Add backend tokenization with SudachiPy.
4. Render token-aware Japanese text in the inspector.
5. Add kanji lookup endpoint and cache.
6. Show kanji info on click with loading state.
7. Add word lookup and token explanations.
8. Add phrase chips from token spans.
9. Add translated overlay slider.
10. Add Kanji Alive visual enrichments.
11. Add scanner-aware learning cards.
12. Add optional phrase/context model.
13. Add KanjiPortraits etymology layer.

## Near-Term MVP

The best first milestone is:

```text
Click a detected text box
-> inspector opens
-> Japanese text is shown as clickable tokens
-> click a kanji
-> loading spinner
-> meanings/readings/stroke count/JLPT/examples appear
```

This milestone proves the full architecture:

- scanner output can carry metadata
- frontend can render controlled Japanese text
- backend can perform lazy enrichment
- cache works
- the UI feels like a reading tutor

After that, the slider and phrase features will have a much cleaner place to attach.

## Open Decisions

Tokenizer:

- Start with SudachiPy unless installation becomes painful.

Interaction:

- Start with click-to-select.
- Add drag selection later.

Kanji source:

- Start with kanjiapi.dev.
- Add Kanji Alive only after basic lookup UI works.

Translated image:

- Start with frontend overlay slider.
- Treat real translated bitmap generation as a later feature.

Embeddings/model:

- Do not use a model for basic segmentation.
- Use a model later to rank phrase candidates and explain context.

Learning mode:

- Ignore the current naive learning mode until scanner metadata is stable.
- Later rebuild learning mode around scanner annotations, tokens, and lookup cache.

## Non-Goals For The First Pass

- Do not rebuild the OCR pipeline.
- Do not call every external API during scanning.
- Do not scrape large article bodies into hoverboxes.
- Do not implement a full image text replacement engine before the overlay slider.
- Do not make the selection system depend on arbitrary browser text selection.
- Do not make the embedding model responsible for tokenization.

## File/Module Map

Likely backend changes:

- `backend/services/japanese_text_service.py`
- `backend/services/lookup_service.py`
- `backend/routes/learning.py`
- `backend/routes/scanner.py`
- `requirements.txt`

Likely frontend changes:

- `frontend/js/scanner.js`
- `frontend/js/api.js`
- `frontend/css/scanner.css`
- `frontend/index.html`

Likely data directories:

- `backend/data/ocr_cache/`
- `backend/data/lookup_cache/`

Potential docs:

- `roadmap.md`
- `kanji_sources.md`
- cleaned `todo.md`
