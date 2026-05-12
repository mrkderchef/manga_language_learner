# Roadmap Progress

Live checklist for implementation. `roadmap.md` remains the architecture plan.

## Active Milestone

- [x] Generated translated panel image + JP/EN slider
  - [x] Backend renders a translated panel image from existing OCR annotations.
  - [x] Japanese text is cleaned with an OpenCV mask/inpaint pass.
  - [x] English text is fitted into the detected/expanded text areas.
  - [x] Scan response includes `translated_image_url`.
  - [x] Frontend slider compares original image with generated translated image.
  - [x] JP/EN button moves the slider to the original/translated endpoints.
  - [x] Slider now sits on the image with a draggable divider handle.
  - [x] Renderer uses a page-level placement pass to reduce translated text collisions.
  - [x] Renderer refines candidate choices with glyph-mask collision checks.
  - [x] Renderer estimates minimum original-text area and maximum usable light/bubble area.

## Scanner And Rendering

- [x] Native translated-panel renderer
- [x] Render cache under `backend/data/rendered_panels/`
- [x] Render warnings surfaced in scan response
- [x] OCR hover overlay remains aligned above the image comparison
- [x] Non-rectangular light-region masks influence translated text placement
- [x] English text is forced horizontal unless the OCR region has a real mild angle
- [x] Boxed missing translations render as a neutral placeholder instead of vanishing
- [ ] Later: model-based inpainting option

## Reading Tutor Features

- [ ] Rich hover/inspect card
- [ ] Backend Japanese tokenization
- [ ] Controlled clickable Japanese text
- [ ] Lazy kanji lookup with cache
- [ ] Word and phrase explanations
- [ ] Optional context model for phrase scoring

## Learning Mode

- [ ] Scanner-aware learning cards
- [ ] Progress keys for kanji, words, phrases, and sentences

## Done

- [x] Consolidated roadmap into `roadmap.md`
- [x] Chose native renderer direction inspired by `manga-image-translator`
- [x] Removed obsolete `todo.md` and `whatgptsays.txt`
