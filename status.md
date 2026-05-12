# feedback.md

# Manga Learner – Projektstatus / Feedback / Ideen

## Aktueller Stand

Die aktuelle Pipeline basiert hauptsächlich auf:

- comic-text-detector
- OCR
- Übersetzung per LLM

Das Projekt orientiert sich aktuell stark an:
https://github.com/dmMaze/comic-text-detector

Die Detection funktioniert bereits überraschend gut.
Die Bounding Boxes für Speech Bubbles und Textregionen sehen größtenteils korrekt aus.

Die Hauptprobleme liegen aktuell nicht bei der UI oder der Detection, sondern hauptsächlich bei:

- OCR / Text Recognition
- vertikalem Text
- Furigana
- Reihenfolge der Textboxen
- OCR Cleanup
- Übersetzungen bei OCR-Fehlern

---

# Aktuelle Pipeline

Aktuell ungefähr:

```text
Page
→ comic-text-detector
→ OCR
→ Translation
→ UI
```

Problem:
comic-text-detector ist hauptsächlich für Detection zuständig und nicht für hochwertiges Manga-OCR.

Das erklärt warum:

- die roten Boxen gut aussehen
- der erkannte Text aber oft kaputt ist

---

# OCR Probleme

Der aktuelle OCR-Output produziert oft:

- zufällige Hiragana-Ketten
- falsche Wörter
- unlesbare Sätze
- zerstörten Vertikaltext

Beispiele:

```text
全国ネピ
こんではないったくわねえ
```

Das sind keine echten japanischen Sätze mehr, sondern OCR-Artefakte.

---

# Warum normales OCR bei Manga schlecht funktioniert

Normales OCR ist meistens trainiert auf:

- Dokumente
- klare Fonts
- horizontale Texte
- hohe Auflösung

Manga dagegen enthält:

- handschriftartige Fonts
- vertikale Texte
- Furigana
- kleine Kana
- schlechte Scans
- alte Druckqualität
- komplexe Hintergründe
- variable Schriftgrößen

Dadurch brechen viele Standard-OCR-Modelle schnell auseinander.

---

# Größter nächster Schritt

## MangaOCR integrieren

Aktuell scheint generisches OCR verwendet zu werden.

Stattdessen testen:

https://github.com/kha-white/manga-ocr

Installation:

```bash
pip install manga-ocr
```

Beispiel:

```python
from manga_ocr import MangaOcr
from PIL import Image

mocr = MangaOcr()

text = mocr(Image.open("crop.png"))
print(text)
```

Wichtig:
comic-text-detector sollte nicht ersetzt werden.

Sinnvoller wäre:

```text
comic-text-detector
→ Crop jeder Box
→ MangaOCR
```

Die Detection funktioniert bereits brauchbar.

---

# Empfohlene Zielpipeline

```text
Page
→ Panel Detection
→ Text Detection
→ Crop jeder Textbox
→ Preprocessing
→ MangaOCR
→ Cleanup
→ Reihenfolge bestimmen
→ Übersetzung
→ UI Rendering
```

---

# Preprocessing verbessern

Aktuell wirkt es so, als würde OCR direkt auf den Raw-Crops laufen.

Vor OCR sollte preprocessing stattfinden.

## Grayscale

```python
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

## Contrast Enhancement

CLAHE testen:

```python
cv2.createCLAHE()
```

## Thresholding

```python
cv2.adaptiveThreshold()
```

## Denoising

```python
cv2.fastNlMeansDenoising()
```

## Upscaling

2x oder 4x Upscaling testen.

Möglichkeiten:

- ESRGAN
- waifu2x
- RealESRGAN
- OpenCV resize

MangaOCR profitiert stark von höherer Auflösung.

---

# Vertical Text Handling

Vertikaler Text ist aktuell wahrscheinlich einer der größten Failure Points.

Viele Manga nutzen:

- vertikale Leserichtung
- gemischte Leserichtungen
- kleine Furigana

Ein einfacher erster Ansatz:

```python
if height > width:
    rotate image 90°
    OCR
```

Danach zurückdrehen.

---

# Furigana Handling

Furigana verursacht aktuell wahrscheinlich viele OCR-Fehler.

Ideen:

- kleine Zeichen filtern
- Furigana separat erkennen
- optional ignorieren

---

# Reihenfolge der Textboxen

Aktuell vermutlich noch nicht optimal.

Japanische Leserichtung:

```text
rechts → links
oben → unten
```

Viele Standard-Pipelines sortieren westlich:

```text
links → rechts
```

Das zerstört Kontext und Übersetzungen.

---

# Panel-aware Verarbeitung

Aktuell wahrscheinlich:

```text
ganze Seite → OCR
```

Besser:

```text
Page
→ Panels erkennen
→ OCR pro Panel
```

Vorteile:

- bessere Reihenfolge
- besserer Kontext
- weniger OCR-Konflikte
- bessere Übersetzungen

---

# OCR Cleanup Layer

Aktuell wahrscheinlich:

```text
OCR → Translate
```

Problem:
Wenn OCR Müll produziert, übersetzt der Translator ebenfalls Müll.

Besser:

```text
OCR
→ Cleanup
→ Translation
```

Mögliche Modelle:

- GPT
- DeepSeek
- Qwen
- lokale Modelle

---

# Kontext-aware Übersetzung

Textboxweise Übersetzung ist problematisch.

Japanisch nutzt oft:

- implizite Subjekte
- ausgelassene Pronomen
- Kontextreferenzen

Sinnvoller:

```text
alle Boxen eines Panels gemeinsam übersetzen
```

oder:

```text
vorherige Boxen als Kontext mitsenden
```

---

# OCR Confidence anzeigen

Hilfreich fürs Debugging.

UI Idee:

- grün = hohe confidence
- gelb = mittel
- rot = schlecht

Dann sofort sichtbar:
- wo OCR versagt

---

# OCR Debug Mode

Nützliche Informationen:

- Bounding Boxes
- OCR crops
- preprocessing output
- confidence
- erkannter Text
- reading order
- panel ids

---

# UI Feedback

Die UI wirkt bereits ziemlich sauber.

Das Projekt wirkt aktuell eher wie:

```text
gute Detection + schwaches OCR backend
```

und nicht wie ein grundsätzlich kaputtes System.

---

# Langfristige Ideen

## Hover Translation

- Originaltext anzeigen
- Übersetzung beim Hover

---

## Wörter anklickbar machen

Features:

- JLPT Level
- Bedeutung
- Kanji Breakdown
- Beispielsätze
- Pitch Accent

---

## Spaced Repetition

Anki-artiges Lernen direkt aus Manga.

---

## Audio / TTS

Japanese Text vorlesen.

Möglichkeiten:

- Kokoro TTS
- Coqui
- ElevenLabs
- OpenAI TTS

---

# Full Manga Translation

Später eventuell:

```text
Page
→ OCR
→ Translation
→ Inpainting
→ neuen Text rendern
```

---

# Interessante Modelle später testen

## OCR

- MangaOCR
- PaddleOCR Japanese
- TrOCR
- PARSeq
- Florence
- GOT-OCR

## Detection

- comic-text-detector
- YOLO
- GroundingDINO

## Translation

- GPT
- Qwen
- DeepL
- NLLB

---

# Weitere Ideen

## Offline Mode

Lokale Modelle unterstützen.

---

## Mobile Support

Handy Manga Reader.

---

## Browser Extension

OCR direkt im Browser.

---

## Live Camera OCR

Manga direkt mit Kamera lesen.

---

# Prioritäten

## Priorität 1

- MangaOCR integrieren
- preprocessing verbessern
- vertical text handling

## Priorität 2

- reading order
- panel-aware translation
- OCR confidence

## Priorität 3

- cleanup layer
- vocab features
- translation rendering

---

# Fazit

Die aktuelle Situation ist eigentlich ziemlich gut.

Die Detection funktioniert bereits relativ stabil.
Das Hauptproblem ist aktuell die OCR-Qualität.

Das ist deutlich einfacher iterativ zu verbessern als:

- schlechte Detection
- schlechte Architektur
- kaputte UI

Der größte Hebel dürfte aktuell sein:

```text
MangaOCR + besseres preprocessing
```