# Manga Learner - Projektstatus

## Aktueller Stand

Manga Learner besteht aktuell aus drei Kernbausteinen:

- **Manga Text Detection Engine**: findet Textbereiche, Speech Bubbles und Textzeilen im Panel.
- **MangaOCR Pipeline**: liest japanischen Text aus jedem erkannten Crop.
- **Translation + Learning Layer**: uebersetzt erkannte Texte und macht sie im Lernmodus nutzbar.

Die Detection funktioniert bereits ueberraschend stabil. Bounding Boxes fuer Textregionen sehen groesstenteils korrekt aus. Die groessten Qualitaetshebel liegen weiterhin bei OCR-Qualitaet, vertikalem Text, Furigana, Reihenfolge, Cleanup und kontextbewusster Uebersetzung.

---

## Aktuelle Pipeline

```text
Manga page
-> Manga Text Detection Engine
-> Textbox crops
-> OCR preprocessing
-> MangaOCR
-> OCR cleanup
-> Manga reading order
-> Context-aware translation
-> UI rendering
```

Das Projekt behandelt die Text Detection als eigenes Produktmodul. Die UI und die Backend-Routen sollen deshalb nicht wie ein Wrapper um eine externe Referenz wirken, sondern wie eine zusammenhaengende Manga-Lernanwendung mit eigener Detection-Schicht.

---

## Was bereits verbessert wurde

### MangaOCR Integration

MangaOCR ist jetzt als primaere OCR-Stufe eingebunden.

Detection und OCR sind getrennt:

```text
Textregion erkennen
-> Crop erzeugen
-> Crop fuer OCR vorbereiten
-> MangaOCR auf Crop ausfuehren
```

Das ist deutlich robuster als OCR auf der ganzen Seite oder generisches Dokumenten-OCR.

### Preprocessing pro Textbox

Vor MangaOCR wird jeder Crop vorbereitet:

- Upscaling auf 2x bis 4x
- Grayscale
- CLAHE fuer lokalen Kontrast
- Denoising
- Adaptive Thresholding

Ziel: kleine Kana, schwacher Druck, graue Scans und verrauschte Hintergruende besser lesbar machen.

### Vertical Text Handling

Bei vertikalen Textregionen werden mehrere OCR-Varianten getestet:

```text
preprocessed
preprocessed_rot90_ccw
preprocessed_rot90_cw
```

Danach waehlt eine einfache Heuristik den plausibelsten OCR-Text aus. Bewertet werden unter anderem japanische Zeichen, Textlaenge und kaputte Zeichen.

### Reading Order

Textboxen werden vor der Batch-Uebersetzung nochmal explizit sortiert:

```text
oben nach unten
innerhalb einer Zeile: rechts nach links
```

Das verbessert die Uebersetzung, weil Dialoge in sinnvollerer Reihenfolge beim Sprachmodell landen.

### OCR Cache

Erfolgreiche OCR-Ergebnisse werden anhand von Dateipfad, Dateigroesse und Aenderungszeit gecacht. Wiederholtes Scannen desselben Panels ist dadurch deutlich schneller.

---

## Aktuelle Hauptprobleme

### OCR Artefakte

Auch mit MangaOCR koennen noch kaputte Texte entstehen:

- falsche Kana
- zusammengezogene Woerter
- fehlende Zeichen
- falsch gelesene kleine Zeichen
- unplausible Vertikaltext-Ergebnisse

### Furigana

Furigana ist noch nicht separat behandelt. Kleine Lesehilfen koennen Haupttext stoeren oder als eigener Unsinn erkannt werden.

Moegliche Ansaetze:

- sehr kleine Textkomponenten filtern
- Furigana separat erkennen
- Furigana optional ignorieren
- spaeter: Haupttext und Furigana in getrennten Layern anzeigen

### OCR Confidence

Die API gibt aktuell noch keine echte Modell-Confidence aus. Es gibt eine interne Kandidatenbewertung fuer OCR-Varianten, aber die UI nutzt sie noch nicht als Debug- oder Qualitaetsanzeige.

Gewuenscht:

- gruen: wahrscheinlich gut
- gelb: unsicher
- rot: vermutlich OCR-Problem

### OCR Cleanup Layer

Aktuell gibt es nur leichtes Cleanup:

- Whitespace entfernen
- vertikale Trennzeichen entfernen

Noch nicht umgesetzt:

```text
OCR
-> plausibilitaetsbasierte Korrektur
-> Translation
```

Ein LLM-Cleanup koennte OCR-Artefakte vor der Uebersetzung reparieren, muss aber vorsichtig sein, damit kein Text frei erfunden wird.

### Panel-aware Processing

Aktuell wird das Bild als Ganzes verarbeitet und danach werden Textregionen sortiert. Langfristig waere besser:

```text
Manga page
-> Panels erkennen
-> Textregionen pro Panel erkennen
-> OCR pro Panel
-> Uebersetzung pro Panel-Kontext
```

Das verbessert Lesereihenfolge, Kontext und spaetere Full-Page-Translation.

---

## Empfohlene Zielpipeline

```text
Manga page
-> Panel segmentation
-> Manga Text Detection Engine
-> Textbox crops
-> OCR preprocessing
-> MangaOCR
-> Furigana handling
-> OCR cleanup
-> Reading order
-> Context-aware translation
-> UI overlay
-> Learning extraction
```

---

## Kontextbewusste Uebersetzung

Textboxweise Uebersetzung bleibt schwierig, weil Japanisch oft Kontext aus vorherigen Sprechblasen braucht:

- implizite Subjekte
- ausgelassene Pronomen
- Satzteile ueber mehrere Boxen
- emotionale Nuancen

Besser:

```text
alle Boxen eines Panels gemeinsam uebersetzen
```

oder:

```text
vorherige Boxen als Kontext mitsenden
```

Die aktuelle Batch-Uebersetzung geht bereits in diese Richtung.

---

## OCR Debug Mode

Ein Debug Mode waere sehr hilfreich.

Sinnvolle Informationen:

- Bounding Boxes
- OCR crops
- preprocessing output
- erkannter Text
- ausgewaehlte OCR-Variante
- interne OCR-Bewertung
- reading order index
- panel id

Das wuerde sichtbar machen, ob Fehler aus Detection, Crop, Preprocessing, OCR oder Translation kommen.

---

## UI Feedback

Die UI wirkt bereits sauber. Das Projekt fuehlt sich aktuell nicht kaputt an, sondern eher wie:

```text
gute Detection
+ solide UI
+ OCR-Qualitaet als naechster grosser Hebel
```

Die Scanner-Ansicht mit Hover-Translation ist ein guter Kern. Als naechstes sollte die UI mehr Diagnoseinformationen anzeigen, wenn OCR unsicher ist.

---

## Prioritaeten

## Prioritaet 1

- OCR Debug Mode
- echte oder heuristische OCR Confidence anzeigen
- Furigana-Probleme sichtbar machen
- OCR-Varianten in der UI nachvollziehbar machen

## Prioritaet 2

- panel-aware processing
- besserer Cleanup Layer
- Translation mit Panel-Kontext
- Lesereihenfolge weiter verbessern

## Prioritaet 3

- Vocab Features
- JLPT-Level
- Kanji Breakdown
- Beispielsaetze
- Pitch Accent
- Spaced Repetition

---

## Langfristige Ideen

### Hover Translation

- Originaltext anzeigen
- Uebersetzung beim Hover
- Confidence/Debug-Status optional einblenden

### Woerter anklickbar machen

- Bedeutung
- Lesung
- Kanji Breakdown
- Beispielsatz
- Lernstatus

### Spaced Repetition

Anki-artiges Lernen direkt aus gescannten Manga-Panels.

### Audio / TTS

Japanischen Text vorlesen lassen.

Moegliche Richtungen:

- lokale TTS-Modelle
- Cloud TTS
- spaeter verschiedene Stimmen

### Full Manga Translation

Spaeter eventuell:

```text
Page
-> OCR
-> Translation
-> Text removal / inpainting
-> translated text rendering
```

---

## Modelle und Technologien fuer spaeter

### OCR

- MangaOCR
- PaddleOCR Japanese
- TrOCR
- PARSeq
- Florence
- GOT-OCR

### Detection

- eigene Manga Text Detection Engine weiter ausbauen
- Panel segmentation
- layout-aware detection
- alternative ONNX/YOLO-basierte Modelle testen

### Translation

- Ollama local models
- Qwen
- DeepL
- GPT
- NLLB

---

## Fazit

Die aktuelle Situation ist gut: Detection und UI sind stabil genug, um iterativ an OCR-Qualitaet zu arbeiten.

Der groesste Hebel war und bleibt:

```text
MangaOCR + besseres preprocessing + bessere Debug-Sichtbarkeit
```

Als naechstes sollte nicht blind ein neues Modell eingebaut werden, sondern sichtbar gemacht werden, wo die Pipeline scheitert: Crop, Rotation, Preprocessing, OCR oder Translation.
