# Manga Learner - Projektstatus

## Aktueller Stand

Manga Learner besteht aktuell aus vier Kernbausteinen:

- **Manga Text Detection Engine**: findet Textbereiche, Speech Bubbles und Textzeilen im Panel.
- **MangaOCR Pipeline**: liest japanischen Text aus jedem erkannten Crop.
- **Translation Layer**: uebersetzt erkannte Texte im Batch und gibt Annotationen fuer die UI zurueck.
- **Translated Panel Renderer**: entfernt japanischen Text bestmoeglich aus dem Panel, rendert die englische Uebersetzung ins Bild und macht sie per JP/EN-Slider vergleichbar.

Die Detection funktioniert bereits ueberraschend stabil. Bounding Boxes fuer Textregionen sehen groesstenteils korrekt aus. Seit Emirs Merge ist das Projekt nicht mehr nur eine OCR-/Hover-Ansicht, sondern hat eine echte erste Manga-Translation-Ansicht mit gerendertem englischem Panel.

Die groessten offenen Qualitaetshebel liegen weiterhin bei OCR-Qualitaet, Furigana, echter Confidence, besserem Debugging, Tokenisierung und den eigentlichen Lernfeatures.

---

## Aktuelle Pipeline

```text
Manga page / panel
-> Manga Text Detection Engine
-> Textbox crops
-> OCR preprocessing
-> MangaOCR
-> OCR cleanup
-> Manga reading order
-> Batch translation
-> Annotation normalization
-> Translated panel rendering
-> Scanner UI with OCR boxes + JP/EN slider
```

Das Projekt behandelt die Text Detection und den nativen Renderer als eigene Produktmodule. Die App soll deshalb nicht wie ein Wrapper um eine externe Referenz wirken, sondern wie eine zusammenhaengende Manga-Lernanwendung mit eigener Detection-, OCR-, Rendering- und Learning-Schicht.

---

## Was bereits umgesetzt ist

### MangaOCR Integration

MangaOCR ist als primaere OCR-Stufe eingebunden.

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

Textboxen werden vor der Batch-Uebersetzung explizit sortiert:

```text
oben nach unten
innerhalb einer Zeile: rechts nach links
```

Das verbessert die Uebersetzung, weil Dialoge in sinnvollerer Reihenfolge beim Sprachmodell landen.

### OCR Cache

Erfolgreiche OCR-Ergebnisse werden anhand von Dateipfad, Dateigroesse und Aenderungszeit gecacht. Wiederholtes Scannen desselben Panels ist dadurch deutlich schneller.

### Native Translated Panel Renderer

Neu umgesetzt:

- `backend/services/panel_renderer.py`
- Render-Cache unter `backend/data/rendered_panels/`
- Static Route `/rendered-panels`
- Scan Response mit `translated_image_url`, `render_method` und `render_warnings`
- Textmasken aus OCR-Lines oder Bounding Boxes
- OpenCV Inpainting/Cleanup der japanischen Textbereiche
- englisches Text-Fitting mit Pillow
- Platzierungslogik gegen ueberlappende Uebersetzungen
- neutraler Platzhalter fuer fehlende Uebersetzungen

Das ist die erste echte Version von:

```text
OCR annotations
-> Text removal / inpainting
-> translated text rendering
-> translated panel image
```

### JP/EN Slider im Frontend

Neu umgesetzt:

- Originalbild und gerendertes Uebersetzungsbild liegen uebereinander.
- Der Slider steuert, wie viel vom englischen Panel sichtbar ist.
- Der JP/EN Toggle bewegt den Slider zu Original oder Uebersetzung.
- OCR-Hoverboxen bleiben als Overlay ueber dem Bild.
- Es gibt einen Fullscreen-Button fuer die Panelansicht.

### OCR Debug Panel

Der Debug Panel ist ausgebaut:

- Debug-Toggle in der Scanner UI
- Anzeige pro Box mit Reihenfolge, japanischem OCR-Text und Uebersetzung
- heuristische OCR-Confidence pro Box
- Qualitaetsstufen `good`, `warn`, `bad`
- farbige OCR-Boxen im Overlay
- ausgewaehlte OCR-Variante und Score
- ausklappbare Liste aller getesteten OCR-Kandidaten
- Original-Crop-Preview und Preview der gewaehlten OCR-Variante
- Preview-Bilder fuer alle OCR-Kandidaten
- OCR vergleicht mehrere Preprocessing-Modi: `raw_upscaled`, `contrast`, `threshold`
- vertikale Regionen testen diese Modi zusaetzlich mit `rot90_ccw` und `rot90_cw`
- OCR-Warnungen wie `low_ocr_score`, `very_short_text` oder `close_ocr_variant_scores`
- erkannte Box, Crop-Box, Richtung, Winkel, Font-Groesse und Lines-Anzahl

Das ist noch keine echte Modell-Confidence, aber jetzt ist sichtbar, warum die Pipeline eine OCR-Variante gewaehlt hat.

---

## Noch nicht umgesetzt / weiterhin offen

### Rich Inspector statt kleiner Hoverbox

Aktuell gibt es weiterhin einfache Hover-Tooltips auf den OCR-Boxen. Noch offen:

- groesserer Inspector
- Click-to-pin Verhalten
- aktive Box markieren
- lange Texte sauber scrollen/wrappen
- Debugdaten und Lernbereiche in einer stabilen Karte anzeigen

### Japanische Tokenisierung

Noch nicht umgesetzt:

- SudachiPy oder MeCab Integration
- Tokens pro Annotation
- Short/Middle/Long Split Modes
- Phrase Candidates
- tokenization status in der API

Ohne Tokenisierung koennen Kanji, Woerter und Phrasen noch nicht kontrolliert anklickbar gemacht werden.

### Kanji-, Wort- und Phrasen-Lookups

Noch nicht umgesetzt:

- `GET /api/learning/kanji/{character}`
- `GET /api/learning/word?text=...`
- Lookup-Cache unter `backend/data/lookup_cache/`
- kanjiapi.dev Integration
- Kanji Alive oder KanjiPortraits als optionale Quellen
- UI fuer Bedeutungen, Lesungen, Stroke Count, JLPT, Beispiele

Der vorhandene Learning Mode erzeugt weiterhin einfache Karten aus ganzen OCR-Annotationen. Er ist noch nicht scanner-aware auf Token-/Kanji-Ebene.

### Echte OCR Confidence

Die API gibt aktuell noch keine echte Modell-Confidence aus. Stattdessen gibt es jetzt eine heuristische Confidence aus OCR-Score und Warnungen.

Weiterhin gewuenscht:

- echte OCR-Modellwahrscheinlichkeit, falls das Modell oder eine Alternative sie liefert
- bessere Kalibrierung der heuristischen Confidence mit echten Beispielen
- bessere Gewichtung, welcher Preprocessing-Modus wann gewinnen sollte

### OCR Artefakte

Auch mit MangaOCR entstehen noch kaputte Texte:

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

### OCR Cleanup Layer

Aktuell gibt es nur leichtes Cleanup:

- Whitespace entfernen
- vertikale Trennzeichen entfernen
- fehlende Uebersetzungen normalisieren

Noch offen:

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
-> Rendering pro Panel oder ganze Seite
```

Das verbessert Lesereihenfolge, Kontext und spaetere Full-Page-Translation.

### Renderer-Qualitaet

Die native Renderer-Version ist umgesetzt, aber qualitativ noch ausbaufaehig:

- bessere Speech-Bubble-Innenraumerkennung
- bessere Textfarben-/Stil-Erkennung
- optional bessere Inpainting-Methode
- Umgang mit SFX und stark dekorativem Text
- visuelle QA auf verschiedenen Manga-Stilen

---

## Aktualisierte Prioritaeten

## Prioritaet 1

- Rich Inspector mit Click-to-pin statt nur Hover-Tooltip
- heuristische OCR Confidence anhand echter Scanbeispiele kalibrieren
- Furigana-Probleme sichtbar machen

## Prioritaet 2

- Furigana-Erkennung/Filterung experimentell verbessern
- Preprocessing-Modi anhand echter Panels auswerten und gewichten
- OCR Cleanup Layer vor der Uebersetzung einfuehren
- Reading Order mit sichtbaren Indizes gegen echte Panels pruefen

## Prioritaet 3

- panel-aware processing
- Translation mit Panel-Kontext
- Renderer-Qualitaet verbessern

## Prioritaet 4

- Backend-Tokenisierung mit SudachiPy
- kontrolliert anklickbare japanische Tokens im Inspector
- Kanji-Lookup mit Cache
- Wort-Lookup und erste Phrase Candidates
- scanner-aware Learning Mode
- JLPT-Level
- Kanji Breakdown
- Beispielsaetze
- Pitch Accent
- Spaced Repetition

---

## Naechster sinnvoller Meilenstein

Der beste naechste Produkt-Meilenstein ist jetzt OCR-Qualitaet, nicht Lookup:

```text
Scan eines schwierigen Panels
-> Debug Panel zeigt Gewinner und OCR-Kandidaten
-> Crop/Preprocessing-Probleme werden ueber Preview-Bilder sichtbar
-> Furigana und unsichere Regionen werden markiert
-> Heuristiken koennen gezielt verbessert werden
```

Damit wird die Grundlage stabil, bevor Lernfeatures auf den erkannten Texten aufbauen.

---

## Fazit

Die aktuelle Situation ist besser als der alte Status vermuten liess:

```text
gute Detection
+ MangaOCR Pipeline
+ Batch Translation
+ nativer translated panel renderer
+ JP/EN Slider
+ erster Debug Panel
```

Der groesste offene Hebel ist jetzt nicht mehr "uebersetzten Text irgendwann ins Bild rendern" und auch noch nicht "Kanji-Lookups", sondern:

```text
OCR-Qualitaet sichtbar machen
+ Debugdaten an echten Panels auswerten
+ Preprocessing/Furigana/Cleanup verbessern
```

Als naechstes sollte also nicht blind ein neues Modell eingebaut werden und auch noch nicht die Token-/Lookup-Schicht. Erst muessen Fehler aus Detection, Crop, Rotation, Preprocessing, OCR und Translation sichtbar und vergleichbar werden.
