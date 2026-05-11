from typing import List, Dict
import os
import math
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps
from config import GOOGLE_APPLICATION_CREDENTIALS
import logging

logger = logging.getLogger(__name__)

try:
    from google.cloud import vision
    HAS_GOOGLE_VISION = True
except ImportError:
    HAS_GOOGLE_VISION = False

try:
    from manga_ocr import MangaOcr
    HAS_MANGA_OCR = True
except ImportError:
    HAS_MANGA_OCR = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

# Minimum confidence to keep an OCR result
MIN_CONFIDENCE = 0.1
# Distance threshold (pixels, in original image coords) for grouping text blocks
GROUP_DISTANCE_THRESHOLD = 50
# Minimum area ratio for a speech bubble (fraction of total image area)
MIN_BUBBLE_AREA_RATIO = 0.002
# Maximum area ratio for a speech bubble
MAX_BUBBLE_AREA_RATIO = 0.15


class OCRService:
    def __init__(self):
        self.use_google_vision = HAS_GOOGLE_VISION and bool(GOOGLE_APPLICATION_CREDENTIALS)
        self.manga_ocr = None
        self.easyocr_reader = None

        if self.use_google_vision:
            try:
                self.client = vision.ImageAnnotatorClient()
                logger.info("Google Vision OCR Service initialized")
            except Exception as e:
                logger.warning(f"Google Vision not available: {e}")
                self.use_google_vision = False

        if not self.use_google_vision and HAS_MANGA_OCR:
            try:
                self.manga_ocr = MangaOcr()
                logger.info("manga-ocr initialized (kha-white/manga-ocr-base)")
            except Exception as e:
                logger.warning(f"manga-ocr init failed: {e}")

        if not self.use_google_vision and not self.manga_ocr and HAS_EASYOCR:
            try:
                self.easyocr_reader = easyocr.Reader(['ja', 'en'], gpu=False)
                logger.info("EasyOCR initialized (Japanese + English, fallback)")
            except Exception as e:
                logger.warning(f"EasyOCR init failed: {e}")

    def extract_text_from_image(self, image_path: str) -> Dict:
        if self.use_google_vision:
            return self._google_vision_ocr(image_path)
        if self.manga_ocr:
            return self._manga_ocr(image_path)
        if self.easyocr_reader:
            return self._easyocr(image_path)
        return {
            "success": False,
            "error": "No OCR engine available. Install manga-ocr or configure Google Vision.",
            "text": "",
            "annotations": []
        }

    def _google_vision_ocr(self, image_path: str) -> Dict:
        try:
            with open(image_path, "rb") as f:
                content = f.read()

            image = vision.Image(content=content)
            response = self.client.text_detection(image=image)
            texts = response.text_annotations

            if response.error.message:
                return {"success": False, "error": response.error.message, "text": "", "annotations": []}

            full_text = texts[0].description if texts else ""
            annotations = []
            for text in texts[1:]:
                annotations.append({
                    "text": text.description,
                    "bbox": [
                        [vertex.x, vertex.y]
                        for vertex in text.bounding_poly.vertices
                    ]
                })

            return {"success": True, "text": full_text, "annotations": annotations}
        except Exception as e:
            logger.error(f"Google Vision OCR failed: {e}")
            return {"success": False, "error": str(e), "text": "", "annotations": []}

    def _manga_ocr(self, image_path: str) -> Dict:
        """Use manga-ocr with speech bubble detection for high-quality manga text recognition."""
        try:
            orig_img = Image.open(image_path)
            orig_w, orig_h = orig_img.size

            # Step 1: Detect speech bubbles
            bubbles = self._detect_speech_bubbles(image_path)
            logger.info(f"manga-ocr: detected {len(bubbles)} speech bubbles")

            annotations = []

            if bubbles:
                # Step 2: Run manga-ocr on each bubble crop
                for bubble in bubbles:
                    bx, by, bw, bh = bubble["bbox"]
                    # Crop the bubble region from the original image
                    crop = orig_img.crop((bx, by, bx + bw, by + bh))

                    # manga-ocr works on PIL images directly
                    text = self.manga_ocr(crop)
                    text = text.strip()

                    if not text:
                        continue

                    # Use the bubble bbox as the annotation bbox
                    bbox = [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]]

                    annotations.append({
                        "text": text,
                        "confidence": 0.95,  # manga-ocr doesn't provide confidence, assume high
                        "bbox": bbox,
                        "char_count": len(text)
                    })

                    logger.info(f"  Bubble ({bx},{by} {bw}x{bh}): '{text}'")
            else:
                # Fallback: run on the whole image if no bubbles detected
                text = self.manga_ocr(orig_img)
                text = text.strip()
                if text:
                    annotations.append({
                        "text": text,
                        "confidence": 0.95,
                        "bbox": [[0, 0], [orig_w, 0], [orig_w, orig_h], [0, orig_h]],
                        "char_count": len(text)
                    })

            full_text = "\n".join(a["text"] for a in annotations)

            return {
                "success": True,
                "text": full_text,
                "annotations": annotations,
                "method": "manga-ocr",
                "bubbles_detected": len(bubbles),
                "recognized_blocks": len(annotations)
            }
        except Exception as e:
            logger.error(f"manga-ocr failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "text": "", "annotations": []}

    def _detect_speech_bubbles(self, image_path: str) -> List[Dict]:
        """Detect speech bubble regions in a manga panel.
        Uses multiple strategies: contour detection + connected components."""
        img = cv2.imread(image_path)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        total_area = h * w

        bubbles = []

        # Strategy 1: Otsu threshold to find bright areas (speech bubbles)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Strategy 2: Also try fixed threshold for very white bubbles
        _, fixed = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Combine both (union)
        combined = cv2.bitwise_and(otsu, fixed)

        # Clean up: close small gaps, remove noise
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # Erode slightly to separate touching regions
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.erode(combined, kernel_erode, iterations=2)

        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None:
            return []

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            area_ratio = area / total_area

            # Filter by area
            if area_ratio < MIN_BUBBLE_AREA_RATIO or area_ratio > MAX_BUBBLE_AREA_RATIO:
                continue

            # Get bounding rectangle
            x, y, bw, bh = cv2.boundingRect(contour)

            # Filter by aspect ratio (skip very thin/wide shapes)
            aspect = max(bw, bh) / (min(bw, bh) + 1)
            if aspect > 6:
                continue

            # Filter by solidity (speech bubbles are fairly solid/convex)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.4:  # Very irregular shape, probably not a bubble
                    continue

            # Check if region actually contains dark pixels (text)
            roi = gray[y:y+bh, x:x+bw]
            dark_pixels = np.sum(roi < 100) / (bw * bh + 1)
            if dark_pixels < 0.02:  # No text content
                continue
            if dark_pixels > 0.7:  # Mostly dark (not a white bubble)
                continue

            # Add padding
            pad_x = int(bw * 0.05)
            pad_y = int(bh * 0.05)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)

            crop = img[y1:y2, x1:x2]

            bubbles.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "crop": crop,
                "area_ratio": area_ratio
            })

        # Remove duplicates (overlapping regions)
        bubbles = self._remove_overlapping_bubbles(bubbles)

        # Sort by manga reading order: top-to-bottom, right-to-left
        bubbles.sort(key=lambda b: (b["bbox"][1] // (h // 6), -b["bbox"][0]))

        logger.info(f"Detected {len(bubbles)} speech bubble regions")
        return bubbles

    @staticmethod
    def _remove_overlapping_bubbles(bubbles: List[Dict]) -> List[Dict]:
        """Remove duplicate/overlapping bubble detections."""
        if len(bubbles) <= 1:
            return bubbles

        # Sort by area descending (keep larger ones)
        bubbles.sort(key=lambda b: b["bbox"][2] * b["bbox"][3], reverse=True)
        keep = []

        for bubble in bubbles:
            bx, by, bw, bh = bubble["bbox"]
            is_duplicate = False
            for kept in keep:
                kx, ky, kw, kh = kept["bbox"]
                # Check overlap (IoU-like)
                ix1, iy1 = max(bx, kx), max(by, ky)
                ix2, iy2 = min(bx + bw, kx + kw), min(by + bh, ky + kh)
                if ix2 > ix1 and iy2 > iy1:
                    overlap_area = (ix2 - ix1) * (iy2 - iy1)
                    smaller_area = min(bw * bh, kw * kh)
                    if overlap_area / (smaller_area + 1) > 0.5:
                        is_duplicate = True
                        break
            if not is_duplicate:
                keep.append(bubble)

        return keep

    def _ocr_region(self, crop: np.ndarray) -> List:
        """Run OCR on a cropped speech bubble region."""
        h, w = crop.shape[:2]

        # Scale up small crops for better recognition
        scale = 1.0
        if max(h, w) < 300:
            scale = 300 / max(h, w)
            crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)

        # Run EasyOCR on the cropped region directly
        results = self.easyocr_reader.readtext(
            crop,
            detail=1,
            paragraph=False,       # Get individual text detections
            text_threshold=0.4,
            low_text=0.3,
            link_threshold=0.4,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            width_ths=1.5,
            height_ths=1.5,
            min_size=8,
            mag_ratio=1.5,
        )

        # Scale bbox back if we upscaled
        if scale != 1.0:
            scaled_results = []
            for item in results:
                if len(item) == 3:
                    bbox, text, conf = item
                    bbox = [[p[0] / scale, p[1] / scale] for p in bbox]
                    scaled_results.append((bbox, text, conf))
                else:
                    scaled_results.append(item)
            return scaled_results

        return results

    def _preprocess_manga_image(self, image_path: str) -> np.ndarray:
        """Preprocess manga panel for better OCR: resize to optimal size."""
        img = Image.open(image_path)

        # For large images, resize down to reasonable size for EasyOCR
        # CRAFT detection works best on ~1500px images
        w, h = img.size
        max_dim = 1500
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        elif max(w, h) < 800:
            scale = 800 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        return np.array(img)

    @staticmethod
    def _bbox_center(bbox):
        """Get center point of a bounding box."""
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @staticmethod
    def _bbox_distance(bbox1, bbox2):
        """Get minimum edge distance between two bounding boxes."""
        # Use bounding rectangles
        x1_min, x1_max = min(p[0] for p in bbox1), max(p[0] for p in bbox1)
        y1_min, y1_max = min(p[1] for p in bbox1), max(p[1] for p in bbox1)
        x2_min, x2_max = min(p[0] for p in bbox2), max(p[0] for p in bbox2)
        y2_min, y2_max = min(p[1] for p in bbox2), max(p[1] for p in bbox2)

        dx = max(0, max(x1_min - x2_max, x2_min - x1_max))
        dy = max(0, max(y1_min - y2_max, y2_min - y1_max))
        return math.sqrt(dx * dx + dy * dy)

    def _group_annotations(self, annotations: List[Dict]) -> List[Dict]:
        """Group nearby text detections into speech bubble blocks."""
        if not annotations:
            return []

        # Build groups using simple distance-based clustering
        used = [False] * len(annotations)
        groups = []

        for i in range(len(annotations)):
            if used[i]:
                continue
            group = [i]
            used[i] = True
            # Find all annotations close to any member of this group
            changed = True
            while changed:
                changed = False
                for j in range(len(annotations)):
                    if used[j]:
                        continue
                    for member_idx in group:
                        dist = self._bbox_distance(
                            annotations[member_idx]["bbox"],
                            annotations[j]["bbox"]
                        )
                        if dist < GROUP_DISTANCE_THRESHOLD:
                            group.append(j)
                            used[j] = True
                            changed = True
                            break
            groups.append(group)

        # Merge each group into a single annotation
        merged = []
        for group in groups:
            group_anns = [annotations[i] for i in group]

            # Sort by vertical position (top to bottom for vertical Japanese text),
            # then right to left for manga reading order
            group_anns.sort(key=lambda a: (
                min(p[1] for p in a["bbox"]),  # top Y
                -min(p[0] for p in a["bbox"])   # rightmost first (manga reading)
            ))

            # Combine text
            combined_text = "".join(a["text"] for a in group_anns)

            # Merge bounding boxes
            all_points = []
            for a in group_anns:
                all_points.extend(a["bbox"])
            x_min = min(p[0] for p in all_points)
            y_min = min(p[1] for p in all_points)
            x_max = max(p[0] for p in all_points)
            y_max = max(p[1] for p in all_points)

            avg_confidence = sum(a.get("confidence", 0.5) for a in group_anns) / len(group_anns)

            merged.append({
                "text": combined_text,
                "confidence": float(avg_confidence),
                "bbox": [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                "char_count": len(group_anns)
            })

        return merged

    def _easyocr(self, image_path: str) -> Dict:
        try:
            # Get original image dimensions
            orig_img = Image.open(image_path)
            orig_w, orig_h = orig_img.size

            # Step 1: Detect speech bubble regions for filtering
            bubbles = self._detect_speech_bubbles(image_path)
            bubble_rects = [(b["bbox"][0], b["bbox"][1],
                            b["bbox"][0] + b["bbox"][2], b["bbox"][1] + b["bbox"][3])
                           for b in bubbles]

            # Step 2: Run EasyOCR on the full image
            # Read as numpy array via cv2 for consistent handling
            img_array = cv2.imread(image_path)
            h, w = img_array.shape[:2]

            # Don't resize - keep full resolution for better text detection
            # EasyOCR's mag_ratio handles magnification internally
            img_resized = img_array
            scale = 1.0

            results = self.easyocr_reader.readtext(
                img_resized,
                detail=1,
                paragraph=False,
                text_threshold=0.3,
                low_text=0.2,
                link_threshold=0.3,
                contrast_ths=0.1,
                adjust_contrast=0.5,
                width_ths=0.7,
                height_ths=0.7,
                min_size=10,
                mag_ratio=1.0,
            )

            logger.info(f"Full-image OCR: {len(results)} raw detections")

            # Step 3: Filter detections - keep only those within speech bubbles
            annotations = []
            outside_bubble = []

            for item in results:
                if len(item) == 3:
                    bbox, text, confidence = item
                elif len(item) == 2:
                    bbox, text = item
                    confidence = 0.5
                else:
                    continue

                clean_text = text.strip()
                if not clean_text or confidence < MIN_CONFIDENCE:
                    continue
                if all(c in '.,;:!?-_=+*/\\|[]{}()<>@#$%^&~`\'"‥…。、' for c in clean_text):
                    continue

                # Scale bbox back to original coordinates
                if scale != 1.0:
                    orig_bbox = [[int(p[0] / scale), int(p[1] / scale)] for p in bbox]
                else:
                    orig_bbox = [[int(p[0]), int(p[1])] for p in bbox]

                # Check if this detection is inside a speech bubble
                cx = sum(p[0] for p in orig_bbox) / 4
                cy = sum(p[1] for p in orig_bbox) / 4

                in_bubble = False
                for bx1, by1, bx2, by2 in bubble_rects:
                    if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                        in_bubble = True
                        break

                ann = {
                    "text": clean_text,
                    "confidence": float(confidence),
                    "bbox": orig_bbox,
                    "in_bubble": in_bubble
                }

                if in_bubble:
                    annotations.append(ann)
                else:
                    outside_bubble.append(ann)

            # If we found bubbles, only use in-bubble detections
            # If no bubbles found, use all detections (fallback)
            if bubbles and annotations:
                final_annotations = annotations
            else:
                final_annotations = annotations + outside_bubble

            logger.info(f"In-bubble: {len(annotations)}, outside: {len(outside_bubble)}")

            # Group nearby annotations into text blocks
            grouped = self._group_annotations(final_annotations)
            logger.info(f"Final grouped: {len(grouped)} text blocks")

            full_text = "\n".join(a["text"] for a in grouped)

            return {
                "success": True,
                "text": full_text,
                "annotations": grouped,
                "method": "easyocr",
                "bubbles_detected": len(bubbles),
                "raw_detections": len(results),
                "grouped_blocks": len(grouped)
            }
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "text": "", "annotations": []}

    def get_service_status(self) -> Dict:
        if self.use_google_vision:
            engine = "google_vision"
        elif self.manga_ocr:
            engine = "manga-ocr"
        elif self.easyocr_reader:
            engine = "easyocr"
        else:
            engine = "none"
        return {"ocr_service": engine, "available": engine != "none"}
