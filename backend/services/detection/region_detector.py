"""
Manga/comic text region detector using the comic-text-detector ONNX model.

Faithfully replicates the detection pipeline from:
  https://github.com/dmMaze/comic-text-detector

Uses the model from https://github.com/zyddnys/manga-image-translator via OpenCV
DNN backend.  The model outputs three heads:
  - blk: YOLOv5-based text block detector (bounding boxes)
  - mask: pixel-level text segmentation mask (UNet)
  - lines_map: DBNet text-line detection map

Pipeline (matching the original repo):
  1. Preprocess: letterbox resize to 1024x1024, normalize
  2. Inference: 3 heads via cv2.dnn
  3. Post-process blocks: YOLOv5 NMS, rescale
  4. Post-process mask: squeeze, binarize, resize
  5. Post-process lines: SegDetectorRepresenter (DBNet contour extraction + unclip)
  6. Group output: assign lines to blocks, examine textblocks, merge scattered,
     split blocks, sort in reading order
  7. Refine mask per text block
"""

import copy
import cv2
import logging
import math
import numpy as np
import pyclipper
from pathlib import Path
from shapely.geometry import Polygon
from typing import List, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model paths & thresholds (matching comic-text-detector defaults)
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
_MODEL_PATH = _MODEL_DIR / "comictextdetector.pt.onnx"
_MODEL_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/"
    "download/beta-0.2.1/comictextdetector.pt.onnx"
)
_INPUT_SIZE = 1024
_CONF_THRESH = 0.4
_NMS_THRESH = 0.35
_MASK_THRESH = 0.3
_BOX_THRESH = 0.6

LANG_LIST = ['eng', 'ja', 'unknown']
LANGCLS2IDX = {'eng': 0, 'ja': 1, 'unknown': 2}

REFINEMASK_INPAINT = 0
REFINEMASK_ANNOTATION = 1

# Lazy-loaded singleton
_net = None


# ===========================================================================
# TextBlock class (from comic-text-detector/utils/textblock.py)
# ===========================================================================
class TextBlock:
    def __init__(self, xyxy: List,
                 lines: List = None,
                 language: str = 'unknown',
                 vertical: bool = False,
                 font_size: float = -1,
                 distance: List = None,
                 angle: int = 0,
                 vec: List = None,
                 norm: float = -1,
                 merged: bool = False,
                 weight: float = -1,
                 text: List = None,
                 translation: str = "",
                 fg_r=0, fg_g=0, fg_b=0,
                 bg_r=0, bg_g=0, bg_b=0,
                 line_spacing=1.,
                 font_family: str = "",
                 bold: bool = False,
                 underline: bool = False,
                 italic: bool = False,
                 alignment: int = -1,
                 alpha: float = 255,
                 rich_text: str = "",
                 _bounding_rect: List = None,
                 accumulate_color=True,
                 default_stroke_width=0.2,
                 target_lang: str = "",
                 **kwargs) -> None:
        self.xyxy = [int(num) for num in xyxy]
        self.lines = [] if lines is None else lines
        self.vertical = vertical
        self.language = language
        self.font_size = font_size
        self.distance = None if distance is None else np.array(distance, np.float64)
        self.angle = angle
        self.vec = None if vec is None else np.array(vec, np.float64)
        self.norm = norm
        self.merged = merged
        self.weight = weight
        self.text = text if text is not None else []
        self.prob = 1
        self.translation = translation
        self.fg_r = fg_r
        self.fg_g = fg_g
        self.fg_b = fg_b
        self.bg_r = bg_r
        self.bg_g = bg_g
        self.bg_b = bg_b
        self.font_family = font_family
        self.bold = bold
        self.underline = underline
        self.italic = italic
        self.alpha = alpha
        self.rich_text = rich_text
        self.line_spacing = line_spacing
        self._alignment = alignment
        self._target_lang = target_lang
        self._bounding_rect = _bounding_rect
        self.default_stroke_width = default_stroke_width
        self.accumulate_color = accumulate_color

    def adjust_bbox(self, with_bbox=False):
        lines = self.lines_array().astype(np.int32)
        if with_bbox:
            self.xyxy[0] = min(lines[..., 0].min(), self.xyxy[0])
            self.xyxy[1] = min(lines[..., 1].min(), self.xyxy[1])
            self.xyxy[2] = max(lines[..., 0].max(), self.xyxy[2])
            self.xyxy[3] = max(lines[..., 1].max(), self.xyxy[3])
        else:
            self.xyxy[0] = lines[..., 0].min()
            self.xyxy[1] = lines[..., 1].min()
            self.xyxy[2] = lines[..., 0].max()
            self.xyxy[3] = lines[..., 1].max()

    def sort_lines(self):
        if self.distance is not None:
            idx = np.argsort(self.distance)
            self.distance = self.distance[idx]
            lines = np.array(self.lines, dtype=np.int32)
            self.lines = lines[idx].tolist()

    def lines_array(self, dtype=np.float64):
        return np.array(self.lines, dtype=dtype)

    def center(self):
        xyxy = np.array(self.xyxy)
        return (xyxy[:2] + xyxy[2:]) / 2

    def min_rect(self, rotate_back=True):
        angled = self.angle != 0
        center = self.center()
        polygons = self.lines_array().reshape(-1, 8)
        if angled:
            polygons = rotate_polygons(center, polygons, self.angle)
        min_x = polygons[:, ::2].min()
        min_y = polygons[:, 1::2].min()
        max_x = polygons[:, ::2].max()
        max_y = polygons[:, 1::2].max()
        min_bbox = np.array([[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]])
        if angled and rotate_back:
            min_bbox = rotate_polygons(center, min_bbox, -self.angle)
        return min_bbox.reshape(-1, 4, 2).astype(np.int64)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

    def to_dict(self):
        blk_dict = copy.deepcopy(vars(self))
        return blk_dict

    def get_transformed_region(self, img, idx, textheight) -> np.ndarray:
        im_h, im_w = img.shape[:2]
        direction = 'v' if self.vertical else 'h'
        src_pts = np.array(self.lines[idx], dtype=np.float64)

        if self.language == 'eng' or (self.language == 'unknown' and not self.vertical):
            e_size = self.font_size / 3
            src_pts[..., 0] += np.array([-e_size, e_size, e_size, -e_size])
            src_pts[..., 1] += np.array([-e_size, -e_size, e_size, e_size])
            src_pts[..., 0] = np.clip(src_pts[..., 0], 0, im_w)
            src_pts[..., 1] = np.clip(src_pts[..., 1], 0, im_h)

        middle_pnt = (src_pts[[1, 2, 3, 0]] + src_pts) / 2
        vec_v = middle_pnt[2] - middle_pnt[0]
        vec_h = middle_pnt[1] - middle_pnt[3]
        ratio = np.linalg.norm(vec_v) / np.linalg.norm(vec_h)

        if direction == 'h':
            h = int(textheight)
            w = int(round(textheight / ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            region = cv2.warpPerspective(img, M, (w, h))
        elif direction == 'v':
            w = int(textheight)
            h = int(round(textheight * ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            region = cv2.warpPerspective(img, M, (w, h))
            region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return region

    def get_text(self):
        if isinstance(self.text, str):
            return self.text
        return ' '.join(self.text).strip()


# ===========================================================================
# Geometry utilities (from comic-text-detector/utils/imgproc_utils.py)
# ===========================================================================
def union_area(bboxa, bboxb):
    x1 = max(bboxa[0], bboxb[0])
    y1 = max(bboxa[1], bboxb[1])
    x2 = min(bboxa[2], bboxb[2])
    y2 = min(bboxa[3], bboxb[3])
    if y2 < y1 or x2 < x1:
        return -1
    return (y2 - y1) * (x2 - x1)


def xywh2xyxypoly(xywh, to_int=True):
    """4 points bbox (x,y,w,h) to 8 points polygon."""
    xyxypoly = np.tile(xywh[:, [0, 1]], 4)
    xyxypoly[:, [2, 4]] += xywh[:, [2]]
    xyxypoly[:, [5, 7]] += xywh[:, [3]]
    if to_int:
        xyxypoly = xyxypoly.astype(np.int64)
    return xyxypoly


def rotate_polygons(center, polygons, rotation, new_center=None, to_int=True):
    if new_center is None:
        new_center = center
    rotation = np.deg2rad(rotation)
    s, c = np.sin(rotation), np.cos(rotation)
    polygons = polygons.astype(np.float32)
    polygons[:, 1::2] -= center[1]
    polygons[:, ::2] -= center[0]
    rotated = np.copy(polygons)
    rotated[:, 1::2] = polygons[:, 1::2] * c - polygons[:, ::2] * s
    rotated[:, ::2] = polygons[:, 1::2] * s + polygons[:, ::2] * c
    rotated[:, 1::2] += new_center[1]
    rotated[:, ::2] += new_center[0]
    if to_int:
        return rotated.astype(np.int64)
    return rotated


def expand_textwindow(img_size, xyxy, expand_r=8, shrink=False):
    im_h, im_w = img_size[:2]
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    paddings = int(round((max(h, w) * 0.25 + min(h, w) * 0.75) / expand_r))
    if shrink:
        paddings *= -1
    x1, y1 = max(0, x1 - paddings), max(0, y1 - paddings)
    x2, y2 = min(im_w - 1, x2 + paddings), min(im_h - 1, y2 + paddings)
    return [x1, y1, x2, y2]


# ===========================================================================
# SegDetectorRepresenter (from comic-text-detector/utils/db_utils.py)
# ===========================================================================
class SegDetectorRepresenter:
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=1.5):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, batch_size, pred, is_output_polygon=False):
        """
        pred: DBNet lines_map output from model, shape (1, 2, H, W) or (2, H, W) or (1, H, W)
        Returns: (boxes_batch, scores_batch)
        """
        if isinstance(pred, np.ndarray):
            if pred.ndim == 4:
                pred = pred[:, 0, :, :]
            elif pred.ndim == 3:
                pred = pred[0:1, :, :]
        else:
            pred = pred[:, 0, :, :]

        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []

        batch_sz = pred.shape[0] if hasattr(pred, 'shape') else 1
        for batch_index in range(batch_sz):
            height, width = pred.shape[1], pred.shape[2]
            if is_output_polygon:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def binarize(self, pred):
        return pred > self.thresh

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        assert len(_bitmap.shape) == 2
        bitmap = _bitmap if isinstance(_bitmap, np.ndarray) else _bitmap.cpu().numpy()
        if not isinstance(pred, np.ndarray):
            pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        assert len(_bitmap.shape) == 2
        if isinstance(_bitmap, np.ndarray):
            bitmap = _bitmap
        else:
            bitmap = _bitmap.cpu().numpy()
        if not isinstance(pred, np.ndarray):
            pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < 2:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)

            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int64), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int64), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int64), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int64), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        if bitmap.dtype == np.float16:
            bitmap = bitmap.astype(np.float32)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


# ===========================================================================
# Text block examination & grouping (from comic-text-detector/utils/textblock.py)
# ===========================================================================
def examine_textblk(blk: TextBlock, im_w: int, im_h: int, sort: bool = False) -> None:
    """Determine text block orientation, font size, angle, etc."""
    lines = blk.lines_array()
    middle_pnts = (lines[:, [1, 2, 3, 0]] + lines) / 2
    vec_v = middle_pnts[:, 2] - middle_pnts[:, 0]   # vertical vectors
    vec_h = middle_pnts[:, 1] - middle_pnts[:, 3]   # horizontal vectors

    center_pnts = (lines[:, 0] + lines[:, 2]) / 2
    v = np.sum(vec_v, axis=0)
    h = np.sum(vec_h, axis=0)
    norm_v, norm_h = np.linalg.norm(v), np.linalg.norm(h)

    if blk.language == 'ja':
        vertical = norm_v > norm_h
    else:
        vertical = norm_v > norm_h * 2

    if vertical:
        primary_vec, primary_norm = v, norm_v
        distance_vectors = center_pnts - np.array([[im_w, 0]], dtype=np.float64)
        font_size = int(round(norm_h / len(lines)))
    else:
        primary_vec, primary_norm = h, norm_h
        distance_vectors = center_pnts - np.array([[0, 0]], dtype=np.float64)
        font_size = int(round(norm_v / len(lines)))

    rotation_angle = int(math.atan2(primary_vec[1], primary_vec[0]) / math.pi * 180)
    distance = np.linalg.norm(distance_vectors, axis=1)
    rad_matrix = np.arccos(
        np.clip(np.einsum('ij, j->i', distance_vectors, primary_vec) / (distance * primary_norm + 1e-10), -1, 1)
    )
    distance = np.abs(np.sin(rad_matrix) * distance)

    blk.lines = lines.astype(np.int32).tolist()
    blk.distance = distance
    blk.angle = rotation_angle
    if vertical:
        blk.angle -= 90
    if abs(blk.angle) < 3:
        blk.angle = 0
    blk.font_size = font_size
    blk.vertical = vertical
    blk.vec = primary_vec
    blk.norm = primary_norm
    if sort:
        blk.sort_lines()


def try_merge_textline(blk: TextBlock, blk2: TextBlock, fntsize_tol=1.3, distance_tol=2) -> bool:
    if blk2.merged:
        return False
    fntsize_div = blk.font_size / (blk2.font_size + 1e-10)
    num_l1, num_l2 = len(blk), len(blk2)
    fntsz_avg = (blk.font_size * num_l1 + blk2.font_size * num_l2) / (num_l1 + num_l2)
    vec_prod = blk.vec @ blk2.vec
    vec_sum = blk.vec + blk2.vec
    cos_vec = vec_prod / (blk.norm * blk2.norm + 1e-10)
    distance = blk2.distance[-1] - blk.distance[-1]
    distance_p1 = np.linalg.norm(np.array(blk2.lines[-1][0]) - np.array(blk.lines[-1][0]))

    l1, l2 = Polygon(blk.lines[-1]), Polygon(blk2.lines[-1])
    if not l1.intersects(l2):
        if fntsize_div > fntsize_tol or 1 / fntsize_div > fntsize_tol:
            return False
        if abs(cos_vec) < 0.866:  # cos30
            return False
        if distance > distance_tol * fntsz_avg or distance_p1 > fntsz_avg * 2.5:
            return False

    # merge
    blk.lines.append(blk2.lines[0])
    blk.vec = vec_sum
    blk.angle = int(round(np.rad2deg(math.atan2(vec_sum[1], vec_sum[0]))))
    if blk.vertical:
        blk.angle -= 90
    blk.norm = np.linalg.norm(vec_sum)
    blk.distance = np.append(blk.distance, blk2.distance[-1])
    blk.font_size = fntsz_avg
    blk2.merged = True
    return True


def merge_textlines(blk_list: List[TextBlock]) -> List[TextBlock]:
    if len(blk_list) < 2:
        return blk_list
    blk_list.sort(key=lambda blk: blk.distance[0])
    merged_list = []
    for ii, current_blk in enumerate(blk_list):
        if current_blk.merged:
            continue
        for jj, blk in enumerate(blk_list[ii + 1:]):
            try_merge_textline(current_blk, blk)
        merged_list.append(current_blk)
    for blk in merged_list:
        blk.adjust_bbox(with_bbox=False)
    return merged_list


def split_textblk(blk: TextBlock):
    font_size, distance, lines = blk.font_size, blk.distance, blk.lines
    l0 = np.array(blk.lines[0])
    lines.sort(key=lambda line: np.linalg.norm(np.array(line[0]) - l0[0]))
    distance_tol = font_size * 2
    current_blk = copy.deepcopy(blk)
    current_blk.lines = [l0.tolist() if isinstance(l0, np.ndarray) else l0]
    sub_blk_list = [current_blk]
    textblock_splitted = False

    for jj, line in enumerate(lines[1:]):
        l1_poly, l2_poly = Polygon(lines[jj]), Polygon(line)
        split = False
        if not l1_poly.intersects(l2_poly):
            line_distance = abs(distance[jj + 1] - distance[jj]) if distance is not None and len(distance) > jj + 1 else 0
            if line_distance > distance_tol:
                split = True
            elif blk.vertical and abs(blk.angle) < 15:
                if len(current_blk.lines) > 1 or line_distance > font_size:
                    split = abs(lines[jj][0][1] - line[0][1]) > font_size
        if split:
            current_blk = copy.deepcopy(current_blk)
            current_blk.lines = [line]
            sub_blk_list.append(current_blk)
        else:
            current_blk.lines.append(line)

    if len(sub_blk_list) > 1:
        textblock_splitted = True
        for current_blk in sub_blk_list:
            current_blk.adjust_bbox(with_bbox=False)
    return textblock_splitted, sub_blk_list


def sort_textblk_list(blk_list: List[TextBlock], im_w: int, im_h: int) -> List[TextBlock]:
    """Sort text blocks in manga reading order."""
    if len(blk_list) == 0:
        return blk_list
    num_ja = 0
    xyxy = []
    for blk in blk_list:
        if blk.language == 'ja':
            num_ja += 1
        xyxy.append(blk.xyxy)
    xyxy = np.array(xyxy)
    flip_lr = num_ja > len(blk_list) / 2

    im_oriw = im_w
    if im_w > im_h:
        im_w /= 2
    num_gridy, num_gridx = 4, 3
    img_area = im_h * im_w
    center_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
    if flip_lr:
        if im_w != im_oriw:
            center_x = im_oriw - center_x
        else:
            center_x = im_w - center_x
    grid_x = (center_x / im_w * num_gridx).astype(np.int32)
    center_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
    grid_y = (center_y / im_h * num_gridy).astype(np.int32)
    grid_indices = grid_y * num_gridx + grid_x
    grid_weights = grid_indices * img_area + \
        1.2 * (center_x - grid_x * im_w / num_gridx) + (center_y - grid_y * im_h / num_gridy)
    if im_w != im_oriw:
        grid_weights[np.where(grid_x >= num_gridx)] += img_area * num_gridy * num_gridx

    for blk, weight in zip(blk_list, grid_weights):
        blk.weight = weight
    blk_list.sort(key=lambda blk: blk.weight)
    return blk_list


def group_output(blks, lines, im_w, im_h, mask=None, sort_blklist=True) -> List[TextBlock]:
    """
    Assign detected text lines to text blocks, examine blocks,
    merge scattered lines, sort in reading order.
    """
    blk_list: List[TextBlock] = []
    scattered_lines = {'ver': [], 'hor': []}

    for bbox, cls, conf in zip(*blks):
        blk_list.append(TextBlock(bbox, language=LANG_LIST[cls]))

    # step1: filter & assign lines to textblocks
    bbox_score_thresh = 0.4
    mask_score_thresh = 0.1
    for ii, line in enumerate(lines):
        bx1, bx2 = line[:, 0].min(), line[:, 0].max()
        by1, by2 = line[:, 1].min(), line[:, 1].max()
        bbox_score, bbox_idx = -1, -1
        line_area = (by2 - by1) * (bx2 - bx1)
        if line_area <= 0:
            continue
        for jj, blk in enumerate(blk_list):
            score = union_area(blk.xyxy, [bx1, by1, bx2, by2]) / line_area
            if bbox_score < score:
                bbox_score = score
                bbox_idx = jj
        if bbox_score > bbox_score_thresh:
            blk_list[bbox_idx].lines.append(line)
        else:
            # if no textblock was assigned, check text mask
            if mask is not None:
                by1_i, by2_i = int(max(0, by1)), int(min(mask.shape[0], by2))
                bx1_i, bx2_i = int(max(0, bx1)), int(min(mask.shape[1], bx2))
                mask_score = mask[by1_i:by2_i, bx1_i:bx2_i].mean() / 255
                if mask_score < mask_score_thresh:
                    continue
            blk = TextBlock([bx1, by1, bx2, by2], [line])
            examine_textblk(blk, im_w, im_h, sort=False)
            if blk.vertical:
                scattered_lines['ver'].append(blk)
            else:
                scattered_lines['hor'].append(blk)

    # step2: filter textblocks, sort & split textlines
    final_blk_list = []
    for blk in blk_list:
        if len(blk.lines) == 0:
            bx1, by1, bx2, by2 = blk.xyxy
            if mask is not None:
                by1_i, by2_i = int(max(0, by1)), int(min(mask.shape[0], by2))
                bx1_i, bx2_i = int(max(0, bx1)), int(min(mask.shape[1], bx2))
                mask_score = mask[by1_i:by2_i, bx1_i:bx2_i].mean() / 255
                if mask_score < mask_score_thresh:
                    continue
            xywh = np.array([[bx1, by1, bx2 - bx1, by2 - by1]])
            blk.lines = xywh2xyxypoly(xywh).reshape(-1, 4, 2).tolist()
        examine_textblk(blk, im_w, im_h, sort=True)

        # split manga text if there is a distance gap
        textblock_splitted = False
        if len(blk.lines) > 1:
            if blk.language == 'ja':
                textblock_splitted = True
            elif blk.vertical:
                textblock_splitted = True
        if textblock_splitted:
            textblock_splitted, sub_blk_list = split_textblk(blk)
        else:
            sub_blk_list = [blk]

        # modify textblock to fit its textlines
        if not textblock_splitted:
            for blk in sub_blk_list:
                blk.adjust_bbox(with_bbox=True)
        final_blk_list += sub_blk_list

    # step3: merge scattered lines, sort textblocks by "grid"
    final_blk_list += merge_textlines(scattered_lines['hor'])
    final_blk_list += merge_textlines(scattered_lines['ver'])
    if sort_blklist:
        final_blk_list = sort_textblk_list(final_blk_list, im_w, im_h)

    # Expand English text lines slightly
    for blk in final_blk_list:
        if blk.language == 'eng' and not blk.vertical:
            num_lines = len(blk.lines)
            if num_lines == 0:
                continue
            expand_size = max(int(blk.font_size * 0.1), 2)
            rad = np.deg2rad(blk.angle)
            shifted_vec = np.array([[[-1, -1], [1, -1], [1, 1], [-1, 1]]])
            shifted_vec = shifted_vec * np.array([[[np.sin(rad), np.cos(rad)]]]) * expand_size
            lines_arr = blk.lines_array() + shifted_vec
            lines_arr[..., 0] = np.clip(lines_arr[..., 0], 0, im_w - 1)
            lines_arr[..., 1] = np.clip(lines_arr[..., 1], 0, im_h - 1)
            blk.lines = lines_arr.astype(np.int64).tolist()
            blk.font_size += expand_size

    return final_blk_list


# ===========================================================================
# Mask refinement (from comic-text-detector/utils/textmask.py)
# ===========================================================================
def get_topk_color(color_list, bins, k=3, color_var=10, bin_tol=0.001):
    idx = np.argsort(bins * -1)
    color_list, bins = color_list[idx], bins[idx]
    top_colors = [color_list[0]]
    bin_tol_val = np.sum(bins) * bin_tol
    if len(color_list) > 1:
        for color, b in zip(color_list[1:], bins[1:]):
            if np.abs(np.array(top_colors) - color).min() > color_var:
                top_colors.append(color)
            if len(top_colors) >= k or b < bin_tol_val:
                break
    return top_colors


def minxor_thresh(threshed, mask, dilate=False):
    neg_threshed = 255 - threshed
    e_size = 1
    if dilate:
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * e_size + 1, 2 * e_size + 1), (e_size, e_size))
        neg_threshed = cv2.dilate(neg_threshed, element, iterations=1)
        threshed = cv2.dilate(threshed, element, iterations=1)
    neg_xor_sum = cv2.bitwise_xor(neg_threshed, mask).sum()
    xor_sum = cv2.bitwise_xor(threshed, mask).sum()
    if neg_xor_sum < xor_sum:
        return neg_threshed, neg_xor_sum
    else:
        return threshed, xor_sum


def get_otsuthresh_masklist(img, pred_mask, per_channel=False):
    channels = [img[..., 0], img[..., 1], img[..., 2]]
    mask_list = []
    for c in channels:
        _, threshed = cv2.threshold(c, 1, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        threshed, xor_sum = minxor_thresh(threshed, pred_mask, dilate=False)
        mask_list.append([threshed, xor_sum])
    mask_list.sort(key=lambda x: x[1])
    if per_channel:
        return mask_list
    else:
        return [mask_list[0]]


def get_topk_masklist(im_grey, pred_mask):
    if len(im_grey.shape) == 3 and im_grey.shape[-1] == 3:
        im_grey = cv2.cvtColor(im_grey, cv2.COLOR_BGR2GRAY)
    msk = np.ascontiguousarray(pred_mask)
    eroded = cv2.erode(msk, np.ones((3, 3), np.uint8), iterations=1)
    candidate_grey_px = im_grey[np.where(eroded > 127)]
    if len(candidate_grey_px) == 0:
        return []
    bins_arr, his = np.histogram(candidate_grey_px, bins=255)
    topk_color = get_topk_color(his[:-1].astype(np.float64), bins_arr.astype(np.float64), color_var=10, k=3)
    color_range = 30
    mask_list = []
    for color in topk_color:
        c_top = min(int(color) + color_range, 255)
        c_bottom = c_top - 2 * color_range
        threshed = cv2.inRange(im_grey, c_bottom, c_top)
        threshed, xor_sum = minxor_thresh(threshed, msk)
        mask_list.append([threshed, xor_sum])
    return mask_list


def merge_mask_list(mask_list, pred_mask, blk=None, text_window=None,
                    filter_with_lines=False, refine_mode=REFINEMASK_INPAINT, pred_thresh=30):
    mask_list.sort(key=lambda x: x[1])
    linemask = None
    if blk is not None and filter_with_lines:
        linemask = np.zeros_like(pred_mask)
        lines_arr = blk.lines_array(dtype=np.int64)
        for line in lines_arr:
            line[..., 0] -= text_window[0]
            line[..., 1] -= text_window[1]
            cv2.fillPoly(linemask, [line], 255)
        linemask = cv2.dilate(linemask, np.ones((3, 3), np.uint8), iterations=3)

    if pred_thresh > 0:
        e_size = 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e_size + 1, 2 * e_size + 1), (e_size, e_size))
        pred_mask = cv2.erode(pred_mask, element, iterations=1)
        _, pred_mask = cv2.threshold(pred_mask, 60, 255, cv2.THRESH_BINARY)

    connectivity = 8
    mask_merged = np.zeros_like(pred_mask)
    for ii, (candidate_mask, xor_sum) in enumerate(mask_list):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_mask, connectivity, cv2.CV_16U)
        for label_index, stat, centroid in zip(range(num_labels), stats, centroids):
            if label_index != 0:
                x, y, w, h, area = stat
                if w * h < 3:
                    continue
                x1, y1, x2, y2 = x, y, x + w, y + h
                label_local = labels[y1:y2, x1:x2]
                label_coordinates = np.where(label_local == label_index)
                tmp_merged = np.zeros_like(label_local, np.uint8)
                tmp_merged[label_coordinates] = 255
                tmp_merged = cv2.bitwise_or(mask_merged[y1:y2, x1:x2], tmp_merged)
                xor_merged = cv2.bitwise_xor(tmp_merged, pred_mask[y1:y2, x1:x2]).sum()
                xor_origin = cv2.bitwise_xor(mask_merged[y1:y2, x1:x2], pred_mask[y1:y2, x1:x2]).sum()
                if xor_merged < xor_origin:
                    mask_merged[y1:y2, x1:x2] = tmp_merged

    if refine_mode == REFINEMASK_INPAINT:
        mask_merged = cv2.dilate(mask_merged, np.ones((3, 3), np.uint8), iterations=1)

    # fill holes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - mask_merged, connectivity, cv2.CV_16U)
    sorted_area = np.sort(stats[:, -1])
    if len(sorted_area) > 1:
        area_thresh = sorted_area[-2]
    else:
        area_thresh = sorted_area[-1]
    for label_index, stat, centroid in zip(range(num_labels), stats, centroids):
        x, y, w, h, area = stat
        if area < area_thresh:
            x1, y1, x2, y2 = x, y, x + w, y + h
            label_local = labels[y1:y2, x1:x2]
            label_coordinates = np.where(label_local == label_index)
            tmp_merged = np.zeros_like(label_local, np.uint8)
            tmp_merged[label_coordinates] = 255
            tmp_merged = cv2.bitwise_or(mask_merged[y1:y2, x1:x2], tmp_merged)
            xor_merged = cv2.bitwise_xor(tmp_merged, pred_mask[y1:y2, x1:x2]).sum()
            xor_origin = cv2.bitwise_xor(mask_merged[y1:y2, x1:x2], pred_mask[y1:y2, x1:x2]).sum()
            if xor_merged < xor_origin:
                mask_merged[y1:y2, x1:x2] = tmp_merged
    return mask_merged


def refine_mask(img: np.ndarray, pred_mask: np.ndarray, blk_list: List[TextBlock],
                refine_mode: int = REFINEMASK_INPAINT) -> np.ndarray:
    mask_refined = np.zeros_like(pred_mask)
    for blk in blk_list:
        bx1, by1, bx2, by2 = expand_textwindow(img.shape, blk.xyxy, expand_r=16)
        im = np.ascontiguousarray(img[by1:by2, bx1:bx2])
        msk = np.ascontiguousarray(pred_mask[by1:by2, bx1:bx2])
        if im.size == 0 or msk.size == 0:
            continue
        mask_list = get_topk_masklist(im, msk)
        mask_list += get_otsuthresh_masklist(im, msk, per_channel=False)
        if not mask_list:
            continue
        mask_merged = merge_mask_list(mask_list, msk, blk=blk, text_window=[bx1, by1, bx2, by2],
                                      refine_mode=refine_mode)
        mask_refined[by1:by2, bx1:bx2] = cv2.bitwise_or(mask_refined[by1:by2, bx1:bx2], mask_merged)
    return mask_refined


def refine_undetected_mask(img: np.ndarray, mask_pred: np.ndarray, mask_refined: np.ndarray,
                           blk_list: List[TextBlock], refine_mode=REFINEMASK_INPAINT):
    mask_pred[np.where(mask_refined > 30)] = 0
    _, pred_mask_t = cv2.threshold(mask_pred, 30, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask_t, 4, cv2.CV_16U)
    valid_labels = np.where(stats[:, -1] > 50)[0]
    seg_blk_list = []
    if len(valid_labels) > 0:
        for lab_index in valid_labels[1:]:
            x, y, w, h, area = stats[lab_index]
            bx1, by1 = x, y
            bx2, by2 = x + w, y + h
            bbox = [bx1, by1, bx2, by2]
            bbox_score = -1
            for blk in blk_list:
                bbox_s = union_area(blk.xyxy, bbox)
                if bbox_s > bbox_score:
                    bbox_score = bbox_s
            if bbox_score / (w * h + 1e-10) < 0.5:
                seg_blk_list.append(TextBlock(bbox))
    if len(seg_blk_list) > 0:
        mask_refined = cv2.bitwise_or(mask_refined, refine_mask(img, mask_pred, seg_blk_list, refine_mode=refine_mode))
    return mask_refined


# ===========================================================================
# Preprocessing (from comic-text-detector/inference.py)
# ===========================================================================
def _ensure_model() -> None:
    """Download the ONNX model if it is not present on disk."""
    if _MODEL_PATH.exists():
        return
    import urllib.request
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading comic-text-detector model (%s) …", _MODEL_URL)
    urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
    logger.info("Model saved to %s", _MODEL_PATH)


def _get_net():
    """Return the OpenCV DNN network (loaded once)."""
    global _net
    if _net is None:
        _ensure_model()
        _net = cv2.dnn.readNetFromONNX(str(_MODEL_PATH))
        logger.info("comic-text-detector ONNX model loaded.")
    return _net


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0), auto=False,
              scaleFill=False, scaleup=True, stride=128):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]  # current shape [height, width]
    if not isinstance(new_shape, tuple):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dh, dw = int(dh), int(dw)
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def preprocess_img(img, input_size=(1024, 1024), bgr2rgb=True):
    """Preprocess image for model input (matching comic-text-detector)."""
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox(img, new_shape=input_size, auto=False, stride=64)
    return img_in, ratio, int(dw), int(dh)


# ===========================================================================
# Post-processing (from comic-text-detector/inference.py)
# ===========================================================================
def postprocess_mask(img: np.ndarray, thresh=None):
    """Post-process segmentation mask output."""
    img = img.squeeze()
    if thresh is not None:
        img = img > thresh
    img = img * 255
    return img.astype(np.uint8)


def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def postprocess_yolo(det, conf_thresh, nms_thresh, resize_ratio, sort_func=None):
    """YOLOv5-style NMS post-processing for block detections."""
    # det shape: (1, num_preds, 7) or (num_preds, 7)
    if det.ndim == 3:
        det = det[0]

    # Filter by object confidence
    obj_mask = det[:, 4] > conf_thresh
    det = det[obj_mask]
    if len(det) == 0:
        return np.empty((0, 4), dtype=np.int32), np.array([], dtype=np.int32), np.array([])

    # conf = obj_conf * cls_conf
    det[:, 5:] *= det[:, 4:5]
    boxes = _xywh2xyxy(det[:, :4])
    conf = det[:, 5:].max(axis=1)
    cls = det[:, 5:].argmax(axis=1)

    valid = conf > conf_thresh
    boxes, conf, cls = boxes[valid], conf[valid], cls[valid]
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.int32), np.array([], dtype=np.int32), np.array([])

    # NMS via OpenCV
    boxes_xywh = [[float(b[0]), float(b[1]),
                    float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, conf.tolist(), conf_thresh, nms_thresh)
    if len(indices) == 0:
        return np.empty((0, 4), dtype=np.int32), np.array([], dtype=np.int32), np.array([])
    indices = indices.flatten()

    det_result = np.column_stack([boxes[indices], conf[indices, None], cls[indices, None]])

    if sort_func is not None:
        det_result = sort_func(det_result)

    # Rescale to original image coordinates
    det_result[..., [0, 2]] = det_result[..., [0, 2]] * resize_ratio[0]
    det_result[..., [1, 3]] = det_result[..., [1, 3]] * resize_ratio[1]

    blines = det_result[..., 0:4].astype(np.int32)
    confs = np.round(det_result[..., 4], 3)
    cls_out = det_result[..., 5].astype(np.int32)
    return blines, cls_out, confs


# ===========================================================================
# Main TextDetector class (from comic-text-detector/inference.py)
# ===========================================================================
class TextDetector:
    lang_list = ['eng', 'ja', 'unknown']
    langcls2idx = {'eng': 0, 'ja': 1, 'unknown': 2}

    def __init__(self, model_path=None, input_size=1024, nms_thresh=0.35,
                 conf_thresh=0.4, mask_thresh=0.3, act='leaky'):
        if model_path is None:
            model_path = str(_MODEL_PATH)
        _ensure_model()

        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.uoln = self.net.getUnconnectedOutLayersNames()
        self.backend = 'opencv'

        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)

    def __call__(self, img, refine_mode=REFINEMASK_INPAINT, keep_undetected_mask=False, options: dict | None = None):
        """
        Run the full comic-text-detector pipeline on an image.

        Args:
            img: BGR image (numpy array)
            refine_mode: REFINEMASK_INPAINT or REFINEMASK_ANNOTATION
            keep_undetected_mask: whether to refine undetected mask regions

        Returns:
            (mask, mask_refined, blk_list)
        """
        options = options or {}
        conf_thresh = float(options.get("confidence_threshold", self.conf_thresh))
        nms_thresh = float(options.get("nms_threshold", self.nms_thresh))
        mask_thresh = float(options.get("mask_threshold", _MASK_THRESH))
        box_thresh = float(options.get("box_threshold", _BOX_THRESH))
        seg_rep = self.seg_rep if mask_thresh == _MASK_THRESH else SegDetectorRepresenter(thresh=mask_thresh)
        img_in, ratio, dw, dh = preprocess_img(img, input_size=self.input_size)
        im_h, im_w = img.shape[:2]

        # Run DNN inference
        blob = cv2.dnn.blobFromImage(img_in, scalefactor=1 / 255.0,
                                     size=(self.input_size[0], self.input_size[1]))
        self.net.setInput(blob)
        blks, mask, lines_map = self.net.forward(self.uoln)

        # Resize ratio for mapping back to original image
        resize_ratio = (im_w / (self.input_size[0] - dw), im_h / (self.input_size[1] - dh))

        # Post-process block detections
        blks = postprocess_yolo(blks, conf_thresh, nms_thresh, resize_ratio)

        # Handle possible reversed outputs in some OpenCV versions
        if mask.shape[1] == 2:
            tmp = mask
            mask = lines_map
            lines_map = tmp

        # Post-process mask
        mask = postprocess_mask(mask)

        # Post-process text lines via DBNet SegDetectorRepresenter
        lines, scores = seg_rep(self.input_size, lines_map)
        idx = np.where(scores[0] > box_thresh)
        lines, scores = lines[0][idx], scores[0][idx]

        # Map outputs to input image coordinates
        mask = mask[:mask.shape[0] - dh, :mask.shape[1] - dw]
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)

        if lines.size == 0:
            lines = []
        else:
            lines = lines.astype(np.float64)
            lines[..., 0] *= resize_ratio[0]
            lines[..., 1] *= resize_ratio[1]
            lines = lines.astype(np.int32)

        # Group outputs: assign lines to blocks
        blk_list = group_output(blks, lines, im_w, im_h, mask)

        # Refine mask
        mask_refined = refine_mask(img, mask, blk_list, refine_mode=refine_mode)
        if keep_undetected_mask:
            mask_refined = refine_undetected_mask(img, mask, mask_refined, blk_list, refine_mode=refine_mode)

        return mask, mask_refined, blk_list


# ===========================================================================
# Public API (compatible with the rest of the project)
# ===========================================================================
_detector = None


def _get_detector() -> TextDetector:
    """Return the singleton TextDetector instance."""
    global _detector
    if _detector is None:
        _detector = TextDetector(
            model_path=str(_MODEL_PATH),
            input_size=_INPUT_SIZE,
            nms_thresh=_NMS_THRESH,
            conf_thresh=_CONF_THRESH,
            mask_thresh=_MASK_THRESH,
        )
        logger.info("TextDetector initialized (comic-text-detector pipeline).")
    return _detector


def detect_text_regions(image_path: str, max_regions: int | None = None, options: dict | None = None) -> list[dict]:
    """
    Detect text regions in a manga/comic image using the full
    comic-text-detector pipeline.

    Pipeline (1:1 from https://github.com/dmMaze/comic-text-detector):
      1. Preprocess: letterbox resize to 1024x1024
      2. Inference: ONNX model (YOLOv5 blocks + UNet mask + DBNet lines)
      3. Post-process: NMS for blocks, SegDetectorRepresenter for lines
      4. Group: assign lines to blocks, examine orientation/font/angle
      5. Merge scattered lines, split large blocks
      6. Refine mask per text block
      7. Sort in manga reading order

    Args:
        image_path: Path to the manga panel image.
        max_regions: If set, return only the top N regions by confidence.
        options: Optional detector thresholds for this request.

    Returns:
        List of dicts with keys: x, y, width, height (pixel coords).
        Sorted in manga reading order (right-to-left, top-to-bottom).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    options = options or {}
    detector_options = dict(options.get("detection") or {})
    for key in ("confidence_threshold", "nms_threshold", "mask_threshold", "box_threshold", "max_regions"):
        if key in options and options[key] is not None:
            detector_options[key] = options[key]
    max_regions = max_regions or detector_options.get("max_regions")

    detector = _get_detector()
    mask, mask_refined, blk_list = detector(
        img,
        refine_mode=REFINEMASK_INPAINT,
        keep_undetected_mask=True,
        options=detector_options,
    )

    # Convert TextBlock list to region dicts
    regions = []
    for blk in blk_list:
        x1, y1, x2, y2 = blk.xyxy
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img.shape[1], int(x2))
        y2 = min(img.shape[0], int(y2))
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue

        # Convert lines to plain Python lists (no numpy types)
        lines_clean = []
        for line in blk.lines:
            if isinstance(line, np.ndarray):
                lines_clean.append(line.tolist())
            else:
                lines_clean.append([[int(p[0]), int(p[1])] for p in line])

        regions.append({
            "x": int(x1),
            "y": int(y1),
            "width": int(w),
            "height": int(h),
            "vertical": bool(blk.vertical),
            "language": str(blk.language),
            "font_size": int(blk.font_size) if blk.font_size >= 0 else 0,
            "angle": int(blk.angle),
            "lines": lines_clean,
        })

    if max_regions and len(regions) > max_regions:
        regions = regions[:max_regions]

    return regions


def detect_text_regions_full(image_path: str):
    """
    Full detection returning mask, refined mask, and TextBlock list.

    Returns:
        (mask, mask_refined, blk_list) - raw outputs from the detector
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None, []

    detector = _get_detector()
    return detector(img, refine_mode=REFINEMASK_INPAINT, keep_undetected_mask=True)
