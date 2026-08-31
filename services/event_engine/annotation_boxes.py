"""从标注 JSON 加载并缩放到推理分辨率下的货框多边形。"""

from __future__ import annotations

import json
import os

import numpy as np

from services.annotation_service import flatten_annotation_boxes


def _scale_polygon_points(points, sx: float, sy: float):
    out = []
    for pt in points:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            out.append([float(pt[0]) * sx, float(pt[1]) * sy])
    return out


def _polygon_max_extent(points) -> tuple[float, float]:
    max_x = 0.0
    max_y = 0.0
    for pt in points:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            max_x = max(max_x, float(pt[0]))
            max_y = max(max_y, float(pt[1]))
    return max_x, max_y


def _norm_polygon_valid(norm_pts) -> bool:
    if not isinstance(norm_pts, list) or len(norm_pts) < 3:
        return False
    for pt in norm_pts:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        x = float(pt[0])
        y = float(pt[1])
        if x < -0.01 or x > 1.01 or y < -0.01 or y > 1.01:
            return False
    return True


def _scale_polygon_to_frame(pts, norm_pts, ann_w, ann_h, target_w: int, target_h: int):
    tw = float(target_w)
    th = float(target_h)
    if _norm_polygon_valid(norm_pts):
        return _scale_polygon_points(norm_pts, tw, th)

    if not isinstance(pts, list) or len(pts) < 3:
        return []

    max_x, max_y = _polygon_max_extent(pts)
    if ann_w and ann_h and ann_w > 0 and ann_h > 0:
        sx = tw / float(ann_w) if max_x <= float(ann_w) * 1.05 else tw / max_x
        sy = th / float(ann_h) if max_y <= float(ann_h) * 1.05 else th / max_y
    elif max_x > 0 and max_y > 0:
        sx = tw / max_x
        sy = th / max_y
    else:
        sx = sy = 1.0
    return _scale_polygon_points(pts, sx, sy)


def build_scaled_boxes(raw_boxes, ann_w: float | None, ann_h: float | None, target_w: int, target_h: int):
    scaled = []
    for box in raw_boxes:
        pts = box.get("video_polygon", [])
        norm_pts = box.get("video_polygon_norm", [])
        mapped_pts = _scale_polygon_to_frame(pts, norm_pts, ann_w, ann_h, target_w, target_h)

        if len(mapped_pts) < 3:
            continue

        new_box = dict(box)
        new_box["video_polygon"] = mapped_pts
        new_box["orig_contour"] = np.int32(mapped_pts).reshape((-1, 1, 2))
        scaled.append(new_box)

    return scaled


def load_scaled_boxes(json_path: str, infer_w: int, infer_h: int) -> list:
    """读取标注 JSON 并返回推理坐标系下的 boxes（含 orig_contour）。"""
    if not json_path or not os.path.isfile(json_path):
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    raw_boxes = flatten_annotation_boxes(config_data)
    annotation_size = config_data.get("annotation_size", {}) if isinstance(config_data, dict) else {}
    if not isinstance(annotation_size, dict):
        annotation_size = {}

    ann_w = annotation_size.get("width")
    ann_h = annotation_size.get("height")
    try:
        ann_w = float(ann_w) if ann_w is not None else None
        ann_h = float(ann_h) if ann_h is not None else None
    except Exception:
        ann_w, ann_h = None, None

    return build_scaled_boxes(raw_boxes, ann_w, ann_h, infer_w, infer_h)
