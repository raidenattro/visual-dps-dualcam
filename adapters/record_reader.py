"""只读读取 collector record：skeleton.parquet + manifest.json 货框。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from adapters.collector_paths import CollectorPaths, load_baseline_manifest, load_paths

KPT_COUNT = 17


@dataclass
class RecordRef:
    record_id: str
    clip_name: str
    camera_slug: str
    file_name: str
    infer_width: int
    infer_height: int

    @property
    def rel_dir(self) -> str:
        return self.record_id


@dataclass
class BoxDef:
    box_id: str
    token: str
    contour: Any
    center: tuple[float, float]
    inradius: float
    layer: int | None = None
    column: int | None = None


@dataclass
class RecordData:
    ref: RecordRef
    frames: list[dict[str, Any]]
    boxes: list[BoxDef]
    fps: float = 15.0
    meta: dict[str, Any] = field(default_factory=dict)


def list_records(paths: CollectorPaths | None = None) -> list[RecordRef]:
    p = paths or load_paths()
    manifest = load_baseline_manifest(p)
    out: list[RecordRef] = []
    for item in manifest.get("records") or []:
        if not isinstance(item, dict) or item.get("status") != "ok":
            continue
        out.append(
            RecordRef(
                record_id=str(item.get("record_id") or ""),
                clip_name=str(item.get("clip_name") or ""),
                camera_slug=str(item.get("camera_slug") or ""),
                file_name=str(item.get("file") or ""),
                infer_width=int(float(item.get("infer_width") or 0)),
                infer_height=int(float(item.get("infer_height") or 0)),
            )
        )
    return out


def list_records_from_manifest(
    manifest_path: Path | str,
    *,
    split_role: str | None = None,
) -> list[RecordRef]:
    """读本仓 tagged manifest。

    split_role:
      - None: 全部 record
      - "val": 含至少一段 val 的 record（导出/评估用）
      - "train": 含至少一段 train 的 record
    """
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    out: list[RecordRef] = []
    for item in data.get("records") or []:
        if not isinstance(item, dict):
            continue
        n_val = int(item.get("n_val_segments") or 0)
        n_train = int(item.get("n_train_segments") or 0)
        if split_role == "val" and n_val <= 0:
            continue
        if split_role == "train" and n_train <= 0:
            continue
        fn = str(item.get("file") or "")
        if not fn.endswith(".json"):
            fn = f"{item.get('clip_name') or 'clip'}.json"
        out.append(
            RecordRef(
                record_id=str(item.get("record_id") or ""),
                clip_name=str(item.get("clip_name") or ""),
                camera_slug=str(item.get("camera_slug") or ""),
                file_name=fn,
                infer_width=int(float(item.get("infer_width") or 0)),
                infer_height=int(float(item.get("infer_height") or 0)),
            )
        )
    return out


def review_key_map_from_manifest(manifest_path: Path | str) -> dict[str, str]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for item in data.get("records") or []:
        rid = str(item.get("record_id") or "")
        rk = str(item.get("review_key") or "")
        if rid and rk:
            out[rid] = rk
    return out


def load_segment_split(manifest_path: Path | str) -> dict[str, list[dict[str, Any]]]:
    """record_id -> segments（含 split/gt_tokens/tracks/frames）。"""
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    by: dict[str, list[dict[str, Any]]] = {}
    for seg in data.get("segments") or []:
        rid = str(seg.get("record_id") or "")
        by.setdefault(rid, []).append(seg)
    return by


def record_dir(ref: RecordRef, paths: CollectorPaths | None = None) -> Path:
    p = paths or load_paths()
    return p.json_dir / ref.record_id


def _polygon_center_and_inradius(pts: np.ndarray) -> tuple[tuple[float, float], float]:
    """中心取顶点均值；内切半径用中心到各边的最小距离近似。"""
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    n = len(pts)
    dists: list[float] = []
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        seg_len = math.hypot(ex, ey)
        if seg_len < 1e-6:
            continue
        dists.append(abs(ex * (y1 - cy) - (x1 - cx) * ey) / seg_len)
    return (cx, cy), (min(dists) if dists else 1.0)


def load_boxes(rec_dir: Path) -> list[BoxDef]:
    manifest = json.loads((rec_dir / "manifest.json").read_text(encoding="utf-8"))
    ann = manifest.get("annotation") or {}
    raw_boxes = list(ann.get("boxes") or [])
    if not raw_boxes:
        for shelf in ann.get("shelves") or []:
            raw_boxes.extend(shelf.get("boxes") or [])

    out: list[BoxDef] = []
    for b in raw_boxes:
        poly = b.get("video_polygon")
        box_id = str(b.get("box_id") or "").strip()
        if not poly or not box_id:
            continue
        pts = np.array(poly, dtype=np.float64)
        center, inradius = _polygon_center_and_inradius(pts)
        out.append(
            BoxDef(
                box_id=box_id,
                token=f"Box_{box_id}",
                contour=np.int32(pts).reshape((-1, 1, 2)),
                center=center,
                inradius=max(1.0, inradius),
                layer=b.get("layer"),
                column=b.get("column"),
            )
        )
    return out


def load_skeleton_frames(rec_dir: Path) -> list[dict[str, Any]]:
    """skeleton.parquet → [{frame_idx, timestamp_sec, persons:[{keypoints, ...}]}]，按帧聚合。"""
    table = pq.read_table(rec_dir / "skeleton.parquet")
    cols = {name: table.column(name).to_pylist() for name in table.schema.names}
    n = table.num_rows

    by_frame: dict[int, dict[str, Any]] = {}
    for i in range(n):
        frame_idx = int(cols["frame_idx"][i] or 0)
        frame = by_frame.get(frame_idx)
        if frame is None:
            frame = {
                "frame_idx": frame_idx,
                "source_frame_idx": int(cols.get("source_frame_idx", [frame_idx] * n)[i] or frame_idx),
                "timestamp_sec": float(cols.get("timestamp_sec", [0.0] * n)[i] or 0.0),
                "persons": [],
            }
            by_frame[frame_idx] = frame

        keypoints: list[list[float]] = []
        for k in range(KPT_COUNT):
            x = cols.get(f"kpt_{k}_x", [None] * n)[i]
            y = cols.get(f"kpt_{k}_y", [None] * n)[i]
            s = cols.get(f"kpt_{k}_score", [None] * n)[i]
            if x is None or y is None:
                keypoints.append([0.0, 0.0, 0.0])
            else:
                keypoints.append([float(x), float(y), float(s or 0.0)])

        frame["persons"].append(
            {
                "person_id": int(cols.get("person_id", [0] * n)[i] or 0),
                "person_track_id": int(cols.get("person_track_id", [0] * n)[i] or 0),
                "keypoints": keypoints,
            }
        )

    return [by_frame[k] for k in sorted(by_frame)]


def load_record(ref: RecordRef, paths: CollectorPaths | None = None) -> RecordData:
    rec_dir = record_dir(ref, paths)
    manifest = json.loads((rec_dir / "manifest.json").read_text(encoding="utf-8"))
    return RecordData(
        ref=ref,
        frames=load_skeleton_frames(rec_dir),
        boxes=load_boxes(rec_dir),
        fps=float(manifest.get("fps") or 15.0),
        meta={
            "infer_width": int(manifest.get("infer_width") or ref.infer_width or 0),
            "infer_height": int(manifest.get("infer_height") or ref.infer_height or 0),
            "annotation_source_file": (manifest.get("annotation") or {}).get("source_file"),
        },
    )
