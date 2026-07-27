"""手-货框深度关系特征，读 scripts/precompute_depth.py 的缓存。

单目深度是相对逆深度（越大越近）且逐帧尺度漂移，所以只做**同帧内**的差值，
再用该帧的深度跨度归一化。`depth_gap_norm > 0` 表示手腕比货框本体更靠近相机
——人站在货架前、手悬在框外的典型形态。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEPTH_FEATURE_KEYS = [
    "depth_gap_norm",
    "depth_gap_full_norm",
    "depth_box_ex_valid",
]

_EPS = 1e-6


def load_depth_cache(depth_dir: Path, record_id: str) -> dict[tuple[int, int, str], dict[str, Any]]:
    """→ {(frame_idx, person_track_id, token): 深度标量行}"""
    path = Path(depth_dir) / f"{record_id.replace('/', '__')}.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in data.get("rows") or []:
        key = (
            int(row.get("frame_idx") or 0),
            int(row.get("person_track_id") or 0),
            str(row.get("token") or ""),
        )
        out[key] = row
    return out


def compute_depth_features(row: dict[str, Any] | None) -> dict[str, float | None]:
    if not row:
        return {k: None for k in DEPTH_FEATURE_KEYS}
    span = float(row.get("p95") or 0.0) - float(row.get("p5") or 0.0)
    if abs(span) < _EPS:
        return {k: None for k in DEPTH_FEATURE_KEYS}
    d_wrist = float(row.get("d_wrist"))
    return {
        "depth_gap_norm": (d_wrist - float(row.get("d_box_ex"))) / span,
        "depth_gap_full_norm": (d_wrist - float(row.get("d_box_full"))) / span,
        "depth_box_ex_valid": float(row.get("box_ex_valid") or 0.0),
    }
