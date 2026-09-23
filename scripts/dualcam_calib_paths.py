#!/usr/bin/env python3
"""双路 144/24 标定文件路径（与 1-3 对向基准分离）。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = ROOT / "output" / "calib"
SKEL_DIR = ROOT / "output" / "dualcam"
CALIB_144_24 = CALIB_DIR / "dual_144-24.json"
SKEL_144_24 = SKEL_DIR / "skel3d_144_24.json"
CALIB_1_3 = CALIB_DIR / "dual_1-3.json"
CALIB_ID = "dual_144-24"
DEFAULT_VIDEO_SRC = "output/dualcam/src.mp4"


def _safe_stem(path: str) -> str:
    stem = Path(str(path or "video")).stem
    chars = [c if c.isalnum() or c in "-_." else "_" for c in stem]
    name = "".join(chars).strip("._") or "video"
    return name[:80]


def calib_file(video_sources: dict | None) -> Path:
    """拼接片 src.mp4 仍用 dual_144-24.json；其它视频各自一份，避免覆盖原标定。"""
    vs = video_sources or {}
    mode = str(vs.get("mode") or "stitched")
    if mode == "split":
        stem = f"{_safe_stem(vs.get('L') or 'L')}__{_safe_stem(vs.get('R') or 'R')}"
        return CALIB_DIR / f"dual_{stem}.json"
    src = str(vs.get("src") or DEFAULT_VIDEO_SRC)
    if src == DEFAULT_VIDEO_SRC:
        return CALIB_144_24
    return CALIB_DIR / f"dual_{_safe_stem(src)}.json"


def skel_file(video_sources: dict | None) -> Path:
    """与 calib_file 同一套视频名：src.mp4 → skel3d_144_24.json，其它 → skel3d_<stem>.json。"""
    vs = video_sources or {}
    mode = str(vs.get("mode") or "stitched")
    if mode == "split":
        stem = f"{_safe_stem(vs.get('L') or 'L')}__{_safe_stem(vs.get('R') or 'R')}"
        return SKEL_DIR / f"skel3d_{stem}.json"
    src = str(vs.get("src") or DEFAULT_VIDEO_SRC)
    if src == DEFAULT_VIDEO_SRC:
        return SKEL_144_24
    return SKEL_DIR / f"skel3d_{_safe_stem(src)}.json"


def is_144_24_payload(data: dict) -> bool:
    if not data:
        return False
    if data.get("calib_id") == CALIB_ID:
        return True
    vs = data.get("video_sources") or {}
    if vs.get("L") == "data/test-144.mp4" and vs.get("R") == "data/test-24.mp4":
        return True
    if vs.get("mode") == "stitched" and "144" in str(vs.get("L") or ""):
        return True
    layout = data.get("layout") or data.get("stereo_layout")
    if layout in ("same_side", "same") and isinstance(data.get("views"), dict):
        if "L" in data["views"] and "R" in data["views"]:
            # 144/24 会话误存进 dual_1-3.json 时仍带 same_side + 双路
            pr = (data.get("prior") or {}).get("wallDist82") or (data.get("prior") or {}).get("wallDist2")
            if pr is not None:
                return True
    return False


def wall2_complete(data: dict) -> bool:
    views = data.get("views") or {}
    if not isinstance(views, dict):
        return False
    for key in ("L", "R"):
        v = views.get(key) or {}
        w2 = next((w for w in (v.get("walls") or []) if w.get("wall_id") == 2), None)
        if not w2 or len(w2.get("quad") or []) != 4:
            return False
    return True


def stamp_144_24(data: dict) -> dict:
    out = dict(data)
    out["calib_id"] = CALIB_ID
    out.setdefault("layout", "same_side")
    # 已选的视频源保留；缺省才回到 144/24 拼接片
    vs = dict(out.get("video_sources") or {})
    vs.setdefault("mode", "stitched")
    vs.setdefault("src", "output/dualcam/src.mp4")
    vs.setdefault("L", "data/test-144.mp4")
    vs.setdefault("R", "data/test-24.mp4")
    out["video_sources"] = vs
    return out


def migrate_misplaced_144_24() -> str | None:
    """若 144/24 标定误写在 dual_1-3.json，迁到 dual_144-24.json。返回说明文字。"""
    if not CALIB_1_3.is_file():
        return None
    try:
        misplaced = json.loads(CALIB_1_3.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not is_144_24_payload(misplaced):
        return None

    dest = stamp_144_24(misplaced)
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    CALIB_144_24.write_text(json.dumps(dest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 恢复 1-3 入库基准（若 git 可用）
    restored = _restore_1_3_baseline()
    return f"已从 dual_1-3.json 迁到 {CALIB_144_24.relative_to(ROOT)}" + (
        f"；{restored}" if restored else ""
    )


def _restore_1_3_baseline() -> str | None:
    import subprocess

    try:
        raw = subprocess.run(
            ["git", "show", "HEAD:output/calib/dual_1-3.json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        baseline = json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return None
    if is_144_24_payload(baseline):
        return None
    CALIB_1_3.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    return "已恢复 output/calib/dual_1-3.json 为 1-3 组基准"
