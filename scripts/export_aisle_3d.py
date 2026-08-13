#!/usr/bin/env python3
"""导出一段 2D 骨架+货框网格，供 aisle3d WebGL 用给定相机参数抬成 3D。

3D 还原在浏览器里做，参数可调。产物：output/viz/aisle3d/{scene.json,index.html}
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import subprocess

from adapters.collector_paths import load_paths
from adapters.record_reader import list_records_from_manifest, load_record, load_segment_split, record_dir
from scripts.render_fn_fp_frames import _resolve_video

EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]

VIEWER_SRC = ROOT / "scripts" / "aisle3d_viewer.html"


def _wall_id(box_id: str) -> int:
    try:
        return int(box_id) // 1000
    except ValueError:
        return 0


def build_scene(rec, seg: dict[str, Any]) -> dict[str, Any]:
    width = int(rec.meta.get("infer_width") or 852)
    height = int(rec.meta.get("infer_height") or 480)
    gt = set(seg.get("gt_tokens") or [])
    tracks = {int(x) for x in (seg.get("person_track_ids") or []) if str(x).isdigit()}
    layers = [int(b.layer or 1) for b in rec.boxes]
    cols = [int(b.column or 1) for b in rec.boxes]

    boxes_out = []
    for b in rec.boxes:
        boxes_out.append(
            {
                "token": b.token,
                "wall": _wall_id(b.box_id),
                "layer": int(b.layer or 1),
                "column": int(b.column or 1),
                "gt": b.token in gt,
                "center2d": [round(float(b.center[0]), 2), round(float(b.center[1]), 2)],
            }
        )

    f0, f1 = int(seg["frame_start"]), int(seg["frame_end"])
    by = {int(fr["frame_idx"]): fr for fr in rec.frames}
    frames_out = []
    for fi in range(f0, f1 + 1):
        fr = by.get(fi)
        if fr is None:
            continue
        people = []
        for p in fr.get("persons") or []:
            tid = int(p.get("person_track_id") or 0)
            kps = p.get("keypoints") or []
            if len(kps) < 17:
                continue
            people.append(
                {
                    "track": tid,
                    "focus": tid in tracks or not tracks,
                    "kpts": [[round(float(a), 2) for a in kp[:3]] for kp in kps[:17]],
                }
            )
        if people:
            frames_out.append({"frame": fi, "people": people})

    return {
        "record_id": rec.ref.record_id,
        "clip_name": rec.ref.clip_name,
        "camera_slug": rec.ref.camera_slug,
        "image_size": [width, height],
        "n_layers": max(layers) if layers else 4,
        "n_cols": max(cols) if cols else 4,
        "seg": {
            "frame_start": f0,
            "frame_end": f1,
            "gt_tokens": list(gt),
            "tracks": sorted(tracks),
            "split": seg.get("split"),
        },
        "defaults": {
            "camH": 3.0,
            "camDist": 1.5,
            "pitch": 45.0,
            "yaw": 0.0,
            "fovH": 90.0,
            "aisle": 1.7,
            "boxW": 0.42,
            "boxH": 0.36,
            "boxD": 0.48,
            "baseY": 0.12,
            "kptMin": 0.25,
        },
        "fps": float(rec.fps or 25.0),
        "video": "clip.mp4",
        "edges": EDGES,
        "boxes": boxes_out,
        "frames": frames_out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v4.json"))
    ap.add_argument("--record-id", default="")
    ap.add_argument("--seg-id", default="")
    ap.add_argument("--out-dir", default=str(ROOT / "output/viz/aisle3d"))
    args = ap.parse_args()

    man = Path(args.manifest)
    refs = {r.record_id: r for r in list_records_from_manifest(man)}
    segs = load_segment_split(man)
    rid = args.record_id or (
        "rtmpose-m/1-1-1-(2)/00000000907002300_seg10_26-05_to_26-42_rtmpose_m"
    )
    if rid not in refs:
        raise SystemExit(f"record 不存在: {rid}")
    rec = load_record(refs[rid])
    rec_segs = segs.get(rid) or []
    if args.seg_id:
        seg = next((s for s in rec_segs if s.get("seg_id") == args.seg_id), None)
        if seg is None:
            raise SystemExit(f"seg 不存在: {args.seg_id}")
    else:
        val = [s for s in rec_segs if s.get("split") == "val"]
        seg = val[0] if val else rec_segs[0]

    scene = build_scene(rec, seg)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rec_dir = record_dir(refs[rid])
    rec_man = json.loads((rec_dir / "manifest.json").read_text(encoding="utf-8"))
    src_video = _resolve_video(rid, rec_man, load_paths())
    clip_path = out_dir / "clip.mp4"
    if src_video and src_video.is_file():
        fps = max(float(rec.fps or rec_man.get("fps") or 25.0), 1e-6)
        f0 = int(seg["frame_start"])
        f1 = int(seg["frame_end"])
        ss = max(0.0, (f0 - 1) / fps)
        dur = max(0.04, (f1 - f0 + 1) / fps)
        cmd = [
            "ffmpeg", "-y", "-ss", f"{ss:.4f}", "-i", str(src_video),
            "-t", f"{dur:.4f}", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(clip_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise SystemExit(f"ffmpeg 失败: {e.stderr[-800:] if e.stderr else e}") from e
        scene["fps"] = fps
        scene["video"] = "clip.mp4"
    else:
        scene["video"] = ""

    (out_dir / "scene.json").write_text(
        json.dumps(scene, ensure_ascii=False), encoding="utf-8"
    )
    if not VIEWER_SRC.is_file():
        raise SystemExit(f"缺少 {VIEWER_SRC}")
    shutil.copyfile(VIEWER_SRC, out_dir / "index.html")
    print(
        f"out → {out_dir}\n"
        f"record={rid}\n"
        f"seg={seg.get('frame_start')}–{seg.get('frame_end')} gt={seg.get('gt_tokens')}\n"
        f"frames={len(scene['frames'])} boxes={len(scene['boxes'])} "
        f"video={scene.get('video') or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
