#!/usr/bin/env python3
"""从已落的两路 2D 姿态三角化 17 点，写出播放器用 JSON。不重跑 RTMPose。

每帧可有多个人：左右路按交会缝贪心匹配，只保留两路都看见的人。
默认对 3D 做时序平滑（腕/肘更强），贴墙判定与画面用同一套坐标。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dualcam_lift import (
    KPT_MIN, LWRIST, RWRIST, load_cams, pick_pairs, signed_x, triangulate,
)
from scripts.skel3d_smooth import assign_tracks, smooth_frames, wrist_jump_stats

NPZ = ROOT / "output/dualcam/poses_5fps.npz"
OUT = ROOT / "output/dualcam/skel3d.json"
EDGES = [
    [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 5], [0, 6],
]


def _xy(arr: np.ndarray) -> list:
    return [round(float(v), 1) for v in arr]


def _xyz(p: np.ndarray) -> list:
    return [round(float(v), 3) for v in p]


def _infer_stride(pack) -> int:
    if len(pack) < 2:
        return 1
    d = int(pack[1]["i"]) - int(pack[0]["i"])
    return d if d > 0 else 1


def _lift_person(kl, sl, kr, sr, gap: float, cams: dict, plane: dict) -> dict:
    xyz, vis, L2, R2, sL, sR = [], [], [], [], [], []
    dL = dR = None
    for k in range(17):
        L2.append(_xy(kl[k]))
        R2.append(_xy(kr[k]))
        sL.append(round(float(sl[k]), 2))
        sR.append(round(float(sr[k]), 2))
        if sl[k] < KPT_MIN or sr[k] < KPT_MIN:
            xyz.append(None)
            vis.append(0)
            continue
        p, _g = triangulate(kl[k], kr[k], cams)
        xyz.append(_xyz(p))
        vis.append(1)
        if k == LWRIST:
            dL = round(signed_x(p, plane), 3)
        if k == RWRIST:
            dR = round(signed_x(p, plane), 3)
    return {
        "gap": round(float(gap), 3),
        "xyz": xyz,
        "vis": vis,
        "L": L2,
        "R": R2,
        "sL": sL,
        "sR": sR,
        "dL": dL,
        "dR": dR,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="从已落姿态三角化 17 点，默认对腕/肘做时序平滑")
    ap.add_argument("--no-smooth", action="store_true", help="不平滑，写出原始三角化")
    args = ap.parse_args()

    cams, plane, sol = load_cams()
    pack = np.load(NPZ, allow_pickle=True)["frames"]
    stride = _infer_stride(pack)
    frames = []
    n_paired = 0
    n_people = 0
    for fr in pack:
        rec = {"i": int(fr["i"]), "t": round(float(fr["t"]), 3)}
        pairs = pick_pairs(fr["L"], fr["R"], cams)
        persons = []
        for a, b, gap in pairs:
            kl, sl = fr["L"]["k"][a], fr["L"]["s"][a]
            kr, sr = fr["R"]["k"][b], fr["R"]["s"][b]
            persons.append(_lift_person(kl, sl, kr, sr, gap, cams, plane))
        rec["ok"] = len(persons) > 0
        rec["persons"] = persons
        frames.append(rec)
        if persons:
            n_paired += 1
            n_people += len(persons)

    smooth_info = None
    jump_before = jump_after = None
    if not args.no_smooth:
        assign_tracks(frames)
        jump_before = {"L": wrist_jump_stats(frames, LWRIST), "R": wrist_jump_stats(frames, RWRIST)}
        smooth_info = smooth_frames(frames, plane)
        jump_after = {"L": wrist_jump_stats(frames, LWRIST), "R": wrist_jump_stats(frames, RWRIST)}

    def cam_pub(name: str) -> dict:
        c = cams[name]
        return {
            "f": float(c["f"]) if np.ndim(c["f"]) == 0 else float(np.asarray(c["f"]).reshape(-1)[0]),
            "cx": float(c["c0"][0]),
            "cy": float(c["c0"][1]),
            "C": _xyz(c["C"]),
            "right": _xyz(c["right"]),
            "down": _xyz(c["down"]),
            "fwd": _xyz(c["fwd"]),
        }

    payload = {
        "fps": 25.0,
        "stride": stride,
        "kpt_min": KPT_MIN,
        "edges": EDGES,
        "plane": {"x": float(plane["x"]), "n": [1, 0, 0]},
        "walls": sol["walls"],
        "cameras": {"L": cam_pub("L"), "R": cam_pub("R")},
        "n_frames": len(frames),
        "n_paired": n_paired,
        "n_people": n_people,
        "smooth": smooth_info,
        "frames": frames,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        f"wrote {OUT}  frames={len(frames)} paired={n_paired} people={n_people} "
        f"stride={stride}  {OUT.stat().st_size / 1e6:.1f}MB"
    )
    if smooth_info:
        print(
            f"smooth tracks={smooth_info['n_tracks']}  "
            f"wrist {smooth_info['n_wrist_in']}→{smooth_info['n_wrist_out']}"
        )
        if jump_before and jump_after:
            for side in ("L", "R"):
                b, a = jump_before[side], jump_after[side]
                if b.get("n") and a.get("n"):
                    print(
                        f"  {side}wrist jump p90 {b['p90']:.3f}→{a['p90']:.3f}m  "
                        f"p99 {b['p99']:.3f}→{a['p99']:.3f}m  "
                        f">15cm {b['frac_gt_0.15']:.1%}→{a['frac_gt_0.15']:.1%}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
