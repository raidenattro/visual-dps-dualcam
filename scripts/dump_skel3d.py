#!/usr/bin/env python3
"""从已落的两路 2D 姿态三角化 17 点，写出播放器用 JSON。不重跑 RTMPose。

每帧可有多个人：左右路 NMS + 续帧匹配，只保留两路都看见且落在巷道内的人。
单路闪断最多沿用 8 帧。腕点跟高置信度那路建 3D；单路∩面只显示不报贴墙。
默认先短窗平滑左右路 2D 再抬 3D，3D 只做轻滤波+骨长。
L/R 仍是原始检测（贴墙确认）；Lsm/Rsm 仅供画面看平滑效果。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dualcam_lift import (
    KPT_MIN,
    LWRIST,
    PREFER_PX,
    RWRIST,
    _torso_xy,
    lift_point,
    load_cams,
    pick_pairs,
    signed_x,
)
from scripts.skel3d_smooth import (
    assign_tracks,
    copy_pose_pack,
    smooth_frames,
    smooth_pose2d,
    torso_centroid,
    wrist_jump_stats,
)

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


def _keep_every(pack: list, step: int) -> list:
    """按源帧号隔 step 帧取一帧。step=2 即 25fps→约 12.5fps。"""
    if step <= 1:
        return pack
    return [fr for fr in pack if int(fr["i"]) % step == 0]


SHOULDER_WRIST_MAX = 0.85  # 超过则腕点几何不可信
HOLD_FRAMES = 8  # 单路闪断时沿用上一帧 3D，约 0.32s
LSHO, RSHO = 5, 6


def _lift_person(
    kl,
    sl,
    kr,
    sr,
    gap: float,
    cams: dict,
    plane: dict,
    prev_xyz: list | None = None,
    kl_raw=None,
    kr_raw=None,
) -> dict:
    xyz, vis, L2, R2, Lsm, Rsm, sL, sR, jg, src = [], [], [], [], [], [], [], [], [], []
    dL = dR = None
    raw3 = [None] * 17
    kl_raw = kl if kl_raw is None else kl_raw
    kr_raw = kr if kr_raw is None else kr_raw
    for k in range(17):
        L2.append(_xy(kl_raw[k]))
        R2.append(_xy(kr_raw[k]))
        Lsm.append(_xy(kl[k]))
        Rsm.append(_xy(kr[k]))
        sL.append(round(float(sl[k]), 2))
        sR.append(round(float(sr[k]), 2))
        prev = None
        if prev_xyz and k < len(prev_xyz) and prev_xyz[k]:
            prev = np.asarray(prev_xyz[k], float)
        p, g, kind = lift_point(kl[k], float(sl[k]), kr[k], float(sr[k]), cams, plane, prev)
        jg.append(None if g is None else round(float(g), 3))
        src.append(kind)
        if p is None:
            xyz.append(None)
            vis.append(0)
            continue
        raw3[k] = p
        xyz.append(_xyz(p))
        vis.append(1)

    def _drop_wrist(wi: int, shi: int) -> None:
        w, sh = raw3[wi], raw3[shi]
        if w is None or sh is None:
            return
        if float(np.linalg.norm(w - sh)) > SHOULDER_WRIST_MAX:
            xyz[wi] = None
            vis[wi] = 0
            raw3[wi] = None
            src[wi] = None

    _drop_wrist(LWRIST, LSHO)
    _drop_wrist(RWRIST, RSHO)
    if raw3[LWRIST] is not None:
        dL = round(signed_x(raw3[LWRIST], plane), 3)
    if raw3[RWRIST] is not None:
        dR = round(signed_x(raw3[RWRIST], plane), 3)
    return {
        "gap": round(float(gap), 3),
        "xyz": xyz,
        "vis": vis,
        "jg": jg,
        "src": src,
        "L": L2,
        "R": R2,
        "Lsm": Lsm,
        "Rsm": Rsm,
        "sL": sL,
        "sR": sR,
        "dL": dL,
        "dR": dR,
    }


def _prefer_of(persons: list[dict]) -> list[tuple]:
    out = []
    for p in persons:
        if p.get("held"):
            continue
        sl = np.asarray(p.get("sL") or [], float)
        sr = np.asarray(p.get("sR") or [], float)
        kl = np.asarray(p.get("L") or [], float)
        kr = np.asarray(p.get("R") or [], float)
        if len(kl) < 13 or len(kr) < 13:
            continue
        a, b = _torso_xy(kl, sl), _torso_xy(kr, sr)
        if a is not None and b is not None:
            out.append((a, b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="从已落姿态三角化 17 点，默认先滤 2D 再轻滤 3D")
    ap.add_argument("--no-smooth", action="store_true", help="2D/3D 都不平滑，写出原始三角化")
    ap.add_argument(
        "--keep-stride",
        type=int,
        default=1,
        help="源帧号隔几帧取一帧后再平滑/抬 3D；2=约 12.5fps。>1 必须 --out 另存",
    )
    ap.add_argument("--out", type=Path, default=None, help="写出路径，默认 output/dualcam/skel3d.json")
    args = ap.parse_args()
    if args.keep_stride < 1:
        print("--keep-stride 必须 >= 1", file=sys.stderr)
        return 2
    out = (args.out if args.out is not None else OUT).resolve()
    if args.keep_stride > 1 and out == OUT.resolve():
        print("keep-stride>1 必须 --out 另存，避免覆盖 skel3d.json", file=sys.stderr)
        return 2

    cams, plane, sol = load_cams()
    raw_pack = list(np.load(NPZ, allow_pickle=True)["frames"])
    raw_pack = _keep_every(raw_pack, args.keep_stride)
    stride = _infer_stride(raw_pack)
    pack = copy_pose_pack(raw_pack)
    smooth2d_info = None
    if not args.no_smooth:
        smooth2d_info = smooth_pose2d(pack)
    frames = []
    n_paired = 0
    n_people = 0
    prefer: list = []
    holds: list[tuple[dict, int, np.ndarray]] = []  # person, miss, torso
    for fi, fr in enumerate(pack):
        rec = {"i": int(fr["i"]), "t": round(float(fr["t"]), 3)}
        raw = raw_pack[fi]
        pairs = pick_pairs(fr["L"], fr["R"], cams, prefer=prefer)
        persons = []
        for a, b, gap in pairs:
            kl, sl = fr["L"]["k"][a], fr["L"]["s"][a]
            kr, sr = fr["R"]["k"][b], fr["R"]["s"][b]
            lxy = _torso_xy(kl, sl)
            prev_xyz = None
            if lxy is not None:
                best_d = PREFER_PX
                for prev, _miss, _pt in holds:
                    hxy = _torso_xy(
                        np.asarray(prev.get("L") or [], float),
                        np.asarray(prev.get("sL") or [], float),
                    )
                    if hxy is None:
                        continue
                    d = float(np.linalg.norm(lxy - hxy))
                    if d < best_d:
                        best_d = d
                        prev_xyz = prev.get("xyz")
            persons.append(
                _lift_person(
                    kl, sl, kr, sr, gap, cams, plane, prev_xyz,
                    kl_raw=raw["L"]["k"][a], kr_raw=raw["R"]["k"][b],
                )
            )
        cents = [torso_centroid(p.get("xyz") or []) for p in persons]
        used: set[int] = set()
        new_holds: list[tuple[dict, int, np.ndarray]] = []
        for prev, miss, pt in holds:
            best, best_d = None, 0.55
            for ci, c in enumerate(cents):
                if ci in used or c is None:
                    continue
                d = float(np.linalg.norm(pt - c))
                if d < best_d:
                    best, best_d = ci, d
            if best is not None:
                used.add(best)
                new_holds.append((persons[best], 0, cents[best]))
            elif miss + 1 <= HOLD_FRAMES:
                held = copy.deepcopy(prev)
                held["held"] = True
                persons.append(held)
                new_holds.append((held, miss + 1, pt))
        for ci, p in enumerate(persons):
            if ci in used or p.get("held"):
                continue
            c = cents[ci] if ci < len(cents) else torso_centroid(p.get("xyz") or [])
            if c is None:
                continue
            new_holds.append((p, 0, c))
        holds = new_holds
        prefer = _prefer_of(persons)
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
        "smooth2d": smooth2d_info,
        "smooth": smooth_info,
        "frames": frames,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        f"wrote {out}  frames={len(frames)} paired={n_paired} people={n_people} "
        f"stride={stride}  {out.stat().st_size / 1e6:.1f}MB"
    )
    if smooth2d_info:
        print(
            f"smooth2d tracks={smooth2d_info['n_tracks']}  "
            f"wrist {smooth2d_info['n_wrist_in']}→{smooth2d_info['n_wrist_out']}"
        )
    if smooth_info:
        print(
            f"smooth3d tracks={smooth_info['n_tracks']}  "
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
