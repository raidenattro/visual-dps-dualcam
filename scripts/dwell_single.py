#!/usr/bin/env python3
"""单路腕点时间对比：当前帧和 Δ 秒前的同一只手。

10.8 日报里，左右错开约 0.24 秒之后，落点距离量的是手在这段时间里移动了多少。
停在附近才会小，手还在动就会分开。单路没有第二台相机，直接比同一轨迹上
t 与 t−Δ 的腕点。有货架四角时距离走单应、单位米；否则用像素。

产物只写 output/dualcam/。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plane_contact import KPT_MIN, LWRIST, RWRIST, apply_h, homography_to_wall  # noqa: E402

# 日报：25fps 下右路早 6 帧 = 0.24 秒。按视频 fps 换算，不把 6 帧写死。
DEFAULT_DELTA_S = 0.24
DEFAULT_TAU_M = 0.15
DEFAULT_TAU_PX = 20.0
TORSO = (5, 6, 11, 12)
EDGES = [
    [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 5], [0, 6],
]
TRACK_PX = 140.0
TRACK_GAP_S = 0.48
NMS_PX = 50.0


def lag_frames(fps: float, delta_s: float = DEFAULT_DELTA_S) -> int:
    """Δ 秒对应的帧数。25fps → 6，15fps → 4。"""
    return max(1, int(round(float(fps) * float(delta_s))))


def wrist_shift(uv, uv0) -> float | None:
    """两点的平面距离。缺任一端就没有可比较的位移。"""
    if uv is None or uv0 is None:
        return None
    a = np.asarray(uv, float).reshape(-1)
    b = np.asarray(uv0, float).reshape(-1)
    if a.size < 2 or b.size < 2:
        return None
    if not (np.all(np.isfinite(a[:2])) and np.all(np.isfinite(b[:2]))):
        return None
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def dwell_on(dist: float | None, tau: float) -> bool:
    """位移不超过 τ 才算停住。没有对照帧不算。"""
    return dist is not None and float(dist) <= float(tau)


def _torso(k: np.ndarray, s: np.ndarray) -> np.ndarray | None:
    pts = [np.asarray(k[i][:2], float) for i in TORSO if i < len(s) and float(s[i]) >= KPT_MIN]
    if not pts:
        return None
    return np.mean(pts, axis=0)


def _nms(k: np.ndarray, s: np.ndarray) -> list[int]:
    """同帧躯干过近只留均分更高的一个。"""
    n = len(k)
    xy = [_torso(k[i], s[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -float(np.mean(s[i])) if len(s[i]) else 0.0)
    kept: list[int] = []
    for i in order:
        if xy[i] is None:
            continue
        if any(float(np.linalg.norm(xy[i] - xy[j])) < NMS_PX for j in kept if xy[j] is not None):
            continue
        kept.append(i)
    return kept


@dataclass
class _Track:
    tid: int
    last_torso: np.ndarray
    last_t: float
    at: dict[int, int] = field(default_factory=dict)


def assign_tracks(frames: list[dict]) -> list[dict[int, int]]:
    """每帧 det 下标 → track id。躯干对不上就新开一条。"""
    active: list[_Track] = []
    next_id = 0
    out: list[dict[int, int]] = []
    for fr in frames:
        t = float(fr["t"])
        k, s = fr["k"], fr["s"]
        keep = _nms(k, s) if len(k) else []
        xy = {i: _torso(k[i], s[i]) for i in keep}
        alive = [tr for tr in active if (t - tr.last_t) <= TRACK_GAP_S]
        cands: list[tuple[float, int, int]] = []
        for di, p in xy.items():
            if p is None:
                continue
            for ti, tr in enumerate(alive):
                cands.append((float(np.linalg.norm(p - tr.last_torso)), di, ti))
        cands.sort()
        used_d: set[int] = set()
        used_t: set[int] = set()
        mapping: dict[int, int] = {}
        for dist, di, ti in cands:
            if di in used_d or ti in used_t or dist > TRACK_PX:
                continue
            tr = alive[ti]
            tr.at[int(fr["i"])] = di
            tr.last_torso = xy[di]  # type: ignore[assignment]
            tr.last_t = t
            mapping[di] = tr.tid
            used_d.add(di)
            used_t.add(ti)
        for di, p in xy.items():
            if di in used_d or p is None:
                continue
            tr = _Track(next_id, p, t, {int(fr["i"]): di})
            active.append(tr)
            alive.append(tr)
            mapping[di] = next_id
            next_id += 1
        out.append(mapping)
    return out


def _quad_h(calib: dict | None):
    """单路货架四角 → 墙面单应。没有四角就返回 None，距离留在像素。"""
    if not calib:
        return None
    views = calib.get("views") or {}
    view = views.get("L") or views.get("R") or {}
    for w in view.get("walls") or []:
        q = w.get("quad") or []
        if len(q) == 4:
            return homography_to_wall(
                q,
                float(w.get("width") or 2.2),
                float(w.get("height") or 2.0),
                float(w.get("base") or 0.0),
            )
    return None


def _wrist_uv(k: np.ndarray, s: np.ndarray, j: int):
    if j >= len(s) or float(s[j]) < KPT_MIN:
        return None
    return np.asarray(k[j][:2], float)


def build_dwell(
    frames: list[dict],
    *,
    fps: float,
    delta_s: float = DEFAULT_DELTA_S,
    tau_px: float = DEFAULT_TAU_PX,
    tau_m: float = DEFAULT_TAU_M,
    H: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """每帧写出腕点位移。对照帧是同一条轨迹上提前 lag 帧的检测。"""
    lag = lag_frames(fps, delta_s)
    tracks = assign_tracks(frames)
    by_tid: dict[int, dict[int, tuple[int, int]]] = {}
    for fi, mapping in enumerate(tracks):
        src_i = int(frames[fi]["i"])
        for di, tid in mapping.items():
            by_tid.setdefault(tid, {})[src_i] = (fi, di)

    unit = "m" if H is not None else "px"
    tau = tau_m if H is not None else tau_px
    out_frames: list[dict] = []
    dists: list[float] = []
    run = 0
    runs: list[int] = []

    def locate(tid: int, src_i: int):
        hit = by_tid.get(tid, {}).get(src_i)
        if hit is None:
            return None
        fi, di = hit
        return frames[fi]["k"][di], frames[fi]["s"][di]

    for fi, fr in enumerate(frames):
        src_i = int(fr["i"])
        people = []
        frame_on = False
        for di, tid in tracks[fi].items():
            k = np.asarray(fr["k"][di], float)
            s = np.asarray(fr["s"][di], float)
            joints = []
            for j in range(min(17, len(s))):
                joints.append([round(float(k[j][0]), 1), round(float(k[j][1]), 1), round(float(s[j]), 3)])
            wrists = {}
            past = locate(tid, src_i - lag)
            for j in (LWRIST, RWRIST):
                uv = _wrist_uv(k, s, j)
                uv0 = None
                if past is not None:
                    uv0 = _wrist_uv(past[0], past[1], j)
                if H is not None and uv is not None and uv0 is not None:
                    dist = wrist_shift(apply_h(H, uv), apply_h(H, uv0))
                else:
                    dist = wrist_shift(uv, uv0)
                on = dwell_on(dist, tau)
                if dist is not None:
                    dists.append(dist)
                if on:
                    frame_on = True
                rec = {
                    "d": None if dist is None else round(dist, 3 if unit == "m" else 1),
                    "on": on,
                }
                if uv0 is not None:
                    rec["u0"] = [round(float(uv0[0]), 1), round(float(uv0[1]), 1)]
                wrists[str(j)] = rec
            people.append({"id": tid, "k": joints, "w": wrists})
        if frame_on:
            run += 1
        elif run:
            runs.append(run)
            run = 0
        out_frames.append({"i": src_i, "t": round(float(fr["t"]), 4), "people": people})
    if run:
        runs.append(run)

    arr = np.asarray(dists, float)
    runs_a = np.asarray(runs, float) if runs else np.asarray([], float)
    summary = {
        "delta_s": delta_s,
        "lag_frames": lag,
        "unit": unit,
        "tau": tau,
        "n_compared": int(arr.size),
        "le_tau": None if arr.size == 0 else round(float(np.mean(arr <= tau)), 3),
        "dist_p50": None if arr.size == 0 else round(float(np.median(arr)), 3),
        "run_median_frames": None if runs_a.size == 0 else round(float(np.median(runs_a)), 1),
        "run_single": None if runs_a.size == 0 else round(float(np.mean(runs_a == 1)), 3),
        "n_tracks": len(by_tid),
    }
    return out_frames, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="单路腕点：当前帧对比 Δ 秒前")
    ap.add_argument("--poses", type=Path, default=ROOT / "output/dualcam/poses_test_480p.npz")
    ap.add_argument("--video", type=str, default="output/dualcam/test_480p.mp4")
    ap.add_argument("--out", type=Path, default=ROOT / "output/dualcam/dwell_test_480p.json")
    ap.add_argument("--calib", type=Path, default=None, help="有货架四角时，位移改用墙面米")
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA_S, help="对比间隔（秒），默认 0.24")
    ap.add_argument("--tau-px", type=float, default=DEFAULT_TAU_PX)
    ap.add_argument("--tau-m", type=float, default=DEFAULT_TAU_M)
    args = ap.parse_args()

    z = np.load(args.poses, allow_pickle=True)
    frames = list(z["frames"])
    meta_npz = z["meta"].item() if "meta" in z.files else {}
    fps = float(meta_npz.get("fps") or 15.0)
    H = None
    if args.calib is not None:
        H = _quad_h(json.loads(args.calib.read_text(encoding="utf-8")))
        if H is None:
            print("标定里没有四角，距离仍用像素", file=sys.stderr)
    built, summary = build_dwell(
        frames, fps=fps, delta_s=args.delta, tau_px=args.tau_px, tau_m=args.tau_m, H=H,
    )
    payload = {
        "video": args.video,
        "width": int(meta_npz.get("width") or 0),
        "height": int(meta_npz.get("height") or 0),
        "fps": fps,
        "edges": EDGES,
        "kpt_min": KPT_MIN,
        "logic": "同一轨迹上，腕点 t 与 t−Δ 的位移。停住才小，对应左右错开之后比的那个量。",
        **summary,
        "frames": built,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        f"wrote {args.out}  lag={summary['lag_frames']}帧"
        f"  ≤τ={summary['le_tau']}  中位位移={summary['dist_p50']}{summary['unit']}"
        f"  红段中位={summary['run_median_frames']}帧",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
