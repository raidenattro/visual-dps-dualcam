"""在已同步的骨架上模拟左右时间差，另存 skel，不改原文件。

src 上测到右路大约早 6 帧（0.24s）。这里对 94、83-84 做同样的错开：
左路用当前帧，右路用 fi+lag 的同一条轨迹。lag=+6 即右路早 6 帧。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plane_contact import LWRIST, RWRIST, PlaneContact  # noqa: E402

PAIRS = {
    "94": (ROOT / "output/dualcam/skel3d_94.json", ROOT / "output/calib/dual_94.json"),
    "83-84": (ROOT / "output/dualcam/skel3d_83-84.json", ROOT / "output/calib/dual_83-84.json"),
}


def _track_index(frames: list[dict]) -> dict:
    by: dict = defaultdict(dict)
    for fi, fr in enumerate(frames):
        for pi, p in enumerate(fr.get("persons") or []):
            if p.get("held"):
                continue
            tid = p.get("track_id")
            if tid is None:
                continue
            by[tid][fi] = pi
    return by


def _kpts(person: dict, side: str):
    if side == "L":
        return person.get("Lsm") or person.get("L"), person.get("sL")
    return person.get("Rsm") or person.get("R"), person.get("sR")


def apply_lag(frames: list[dict], pc: PlaneContact, lag: int) -> None:
    """左路帧 fi 配右路帧 fi+lag。配不上则清掉 pc，避免留下原来的同步结果。"""
    by = _track_index(frames)
    for fi, fr in enumerate(frames):
        for p in fr.get("persons") or []:
            if p.get("held") or p.get("track_id") is None:
                p["pc"] = {"9": None, "10": None}
                continue
            other_pi = by[p["track_id"]].get(fi + lag)
            if other_pi is None:
                p["pc"] = {"9": None, "10": None}
                continue
            other = frames[fi + lag]["persons"][other_pi]
            kl, sl = _kpts(p, "L")
            kr, sr = _kpts(other, "R")
            p["pc"] = pc.person(kl, sl, kr, sr)


def _stats(frames: list[dict], tau: float = 0.15) -> dict:
    dist = []
    close_speed = []
    far_speed = []
    runs = []
    run = 0
    prev = {}
    for fr in frames:
        on = False
        for p in fr.get("persons") or []:
            if p.get("held"):
                continue
            tid = p.get("track_id")
            kl, sl = _kpts(p, "L")
            for j in (LWRIST, RWRIST):
                r = (p.get("pc") or {}).get(str(j))
                if not r or r.get("dist") is None or not r.get("in"):
                    continue
                dist.append(r["dist"])
                speed = None
                if kl and sl and sl[j] >= 0.3 and tid is not None:
                    key = (tid, j)
                    uv = kl[j]
                    if key in prev:
                        speed = float(np.hypot(uv[0] - prev[key][0], uv[1] - prev[key][1]))
                    prev[key] = uv
                if speed is not None:
                    (close_speed if r["dist"] <= tau else far_speed).append(speed)
                if r.get("contact"):
                    on = True
        if on:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    dist_a = np.asarray(dist, float)
    runs_a = np.asarray(runs, float) if runs else np.asarray([0.0])
    def med(a):
        a = np.asarray(a, float)
        return None if a.size == 0 else round(float(np.median(a)), 3)
    return {
        "in_wall": int(dist_a.size),
        "le_tau": None if dist_a.size == 0 else round(float(np.mean(dist_a <= tau)), 3),
        "contact_runs": int(runs_a.size) if runs else 0,
        "run_median_frames": None if not runs else round(float(np.median(runs_a)), 1),
        "run_single": None if not runs else round(float(np.mean(runs_a == 1)), 3),
        "speed_px_when_close": med(close_speed),
        "speed_px_when_far": med(far_speed),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="模拟右路提前，另存骨架")
    ap.add_argument("--lags", default="6,12", help="右路提前的帧数，逗号分隔")
    args = ap.parse_args()
    lags = [int(x) for x in args.lags.split(",") if x.strip()]
    summary = {"note": "lag>0 表示右路早于左路，模仿 src 测到的右路早约 6 帧", "sets": []}
    for name, (skel_path, calib_path) in PAIRS.items():
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        pc = PlaneContact.from_calib(calib, tau=0.15)
        base = json.loads(skel_path.read_text(encoding="utf-8"))
        summary["sets"].append({"name": name, "lag": 0, "file": str(skel_path.relative_to(ROOT)), **_stats(base["frames"])})
        for lag in lags:
            data = json.loads(skel_path.read_text(encoding="utf-8"))
            apply_lag(data["frames"], pc, lag)
            st = _stats(data["frames"])
            out = ROOT / "output" / "dualcam" / f"skel3d_{name}_lag{lag}.json"
            data["sync_sim"] = {"source": skel_path.name, "lag_frames": lag, "meaning": "右路早 lag 帧", "stats": st}
            out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            print(f"wrote {out.relative_to(ROOT)}  ≤τ={st['le_tau']}  红段中位={st['run_median_frames']}帧")
            summary["sets"].append({"name": name, "lag": lag, "file": str(out.relative_to(ROOT)), **st})
    sum_path = ROOT / "output" / "dualcam" / "sync_sim_summary.json"
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {sum_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
