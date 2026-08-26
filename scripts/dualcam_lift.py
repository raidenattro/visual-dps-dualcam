#!/usr/bin/env python3
"""两路 RTMPose → 三角化，对照单路射线∩拣货面。产物只写 output/dualcam/。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "output/calib/dual_1-3.json"
VIDEO = ROOT / "output/dualcam/src.mp4"
OUT = ROOT / "output/dualcam"
MODELS = Path("/home/hqit/workspace/visual-dps-0817-deploy/weights/rtmpose_onnx")
LWRIST, RWRIST = 9, 10
KPT_MIN = 0.3


def load_cams() -> tuple[dict, dict, dict]:
    data = json.loads(CALIB.read_text(encoding="utf-8"))
    sol = data["solved"]
    cams = {}
    for name, c in sol["cameras"].items():
        cams[name] = {
            "f": np.array(c["f"], float),
            "c0": np.array([c["cx"], c["cy"]], float),
            "C": np.array(c["C"], float),
            "right": np.array(c["right"], float),
            "down": np.array(c["down"], float),
            "fwd": np.array(c["fwd"], float),
        }
    wall = next(w for w in sol["walls"] if w["wall_id"] == 1)
    # 平面：点在 corners[0]，法向沿 +X（巷道内侧）
    p0 = np.array(wall["corners"][0], float)
    n = np.array([1.0 if wall["sign"] < 0 else -1.0, 0.0, 0.0])
    # sign=-1 → 墙在 x=-aisle/2，内侧法向 +X
    return cams, {"p0": p0, "n": n, "x": float(p0[0])}, sol


def ray(uv: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    x = (uv[0] - cam["c0"][0]) / cam["f"]
    y = (uv[1] - cam["c0"][1]) / cam["f"]
    d = x * cam["right"] + y * cam["down"] + cam["fwd"]
    n = np.linalg.norm(d)
    return cam["C"], d / n


def triangulate(uv_l: np.ndarray, uv_r: np.ndarray, cams: dict) -> tuple[np.ndarray, float]:
    c1, d1 = ray(uv_l, cams["L"])
    c2, d2 = ray(uv_r, cams["R"])
    w0 = c1 - c2
    a, b, c = float(d1 @ d1), float(d1 @ d2), float(d2 @ d2)
    d, e = float(d1 @ w0), float(d2 @ w0)
    den = a * c - b * b
    if abs(den) < 1e-9:
        return c1 + d1, 99.0
    t = (b * e - c * d) / den
    s = (a * e - b * d) / den
    p1, p2 = c1 + t * d1, c2 + s * d2
    mid = 0.5 * (p1 + p2)
    return mid, float(np.linalg.norm(p1 - p2))


def ray_plane(uv: np.ndarray, cam: dict, plane: dict) -> np.ndarray | None:
    c, d = ray(uv, cam)
    den = float(d @ plane["n"])
    if abs(den) < 1e-6:
        return None
    t = float((plane["p0"] - c) @ plane["n"]) / den
    if t < 0.05:
        return None
    return c + t * d


def signed_x(p: np.ndarray, plane: dict) -> float:
    """巷道内侧为正：人在通道里 >0，伸进墙为 ≤0。"""
    return float((p - plane["p0"]) @ plane["n"])


def infer_video(stride: int, max_keep: int = 0) -> dict:
    from rtmlib.tools.object_detection.rtmdet import RTMDet
    from rtmlib.tools.pose_estimation.rtmpose import RTMPose

    det = RTMDet(
        onnx_model=str(MODELS / "rtmdet_nano/end2end.onnx"),
        model_input_size=(320, 320),
        backend="onnxruntime",
        device="cuda",
    )
    pose = RTMPose(
        onnx_model=str(MODELS / "rtmpose_m/end2end.onnx"),
        model_input_size=(192, 256),
        backend="onnxruntime",
        device="cuda",
    )

    def one(frame):
        boxes = det(frame)
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
        arr = np.asarray(boxes, np.float32).reshape(-1, np.asarray(boxes).shape[-1])
        bboxes = arr[:, :4].tolist()
        k, s = pose(frame, bboxes=bboxes)
        if k is None or len(k) == 0:
            return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
        k = np.asarray(k, np.float32)
        s = np.asarray(s, np.float32)
        if k.ndim == 2:
            k = k.reshape(1, -1, 2)
        if s.ndim == 1:
            s = s.reshape(1, -1)
        return k, s

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            left, right = fr[:, :1280], fr[:, 1280:]
            kl, sl = one(left)
            kr, sr = one(right)
            frames.append({
                "i": i,
                "t": i / fps,
                "L": {"k": kl, "s": sl},
                "R": {"k": kr, "s": sr},
            })
            if len(frames) % 50 == 0:
                print(f"  pose {i}/{n}  kept={len(frames)}", flush=True)
            if max_keep and len(frames) >= max_keep:
                break
        i += 1
    cap.release()
    return {"fps": fps, "stride": stride, "n_src": i, "frames": frames}


def pick_pair(fl: dict, fr: dict, cams: dict) -> tuple[int, int, float] | None:
    best = None
    for i, (kl, sl) in enumerate(zip(fl["k"], fl["s"])):
        for j, (kr, sr) in enumerate(zip(fr["k"], fr["s"])):
            mid = []
            gap = []
            for k in (5, 6, 11, 12, LWRIST, RWRIST):
                if sl[k] < KPT_MIN or sr[k] < KPT_MIN:
                    continue
                p, g = triangulate(kl[k], kr[k], cams)
                mid.append(p)
                gap.append(g)
            if len(gap) < 2:
                continue
            score = float(np.median(gap))
            if best is None or score < best[2]:
                best = (i, j, score)
    return best


def analyze(pack: dict, cams: dict, plane: dict) -> dict:
    rows = []
    for fr in pack["frames"]:
        pair = pick_pair(fr["L"], fr["R"], cams)
        if pair is None:
            continue
        i, j, gap = pair
        kl, sl = fr["L"]["k"][i], fr["L"]["s"][i]
        kr, sr = fr["R"]["k"][j], fr["R"]["s"][j]
        rec = {"i": fr["i"], "t": fr["t"], "pair_gap": gap}
        for name, idx in (("lwrist", LWRIST), ("rwrist", RWRIST)):
            if sl[idx] < KPT_MIN or sr[idx] < KPT_MIN:
                continue
            p, g = triangulate(kl[idx], kr[idx], cams)
            rec[name] = {
                "xyz": [round(float(v), 3) for v in p],
                "gap_m": round(g, 3),
                "d_stereo": round(signed_x(p, plane), 3),
            }
            mono = []
            for view, uv, cam in (("L", kl[idx], cams["L"]), ("R", kr[idx], cams["R"])):
                hit = ray_plane(uv, cam, plane)
                if hit is None:
                    continue
                mono.append({"view": view, "xyz": [round(float(v), 3) for v in hit]})
            rec[name]["ray_plane"] = mono
        if "lwrist" in rec or "rwrist" in rec:
            rows.append(rec)
    return rows


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ):
        if Path(fp).is_file():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = "Noto Sans CJK SC"
            break

    t, dl, dr = [], [], []
    for r in rows:
        t.append(r["t"])
        dl.append((r.get("lwrist") or {}).get("d_stereo"))
        dr.append((r.get("rwrist") or {}).get("d_stereo"))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axhline(0, color="#888", lw=1)
    ax.plot(t, dl, ".", ms=3, label="左腕→拣货面 (m，内侧为正)")
    ax.plot(t, dr, ".", ms=3, label="右腕→拣货面")
    ax.set_xlabel("秒")
    ax.set_ylabel("距离 m")
    ax.set_ylim(-0.6, 2.2)
    ax.legend()
    ax.set_title("三角化：腕到主拣货面（0=贴面，<0=伸进货架）")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close()


def overlays(rows: list[dict], cams: dict, plane: dict, n: int = 6) -> None:
    if not rows:
        return
    ds = []
    for r in rows:
        vals = [r[k]["d_stereo"] for k in ("lwrist", "rwrist") if k in r]
        if vals:
            ds.append((min(vals), r["i"]))
    ds.sort()
    picks = []
    if ds:
        picks.append(("in_or_near", ds[0][1]))
        picks.append(("mid", ds[len(ds) // 2][1]))
        picks.append(("far", ds[-1][1]))
    cap = cv2.VideoCapture(str(VIDEO))
    od = OUT / "overlays"
    od.mkdir(exist_ok=True)
    for tag, fi in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(od / f"{tag}_{fi:06d}.jpg"), fr)
    cap.release()


def main() -> int:
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    max_keep = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    cams, plane, sol = load_cams()
    print(
        f"calib L {sol['per_view']['L']['resid_px']}px  "
        f"R {sol['per_view']['R']['resid_px']}px  "
        f"align {sol['align_rms_m']}m  wall1 x={plane['x']}",
        flush=True,
    )
    pack = infer_video(stride, max_keep)
    np.savez_compressed(OUT / "poses_5fps.npz", frames=np.array(pack["frames"], dtype=object))
    rows = analyze(pack, cams, plane)
    (OUT / "lift_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ds = [r[k]["d_stereo"] for r in rows for k in ("lwrist", "rwrist") if k in r]
    gaps = [r[k]["gap_m"] for r in rows for k in ("lwrist", "rwrist") if k in r]
    report = {
        "n_pose_frames": len(pack["frames"]),
        "n_paired": len(rows),
        "wrist_d_p10": round(float(np.percentile(ds, 10)), 3) if ds else None,
        "wrist_d_p50": round(float(np.percentile(ds, 50)), 3) if ds else None,
        "wrist_d_p90": round(float(np.percentile(ds, 90)), 3) if ds else None,
        "tri_gap_p50_m": round(float(np.median(gaps)), 3) if gaps else None,
        "frac_d_le_0.10": round(float(np.mean(np.array(ds) <= 0.10)), 3) if ds else None,
        "frac_d_gt_0.40": round(float(np.mean(np.array(ds) > 0.40)), 3) if ds else None,
    }
    (OUT / "lift_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot(rows, OUT / "wrist_dist.png")
    overlays(rows, cams, plane)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
