#!/usr/bin/env python3
"""拼接双路：左右半幅分别 RTMDet-m + RTMPose-m。坐标留在各半幅原像素，不缩到 1280×720。

产物只写 output/dualcam/。解释器须用 visual-dps conda，并设 PYTHONNOUSERSITE=1。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer_pose_single import MODELS, _as_people  # noqa: E402

OUT_DIR = ROOT / "output" / "dualcam"


def infer(
    video: Path,
    out_npz: Path,
    *,
    stride: int = 1,
    max_frames: int = 0,
    det_size: tuple[int, int] = (640, 640),
) -> dict:
    from rtmlib.tools.object_detection.rtmdet import RTMDet
    from rtmlib.tools.pose_estimation.rtmpose import RTMPose

    det = RTMDet(
        onnx_model=str(MODELS / "rtmdet_m/end2end.onnx"),
        model_input_size=det_size,
        backend="onnxruntime",
        device="cuda",
    )
    pose = RTMPose(
        onnx_model=str(MODELS / "rtmpose_m/end2end.onnx"),
        model_input_size=(192, 256),
        backend="onnxruntime",
        device="cuda",
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width < 2 or width % 2:
        raise SystemExit(f"拼接片宽度必须是偶数：{width}")
    half_w = width // 2

    frames: list[dict] = []
    n_people = 0
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            left, right = fr[:, :half_w], fr[:, half_w : half_w * 2]
            sides = {}
            for name, crop in (("L", left), ("R", right)):
                boxes = det(crop)
                if boxes is None or len(boxes) == 0:
                    k, s, _b = _as_people(None, None, None)
                else:
                    arr = np.asarray(boxes, np.float32).reshape(-1, np.asarray(boxes).shape[-1])
                    kpts, scores = pose(crop, bboxes=arr[:, :4].tolist())
                    k, s, _b = _as_people(arr, kpts, scores)
                n_people += int(len(k))
                sides[name] = {"k": k, "s": s}
            frames.append({"i": i, "t": i / fps, "L": sides["L"], "R": sides["R"]})
            if len(frames) % 200 == 0:
                print(f"  pose {i}/{n_src}  kept={len(frames)}  people={n_people}", flush=True)
            if max_frames and len(frames) >= max_frames:
                i += 1
                break
        i += 1
    cap.release()
    if not frames:
        raise SystemExit(f"未读到任何帧：{video}")

    meta = {
        "video": str(video),
        "width": width,
        "height": height,
        "half_w": half_w,
        "half_h": height,
        "fps": fps,
        "stride": stride,
        "n_src": i,
        "n_pose_frames": len(frames),
        "n_people": n_people,
        "det": "rtmdet-m",
        "pose": "rtmpose-m",
        "coord": "half_native_pixel",
    }
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        frames=np.array(frames, dtype=object),
        meta=np.array(meta, dtype=object),
    )
    summary = out_npz.with_suffix(".json")
    summary.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_npz}  pose_frames={len(frames)}  people={n_people}", flush=True)
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="拼接双路 RTMDet-m + RTMPose-m，半幅原像素")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="npz 路径，同名 json 写摘要")
    ap.add_argument("--stride", type=int, default=1, help="隔几帧提一次，1=全帧")
    ap.add_argument("--max-frames", type=int, default=0, help="最多保留多少提姿态帧，0=不限")
    args = ap.parse_args()
    infer(args.video, args.out, stride=args.stride, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
