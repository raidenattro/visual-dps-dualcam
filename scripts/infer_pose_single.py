#!/usr/bin/env python3
"""单路视频：RTMDet-m 检测 + RTMPose-m 骨骼。坐标留在画面原像素，不对半切、不缩到 1280×720。

产物只写 output/dualcam/。解释器须用 visual-dps conda（本仓 .venv 无 rtmlib / CUDA ORT）。
用户目录里的 CPU 版 onnxruntime 会挡住 CUDA，启动前设 PYTHONNOUSERSITE=1。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path("/home/hqit/workspace/visual-dps-0817-deploy/weights/rtmpose_onnx")
OUT_DIR = ROOT / "output" / "dualcam"


def _as_people(boxes, kpts, scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """统一成 (P,5) 框、(P,17,2) 点、(P,17) 分。没人时三个都是 0 行。"""
    empty_k = np.zeros((0, 17, 2), np.float32)
    empty_s = np.zeros((0, 17), np.float32)
    empty_b = np.zeros((0, 5), np.float32)
    if kpts is None or scores is None:
        return empty_k, empty_s, empty_b
    k = np.asarray(kpts, np.float32)
    s = np.asarray(scores, np.float32)
    if k.size == 0:
        return empty_k, empty_s, empty_b
    if k.ndim == 2:
        k = k.reshape(1, -1, 2)
    if s.ndim == 1:
        s = s.reshape(1, -1)
    b = empty_b
    if boxes is not None and len(boxes):
        arr = np.asarray(boxes, np.float32).reshape(-1, np.asarray(boxes).shape[-1])
        if arr.shape[1] >= 5:
            b = arr[: len(k), :5].astype(np.float32, copy=False)
        else:
            pad = np.zeros((len(k), 5), np.float32)
            pad[:, : arr.shape[1]] = arr[: len(k)]
            b = pad
    return k, s, b


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
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[dict] = []
    n_people = 0
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            boxes = det(fr)
            if boxes is None or len(boxes) == 0:
                k, s, b = _as_people(None, None, None)
            else:
                arr = np.asarray(boxes, np.float32).reshape(-1, np.asarray(boxes).shape[-1])
                kpts, scores = pose(fr, bboxes=arr[:, :4].tolist())
                k, s, b = _as_people(arr, kpts, scores)
            n_people += int(len(k))
            frames.append({"i": i, "t": i / fps, "k": k, "s": s, "boxes": b})
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
        "fps": fps,
        "stride": stride,
        "n_src": i,
        "n_pose_frames": len(frames),
        "n_people": n_people,
        "det": "rtmdet-m",
        "pose": "rtmpose-m",
        "coord": "native_pixel",
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
    ap = argparse.ArgumentParser(description="单路 RTMDet-m + RTMPose-m，原像素坐标")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="npz 路径，同名 json 写摘要")
    ap.add_argument("--stride", type=int, default=1, help="隔几帧提一次，1=全帧")
    ap.add_argument("--max-frames", type=int, default=0, help="最多保留多少提姿态帧，0=不限")
    args = ap.parse_args()
    infer(args.video, args.out, stride=args.stride, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
