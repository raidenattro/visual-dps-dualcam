#!/usr/bin/env python3
"""对有手腕命中的帧跑单目深度，把「手腕深度 vs 货框深度」聚合成标量存盘。

只在同一帧内比较，因此不受单目深度跨帧尺度漂移的影响。货框深度取多边形内
**排除人体包围盒**后的像素中位数，拿的是没被手挡住的货架本体。

只读 collector 视频；产物写本仓 output/depth/。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_paths
from adapters.record_reader import list_records, load_record
from features.bank import FeatureBank
from features.geometry import KPT_SCORE_MIN
from pipeline.box_trigger import BoxTrigger

WRIST_PATCH = 2  # 手腕取 (2*2+1)^2 邻域中位数
PERSON_BBOX_PAD = 0.12
MIN_BOX_PIXELS = 20


def resolve_video(paths, record_id: str) -> Path | None:
    for base in ("video", "video.backup"):
        p = paths.localdata / base / f"{record_id}.mp4"
        if p.is_file():
            return p
    return None


def baseline_frame_indices(paths, file_name: str) -> set[int] | None:
    clip = paths.baseline_manifest.parent / file_name
    if not clip.is_file():
        return None
    rows = json.loads(clip.read_text(encoding="utf-8"))
    return {int(r.get("frame_idx") or 0) for r in rows if isinstance(r, dict)}


def person_bbox(person: dict[str, Any], w: int, h: int) -> tuple[int, int, int, int] | None:
    pts = [
        kp
        for kp in (person.get("keypoints") or [])
        if isinstance(kp, (list, tuple)) and len(kp) > 2 and float(kp[2]) >= KPT_SCORE_MIN
    ]
    if len(pts) < 3:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    px, py = (x1 - x0) * PERSON_BBOX_PAD, (y1 - y0) * PERSON_BBOX_PAD
    return (
        max(0, int(x0 - px)),
        max(0, int(y0 - py)),
        min(w, int(x1 + px) + 1),
        min(h, int(y1 + py) + 1),
    )


def collect_jobs(paths, ref, record) -> dict[int, list[dict[str, Any]]]:
    """枚举与 build_pair_dataset 一致的配对，按帧号分组。"""
    bank = FeatureBank(
        infer_width=int(record.meta.get("infer_width") or 1),
        infer_height=int(record.meta.get("infer_height") or 1),
        video_fps=float(record.fps or 15.0),
    )
    trigger = BoxTrigger(record.boxes)
    by_key = {
        int(fr.get("source_frame_idx") or fr.get("frame_idx") or 0): fr for fr in record.frames
    }
    wanted = baseline_frame_indices(paths, ref.file_name)
    keys = sorted(wanted) if wanted is not None else sorted(by_key)

    jobs: dict[int, list[dict[str, Any]]] = {}
    for key in keys:
        frame = by_key.get(key)
        if frame is None:
            continue
        for row in bank.rows_for_frame(frame):
            person = row.get("_person")
            if not isinstance(person, dict):
                continue
            for hit in trigger.hits_for_person(person):
                jobs.setdefault(key, []).append(
                    {
                        "person_track_id": int(row.get("person_track_id") or 0),
                        "token": hit["token"],
                        "wrist_xy": hit["wrist_xy"],
                        "person": person,
                    }
                )
    return jobs


def box_masks(record, w: int, h: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for box in record.boxes:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [box.contour], 1)
        out[box.token] = mask.astype(bool)
    return out


def scalars_for_pair(
    depth: np.ndarray, job: dict[str, Any], mask: np.ndarray, p5: float, p95: float
) -> dict[str, Any]:
    h, w = depth.shape
    wx, wy = int(round(job["wrist_xy"][0])), int(round(job["wrist_xy"][1]))
    x0, x1 = max(0, wx - WRIST_PATCH), min(w, wx + WRIST_PATCH + 1)
    y0, y1 = max(0, wy - WRIST_PATCH), min(h, wy + WRIST_PATCH + 1)
    patch = depth[y0:y1, x0:x1]
    d_wrist = float(np.median(patch)) if patch.size else float("nan")

    box_full = depth[mask]
    d_box_full = float(np.median(box_full)) if box_full.size else float("nan")

    bbox = person_bbox(job["person"], w, h)
    ex_mask = mask.copy()
    if bbox is not None:
        ex_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = False
    box_ex = depth[ex_mask]
    if box_ex.size >= MIN_BOX_PIXELS:
        d_box_ex, valid = float(np.median(box_ex)), 1.0
    else:
        d_box_ex, valid = d_box_full, 0.0

    return {
        "frame_idx": job["frame_idx"],
        "person_track_id": job["person_track_id"],
        "token": job["token"],
        "d_wrist": round(d_wrist, 4),
        "d_box_full": round(d_box_full, 4),
        "d_box_ex": round(d_box_ex, 4),
        "box_ex_valid": valid,
        "p5": round(p5, 4),
        "p95": round(p95, 4),
    }


def process_record(estimator, paths, ref, out_dir: Path, batch_size: int) -> dict[str, Any]:
    record = load_record(ref, paths)
    infer_w = int(record.meta.get("infer_width") or ref.infer_width or 852)
    infer_h = int(record.meta.get("infer_height") or ref.infer_height or 480)

    jobs = collect_jobs(paths, ref, record)
    if not jobs:
        return {"record_id": ref.record_id, "status": "no_hits", "n_pairs": 0}

    video = resolve_video(paths, ref.record_id)
    if video is None:
        return {"record_id": ref.record_id, "status": "no_video", "n_pairs": 0}

    masks = box_masks(record, infer_w, infer_h)
    needed = sorted(jobs)
    needed_set = set(needed)
    last_needed = needed[-1]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"record_id": ref.record_id, "status": "video_open_failed", "n_pairs": 0}

    rows: list[dict[str, Any]] = []
    batch_frames: list[np.ndarray] = []
    batch_keys: list[int] = []

    def flush() -> None:
        if not batch_frames:
            return
        depths = estimator.infer_bgr(batch_frames)
        for depth, key in zip(depths, batch_keys):
            p5, p95 = np.percentile(depth, [5, 95])
            for job in jobs[key]:
                mask = masks.get(job["token"])
                if mask is None:
                    continue
                rows.append(
                    scalars_for_pair(depth, {**job, "frame_idx": key}, mask, float(p5), float(p95))
                )
        batch_frames.clear()
        batch_keys.clear()

    pos0 = 0
    try:
        while True:
            # frame_idx 与骨架/导出一致，从 1 起；OpenCV 是 0-based
            frame_idx = pos0 + 1
            if frame_idx > last_needed:
                break
            if frame_idx in needed_set:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame.shape[1] != infer_w or frame.shape[0] != infer_h:
                    frame = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                batch_frames.append(frame)
                batch_keys.append(frame_idx)
                if len(batch_frames) >= batch_size:
                    flush()
            else:
                if not cap.grab():
                    break
            pos0 += 1
        flush()
    finally:
        cap.release()

    safe = ref.record_id.replace("/", "__")
    (out_dir / f"{safe}.json").write_text(
        json.dumps({"record_id": ref.record_id, "rows": rows}, ensure_ascii=False),
        encoding="utf-8",
    )
    n_expected = sum(len(v) for v in jobs.values())
    return {
        "record_id": ref.record_id,
        "status": "ok",
        "n_pairs": len(rows),
        "n_expected": n_expected,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="预计算手-货框深度标量")
    ap.add_argument("--out", default=str(ROOT / "output/depth"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from depth.estimator import DepthEstimator

    paths = load_paths()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    estimator = DepthEstimator()
    refs = list_records(paths)
    if args.limit:
        refs = refs[: args.limit]

    summary = []
    t0 = time.time()
    for i, ref in enumerate(refs, 1):
        r = process_record(estimator, paths, ref, out_dir, args.batch_size)
        summary.append(r)
        print(
            f"[{i}/{len(refs)}] {ref.record_id.split('/')[-1]} "
            f"{r['status']} pairs={r['n_pairs']}/{r.get('n_expected', '?')} "
            f"({time.time() - t0:.0f}s)"
        )

    (out_dir / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(r["n_pairs"] for r in summary)
    print(f"\n完成：{total} 个配对深度标量 → {out_dir}  用时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
