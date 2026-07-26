#!/usr/bin/env python3
"""从 accuracy_report 的漏报/误报列表出「画面+骨架+货框标注」截图。

只读 collector 视频 / pose / 标注；产物写本仓 output/viz/。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_paths
from adapters.record_reader import load_boxes, load_skeleton_frames, record_dir
from adapters.record_reader import RecordRef

# COCO17 骨架连线
COCO17_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
KPT_SCORE_MIN = 0.3
COLOR_BOX = (80, 200, 80)
COLOR_GT = (0, 220, 255)
COLOR_FP = (60, 60, 255)
COLOR_FN = (0, 140, 255)
COLOR_SKEL = (255, 180, 80)
COLOR_JOINT = (255, 255, 255)


def _resolve_video(record_id: str, manifest: dict[str, Any], paths) -> Path | None:
    for key in ("source", "source_video"):
        raw = manifest.get(key)
        if raw:
            p = Path(str(raw))
            if p.is_file():
                return p
    bak = paths.localdata / "video.backup" / f"{record_id}.mp4"
    if bak.is_file():
        return bak
    vid = paths.localdata / "video" / f"{record_id}.mp4"
    if vid.is_file():
        return vid
    return None


def _read_frame(video: Path, frame_idx: int) -> np.ndarray | None:
    """frame_idx 与 export/skeleton 一致（通常从 1 起）；OpenCV 按 0-based 读。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # 优先按 0-based；若越界再试 1-based 语义（frame_idx 本身）
        candidates = []
        if frame_idx >= 1:
            candidates.append(frame_idx - 1)
        candidates.append(frame_idx)
        for fi in candidates:
            if total > 0 and fi >= total:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fi))
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
        return None
    finally:
        cap.release()


def _persons_at(frames: list[dict], frame_idx: int) -> list[dict]:
    for fr in frames:
        key = int(fr.get("source_frame_idx") or fr.get("frame_idx") or 0)
        if key == frame_idx:
            return list(fr.get("persons") or [])
    # 邻近帧兜底（抽帧间隔导致精确帧无检测）
    best = None
    best_d = 10**9
    for fr in frames:
        key = int(fr.get("source_frame_idx") or fr.get("frame_idx") or 0)
        d = abs(key - frame_idx)
        if d < best_d:
            best_d = d
            best = fr
    if best is not None and best_d <= 2:
        return list(best.get("persons") or [])
    return []


def _draw_skeleton(img: np.ndarray, persons: list[dict]) -> None:
    for person in persons:
        kpts = person.get("keypoints") or []
        pts: list[tuple[float, float] | None] = []
        for i in range(17):
            if i >= len(kpts):
                pts.append(None)
                continue
            kp = kpts[i]
            if not isinstance(kp, (list, tuple)) or len(kp) < 2:
                pts.append(None)
                continue
            score = float(kp[2]) if len(kp) > 2 else 0.0
            if score < KPT_SCORE_MIN:
                pts.append(None)
            else:
                pts.append((float(kp[0]), float(kp[1])))
        for a, b in COCO17_EDGES:
            pa, pb = pts[a], pts[b]
            if pa is None or pb is None:
                continue
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), COLOR_SKEL, 2, cv2.LINE_AA)
        for p in pts:
            if p is None:
                continue
            cv2.circle(img, (int(p[0]), int(p[1])), 3, COLOR_JOINT, -1, cv2.LINE_AA)


def _draw_boxes(img: np.ndarray, boxes, *, highlight: set[str], highlight_color) -> None:
    for box in boxes:
        color = highlight_color if box.token in highlight else COLOR_BOX
        thickness = 3 if box.token in highlight else 1
        cv2.polylines(img, [box.contour], True, color, thickness, cv2.LINE_AA)
        if box.token in highlight:
            cx, cy = int(box.center[0]), int(box.center[1])
            cv2.putText(
                img, box.token, (cx - 20, max(16, cy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )


def _banner(img: np.ndarray, lines: list[str], color) -> None:
    y = 22
    for line in lines:
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def main() -> int:
    ap = argparse.ArgumentParser(description="渲染 FN/FP 截图")
    ap.add_argument(
        "--report",
        default=str(ROOT / "output/export/pickstate-rule-v0-prod-test/accuracy_report.json"),
    )
    ap.add_argument("--out", default=str(ROOT / "output/viz/rule-v0-fn-fp"))
    ap.add_argument("--kind", choices=("all", "fn", "fp"), default="all")
    args = ap.parse_args()

    paths = load_paths()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    fn_dir = out_root / "fn"
    fp_dir = out_root / "fp"
    fn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[str, Any] = {}
    n_fn = n_fp = n_skip = 0
    index_rows: list[dict] = []

    for clip in report.get("clips") or []:
        if clip.get("status") != "ok":
            continue
        record_id = str(clip.get("record_id") or "")
        diag = clip.get("diagnostics") or {}
        jobs: list[tuple[str, dict]] = []
        if args.kind in ("all", "fn"):
            for seg in diag.get("missed_segments") or []:
                jobs.append(("fn", seg))
        if args.kind in ("all", "fp"):
            for item in diag.get("false_alarms") or []:
                jobs.append(("fp", item))
        if not jobs:
            continue

        if record_id not in cache:
            ref = RecordRef(
                record_id=record_id,
                clip_name=str(clip.get("clip") or ""),
                camera_slug=str(clip.get("camera_slug") or ""),
                file_name=str(clip.get("upload_file") or ""),
                infer_width=0,
                infer_height=0,
            )
            rd = record_dir(ref, paths)
            manifest = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
            video = _resolve_video(record_id, manifest, paths)
            cache[record_id] = {
                "frames": load_skeleton_frames(rd),
                "boxes": load_boxes(rd),
                "video": video,
                "manifest": manifest,
            }
            if video is None:
                print(f"[skip-video] {record_id}")

        packed = cache[record_id]
        video: Path | None = packed["video"]
        if video is None:
            n_skip += len(jobs)
            continue

        for kind, item in jobs:
            if kind == "fn":
                frame_idx = int(item.get("seek_frame") or item.get("frame_start") or 0)
                tokens = {str(t) for t in (item.get("gt_tokens") or [])}
                label = str(item.get("label") or f"FN f{frame_idx}")
                color = COLOR_FN
                dest = fn_dir
            else:
                frame_idx = int(item.get("seek_frame") or item.get("frame_idx") or 0)
                tok = str(item.get("box_token") or "").strip()
                tokens = {tok} if tok else set()
                label = str(item.get("label") or f"FP f{frame_idx}")
                color = COLOR_FP
                dest = fp_dir

            frame = _read_frame(video, frame_idx)
            if frame is None:
                n_skip += 1
                print(f"[skip-frame] {record_id} {kind} f={frame_idx}")
                continue

            # 若视频分辨率与 pose 不一致，按 pose 尺寸缩放画面再画
            boxes = packed["boxes"]
            if boxes:
                # pose 坐标系：manifest infer size
                iw = int((packed["manifest"].get("infer_width") or 0) or frame.shape[1])
                ih = int((packed["manifest"].get("infer_height") or 0) or frame.shape[0])
                if frame.shape[1] != iw or frame.shape[0] != ih:
                    frame = cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_AREA)

            persons = _persons_at(packed["frames"], frame_idx)
            _draw_boxes(frame, boxes, highlight=tokens, highlight_color=color)
            _draw_skeleton(frame, persons)
            short_id = record_id.replace("/", "__")
            _banner(
                frame,
                [
                    f"{'漏报 FN' if kind == 'fn' else '误报 FP'} | {record_id.split('/')[-1]}",
                    f"frame={frame_idx} | {','.join(sorted(tokens)) or '-'}",
                    label[:80],
                ],
                color,
            )

            fname = _safe_name(
                f"{kind}_{short_id}_f{frame_idx}_{'_'.join(sorted(tokens)) or 'none'}.jpg"
            )
            out_path = dest / fname
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if kind == "fn":
                n_fn += 1
            else:
                n_fp += 1
            index_rows.append(
                {
                    "kind": kind,
                    "record_id": record_id,
                    "frame_idx": frame_idx,
                    "tokens": sorted(tokens),
                    "label": label,
                    "file": str(out_path.relative_to(out_root)),
                }
            )

        print(f"[ok] {record_id} jobs={len(jobs)}")

    index = {
        "report": str(args.report),
        "fn": n_fn,
        "fp": n_fp,
        "skipped": n_skip,
        "items": index_rows,
    }
    (out_root / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    # 简短 markdown 索引
    lines = [
        f"# FN/FP 截图（{Path(args.report).parent.name}）",
        "",
        f"- 漏报 FN：{n_fn}",
        f"- 误报 FP：{n_fp}",
        f"- 跳过：{n_skip}",
        "",
        "## 漏报",
        "",
    ]
    for row in index_rows:
        if row["kind"] != "fn":
            continue
        lines.append(f"- `{row['file']}` — {row['label']}")
    lines += ["", "## 误报（目录 `fp/`）", "", f"共 {n_fp} 张，见 `fp/` 与 `index.json`。", ""]
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n完成：FN={n_fn} FP={n_fp} skip={n_skip} → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
