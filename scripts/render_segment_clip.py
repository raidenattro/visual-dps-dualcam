"""把一个标真段渲染成 mp4，用于人工确认它是「一次拣货」还是「多次拣货被合并」。

画面上标当前帧有没有人工标注，底部一条分布条展示整段的标注空洞位置：
绿=该帧有标注，深灰=空洞。段内空洞多且成簇，说明这一段其实是多次拣货。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_paths
from adapters.record_reader import list_records_from_manifest, load_record, record_dir
from adapters.review_labels import normalize_verified_true
from scripts.build_tagged_manifest import find_review_key
from scripts.render_fn_fp_frames import _resolve_video
from scripts.render_missed_segments import _banner, _draw_boxes, _draw_person

BAR_H = 22


def annotated_frames(paths, record_id: str, tokens: list[str]) -> set[int]:
    """该 record 上、命中给定货框的人工标注帧号集合。"""
    found = find_review_key(paths, record_id)
    if not found:
        return set()
    want = set(tokens)
    out: set[int] = set()
    for e in normalize_verified_true(found[1].get("verified_true") or []):
        toks = set(e.get("confirmed_box_tokens") or [])
        if not (toks & want):
            continue
        fi = int(e.get("frame_idx") or e.get("source_frame_idx") or 0)
        if fi > 0:
            out.add(fi)
    return out


def draw_bar(img, *, start: int, end: int, cur: int, marked: set[int]) -> None:
    h, w = img.shape[:2]
    y0 = h - BAR_H
    cv2.rectangle(img, (0, y0), (w, h), (35, 35, 35), -1)
    span = max(1, end - start)
    for f in marked:
        if start <= f <= end:
            x = int((f - start) / span * (w - 1))
            cv2.line(img, (x, y0 + 4), (x, h - 6), (90, 220, 90), 1)
    x = int((cur - start) / span * (w - 1))
    cv2.line(img, (x, y0), (x, h), (255, 255, 255), 2)
    cv2.putText(img, "green = human-labeled frame", (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description="渲染标真段为 mp4")
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v1.json"))
    ap.add_argument("--seg-id", default="", help="精确 seg_id；留空则用 --record + --index")
    ap.add_argument("--record", default="", help="record_id 子串")
    ap.add_argument("--index", type=int, default=-1, help="该 record 内第几段（local_index）")
    ap.add_argument("--min-fill", type=float, default=-1.0,
                    help="批量模式：渲染填充率低于该值且跨度>=--min-span 的段")
    ap.add_argument("--min-span", type=int, default=300)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--pad", type=int, default=25, help="段前后各多渲染几帧作为上下文")
    ap.add_argument("--step", type=int, default=2, help="抽帧步长，2 表示隔帧取")
    ap.add_argument("--out", default=str(ROOT / "output/viz/segment_clips"))
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    segs = man["segments"]
    if args.seg_id:
        picked = [s for s in segs if s["seg_id"] == args.seg_id]
    elif args.min_fill > 0:
        cand = [
            s
            for s in segs
            if (s["frame_end"] - s["frame_start"] + 1) >= args.min_span
            and s["entry_count"] / (s["frame_end"] - s["frame_start"] + 1) < args.min_fill
        ]
        cand.sort(key=lambda s: s["entry_count"] / (s["frame_end"] - s["frame_start"] + 1))
        picked = cand[: args.limit]
    else:
        picked = [
            s
            for s in segs
            if args.record in s["record_id"] and (args.index < 0 or s["local_index"] == args.index)
        ][: args.limit]
    if not picked:
        print("[--] 没有匹配的段")
        return 1

    paths = load_paths()
    refs = {r.record_id: r for r in list_records_from_manifest(Path(args.manifest), split_role=None)}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seg in picked:
        rid = seg["record_id"]
        ref = refs.get(rid)
        if ref is None:
            print(f"[--] 找不到 record {rid}")
            continue
        record = load_record(ref, paths)
        video = _resolve_video(
            rid,
            json.loads((record_dir(ref, paths) / "manifest.json").read_text(encoding="utf-8")),
            paths,
        )
        if video is None:
            print(f"[--] 找不到视频 {rid}")
            continue

        marked = annotated_frames(paths, rid, list(seg["gt_tokens"]))
        by_f = {
            int(f.get("source_frame_idx") or f.get("frame_idx") or 0): f for f in record.frames
        }
        tracks = {str(t) for t in (seg.get("person_track_ids") or [])}
        gt_tokens = set(seg["gt_tokens"])
        start = max(1, seg["frame_start"] - args.pad)
        end = seg["frame_end"] + args.pad
        span = seg["frame_end"] - seg["frame_start"] + 1
        fill = seg["entry_count"] / span

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[--] 打不开视频 {video}")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start - 1))
        src_fps = float(record.fps or 25.0)
        name = f"{rid.split('/')[-1][:34]}_seg{seg['local_index']}_fill{int(fill * 100):02d}.mp4"
        writer = None
        n = 0
        for fidx in range(start, end + 1):
            ok, img = cap.read()
            if not ok:
                break
            if (fidx - start) % args.step:
                continue
            frame = by_f.get(fidx) or {}
            _draw_boxes(img, record.boxes, gt=gt_tokens, touched=set())
            for pr in frame.get("persons") or []:
                _draw_person(
                    img,
                    pr,
                    is_main=(not tracks) or str(pr.get("person_track_id")) in tracks,
                    wrist_min=0.15,
                )
            inside = seg["frame_start"] <= fidx <= seg["frame_end"]
            _banner(
                img,
                [
                    f"{name[:40]}  frame {fidx}",
                    f"segment {seg['frame_start']}-{seg['frame_end']}  "
                    f"({span} frames, {span / src_fps:.1f}s)  labeled {seg['entry_count']} frames = {fill:.0%}",
                    ("LABELED: picking " if fidx in marked else "no label      ")
                    + ("(inside segment)" if inside else "(context)")
                    + f"   {sorted(gt_tokens)}",
                ],
            )
            draw_bar(img, start=seg["frame_start"], end=seg["frame_end"], cur=fidx, marked=marked)
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_dir / name), cv2.VideoWriter_fourcc(*"mp4v"), src_fps / args.step, (w, h)
                )
            writer.write(img)
            n += 1
        cap.release()
        if writer is not None:
            writer.release()
        print(f"[ok] {name}  {n} 帧  填充率 {fill:.0%}  标注 {seg['entry_count']}/{span}")

    print(f"→ {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
