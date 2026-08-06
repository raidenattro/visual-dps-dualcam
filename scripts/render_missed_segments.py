#!/usr/bin/env python3
"""渲染漏报段：画面 + 骨架 + 货框 + 每只手腕的置信度，用来人工确认漏报根因。

红框=标注说人在拣的货框，黄框=手腕实际落进的货框，灰框=其他。
手腕画成圆圈并标出置信度，低于触发门槛的画成红色（表示被丢弃）。
只读 collector；产物写本仓 output/viz/。
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
from pipeline.box_trigger import BoxTrigger, _raw_wrist
from scripts.eval_export import _token_match
from scripts.render_fn_fp_frames import COCO17_EDGES, _read_frame, _resolve_video

WRIST_LEFT, WRIST_RIGHT = 9, 10
KPT_MIN = 0.3

C_GT = (60, 60, 255)        # 标注的货框：红
C_TOUCHED = (0, 210, 255)   # 手实际进的货框：黄
C_BOX = (90, 90, 90)        # 其他货框：灰
C_MAIN = (120, 255, 120)    # 标注的那个人：绿
C_OTHER = (170, 130, 90)    # 其他人：暗蓝
C_WRIST_OK = (0, 255, 0)
C_WRIST_LOW = (0, 80, 255)


def _draw_person(img, person: dict[str, Any], *, is_main: bool, wrist_min: float) -> None:
    kpts = person.get("keypoints") or []
    color = C_MAIN if is_main else C_OTHER
    pts: list[tuple[float, float] | None] = []
    for i in range(17):
        kp = kpts[i] if i < len(kpts) else None
        if not isinstance(kp, (list, tuple)) or len(kp) < 2:
            pts.append(None)
            continue
        sc = float(kp[2]) if len(kp) > 2 else 0.0
        pts.append((float(kp[0]), float(kp[1])) if sc >= KPT_MIN else None)
    for a, b in COCO17_EDGES:
        if pts[a] and pts[b]:
            cv2.line(img, tuple(map(int, pts[a])), tuple(map(int, pts[b])), color, 2, cv2.LINE_AA)
    for p in pts:
        if p:
            cv2.circle(img, tuple(map(int, p)), 2, color, -1, cv2.LINE_AA)

    track = person.get("person_track_id")
    anchor = pts[5] or pts[6] or pts[0] or next((p for p in pts if p), None)
    if anchor:
        cv2.putText(img, f"#{track}", (int(anchor[0]) - 10, int(anchor[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"#{track}", (int(anchor[0]) - 10, int(anchor[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 手腕单独标：即使低于门槛也画，好看清是不是被丢弃了
    for wi, side in ((WRIST_LEFT, "L"), (WRIST_RIGHT, "R")):
        raw = _raw_wrist(person, wi)
        if raw is None:
            continue
        wx, wy, sc = raw
        ok = sc >= wrist_min
        wc = C_WRIST_OK if ok else C_WRIST_LOW
        cv2.circle(img, (int(wx), int(wy)), 7, wc, 2, cv2.LINE_AA)
        txt = f"{side}{sc:.2f}" + ("" if ok else " DROP")
        cv2.putText(img, txt, (int(wx) + 9, int(wy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (int(wx) + 9, int(wy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, wc, 1, cv2.LINE_AA)


def _draw_boxes(img, boxes, *, gt: set[str], touched: set[str]) -> None:
    for box in boxes:
        if box.token in gt:
            color, th = C_GT, 3
        elif box.token in touched:
            color, th = C_TOUCHED, 2
        else:
            color, th = C_BOX, 1
        cv2.polylines(img, [box.contour], True, color, th, cv2.LINE_AA)
        if box.token in gt or box.token in touched:
            cx, cy = int(box.center[0]), int(box.center[1])
            cv2.putText(img, box.token, (cx - 24, max(14, cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, box.token, (cx - 24, max(14, cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _banner(img, lines: list[str]) -> None:
    h = 8 + 20 * len(lines)
    strip = img[:h].copy()
    cv2.rectangle(strip, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    img[:h] = cv2.addWeighted(strip, 0.72, img[:h], 0.28, 0)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (8, 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description="渲染漏报段用于人工确认")
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v1.json"))
    ap.add_argument("--wrist-min", type=float, default=0.3, help="判定漏报所用的手腕触发门槛")
    ap.add_argument("--relaxed-wrist-min", type=float, default=0.15, help="对照门槛，用于标注是否可救回")
    ap.add_argument("--out", default=str(ROOT / "output/viz/missed_segments"))
    ap.add_argument("--per-segment", type=int, default=1, help="每段渲染几帧")
    args = ap.parse_args()

    paths = load_paths()
    man_path = Path(args.manifest)
    if not man_path.is_absolute():
        man_path = ROOT / man_path
    man = json.loads(man_path.read_text(encoding="utf-8"))
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    by_rid: dict[str, list[dict]] = {}
    for s in man.get("segments") or []:
        if s.get("split") == "val":
            by_rid.setdefault(str(s["record_id"]), []).append(s)

    rows: list[dict[str, Any]] = []
    for ref in list_records_from_manifest(man_path, split_role="val"):
        segs = by_rid.get(ref.record_id) or []
        if not segs:
            continue
        record = load_record(ref, paths)
        strict = BoxTrigger(record.boxes, wrist_score_min=args.wrist_min)
        relaxed = BoxTrigger(record.boxes, wrist_score_min=args.relaxed_wrist_min)
        by_f = {
            int(f.get("source_frame_idx") or f.get("frame_idx") or 0): f for f in record.frames
        }

        video = _resolve_video(
            ref.record_id,
            json.loads((record_dir(ref, paths) / "manifest.json").read_text(encoding="utf-8")),
            paths,
        )

        for seg in segs:
            a, b = int(seg["frame_start"]), int(seg["frame_end"])
            gt = list(seg.get("gt_tokens") or [])
            tracks = {str(t) for t in (seg.get("person_track_ids") or [])}

            hit_strict = hit_relaxed = False
            touched: set[str] = set()
            best_wrist = -1.0
            frame_pick = None
            for fi in range(a, b + 1):
                fr = by_f.get(fi)
                if not fr:
                    continue
                for p in fr.get("persons") or []:
                    if not isinstance(p, dict):
                        continue
                    mine = not tracks or str(p.get("person_track_id")) in tracks
                    for wi in (WRIST_LEFT, WRIST_RIGHT):
                        raw = _raw_wrist(p, wi)
                        if raw and mine and raw[2] > best_wrist:
                            best_wrist = raw[2]
                            frame_pick = fi
                    for h in strict.hits_for_person(p):
                        if any(_token_match(h["token"], g) for g in gt):
                            hit_strict = True
                        elif mine:
                            touched.add(h["token"])
                    for h in relaxed.hits_for_person(p):
                        if any(_token_match(h["token"], g) for g in gt):
                            hit_relaxed = True
                        elif mine:
                            touched.add(h["token"])
            if hit_strict:
                continue

            if best_wrist < 0:
                cause = "NO_PERSON_OR_WRIST"
            elif best_wrist < args.wrist_min:
                cause = "WRIST_CONF_TOO_LOW"
            elif touched:
                cause = "HAND_IN_OTHER_BOX"
            else:
                cause = "HAND_NOT_IN_ANY_BOX"
            fixed = "RECOVERED_BY_0.15" if hit_relaxed else "STILL_MISSED"

            frames_to_draw = [frame_pick if frame_pick else (a + b) // 2]
            if args.per_segment > 1:
                frames_to_draw = sorted({a, (a + b) // 2, b})[: args.per_segment]

            for fi in frames_to_draw:
                img = _read_frame(video, fi) if video else None
                if img is None:
                    continue
                iw = int(record.meta.get("infer_width") or ref.infer_width or img.shape[1])
                ih = int(record.meta.get("infer_height") or ref.infer_height or img.shape[0])
                if img.shape[1] != iw or img.shape[0] != ih:
                    img = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA)

                _draw_boxes(img, record.boxes, gt=set(gt), touched=touched)
                for p in (by_f.get(fi) or {}).get("persons") or []:
                    if isinstance(p, dict):
                        _draw_person(
                            img, p,
                            is_main=(not tracks or str(p.get("person_track_id")) in tracks),
                            wrist_min=args.wrist_min,
                        )
                _banner(img, [
                    f"MISSED  {cause}  ({fixed})",
                    f"{ref.clip_name[:38]}  seg {a}-{b}  frame {fi}",
                    f"GT box (red) = {','.join(gt)}"
                    + (f"   hand actually in (yellow) = {','.join(sorted(touched))}" if touched else ""),
                    f"annotated person (green) = #{','.join(sorted(tracks)) or '?'}"
                    f"   best wrist conf = {best_wrist:.2f}   trigger min = {args.wrist_min}",
                ])

                name = f"{cause}_{ref.clip_name[:26]}_seg{a}-{b}_f{fi}.jpg"
                name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
                cv2.imwrite(str(out_root / name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                rows.append({
                    "file": name, "cause": cause, "recovered_by_relaxed": hit_relaxed,
                    "record": ref.record_id, "seg": f"{a}-{b}", "frame": fi,
                    "gt_tokens": gt, "tracks": sorted(tracks),
                    "best_wrist_conf": round(best_wrist, 3),
                    "hand_in_boxes": sorted(touched),
                })
        print(f"[ok] {ref.clip_name[:40]}")

    (out_root / "index.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from collections import Counter
    cc = Counter(r["cause"] for r in rows)
    rec = Counter(r["recovered_by_relaxed"] for r in rows)
    print(f"\n共渲染 {len(rows)} 张 → {out_root}")
    for k, v in cc.most_common():
        print(f"  {v:3}  {k}")
    print(f"  放宽到 {args.relaxed_wrist_min} 可救回 {rec[True]} 段，仍漏 {rec[False]} 段")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
