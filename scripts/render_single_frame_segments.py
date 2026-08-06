#!/usr/bin/env python3
"""渲染「只有 1 帧标注」的拣货段：标注帧前后各取两帧拼成一条，用来人工判断
这一帧到底是不是真在拣货（标注偷懒只点一帧），还是误标。

中间那格是被标注的帧（红边）。红框=标注说在拣的货框。
只读 collector；产物写本仓 output/viz/。
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
from adapters.record_reader import list_records_from_manifest, load_record, record_dir
from scripts.render_fn_fp_frames import _read_frame, _resolve_video
from scripts.render_missed_segments import _draw_boxes, _draw_person

OFFSETS = (-10, -5, 0, 5, 10)


def _tile(img, *, marked: bool, label: str) -> Any:
    h, w = img.shape[:2]
    scale = 360.0 / w
    small = cv2.resize(img, (360, int(h * scale)), interpolation=cv2.INTER_AREA)
    color = (60, 60, 255) if marked else (110, 110, 110)
    cv2.rectangle(small, (0, 0), (small.shape[1] - 1, small.shape[0] - 1), color, 3 if marked else 1)
    cv2.putText(small, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(small, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return small


def main() -> int:
    ap = argparse.ArgumentParser(description="渲染单帧标注段")
    ap.add_argument("--segments", default=str(ROOT / "output/train/single_frame_segments.json"))
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v1.json"))
    ap.add_argument("--out", default=str(ROOT / "output/viz/single_frame_segments"))
    ap.add_argument("--limit", type=int, default=0, help="每条 record 最多渲染几个，0=全部")
    args = ap.parse_args()

    paths = load_paths()
    segs = json.loads(Path(args.segments).read_text(encoding="utf-8"))
    by_rid: dict[str, list[dict]] = {}
    for s in segs:
        lst = by_rid.setdefault(str(s["record_id"]), [])
        if args.limit and len(lst) >= args.limit:
            continue
        lst.append(s)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for ref in list_records_from_manifest(Path(args.manifest)):
        mine = by_rid.get(ref.record_id)
        if not mine:
            continue
        record = load_record(ref, paths)
        by_f = {
            int(f.get("source_frame_idx") or f.get("frame_idx") or 0): f for f in record.frames
        }
        video = _resolve_video(
            ref.record_id,
            json.loads((record_dir(ref, paths) / "manifest.json").read_text(encoding="utf-8")),
            paths,
        )
        if not video:
            print(f"[skip] 找不到视频 {ref.clip_name}")
            continue

        iw = int(record.meta.get("infer_width") or ref.infer_width or 0)
        ih = int(record.meta.get("infer_height") or ref.infer_height or 0)

        for s in mine:
            fi0 = int(s["frames"][0])
            gt = set(s.get("tokens") or [])
            tracks = {str(t) for t in (s.get("tracks") or [])}
            tiles = []
            for off in OFFSETS:
                fi = fi0 + off
                img = _read_frame(video, fi)
                if img is None:
                    continue
                if iw and ih and (img.shape[1] != iw or img.shape[0] != ih):
                    img = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA)
                _draw_boxes(img, record.boxes, gt=gt, touched=set())
                for p in (by_f.get(fi) or {}).get("persons") or []:
                    if isinstance(p, dict):
                        _draw_person(
                            img, p,
                            is_main=(not tracks or str(p.get("person_track_id")) in tracks),
                            wrist_min=0.3,
                        )
                tag = "标注帧" if off == 0 else f"{off:+d}"
                tiles.append(_tile(img, marked=(off == 0), label=f"f{fi} {tag}"))
            if not tiles:
                continue
            hh = max(t.shape[0] for t in tiles)
            tiles = [
                cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                for t in tiles
            ]
            strip = np.hstack(tiles)
            head = np.zeros((26, strip.shape[1], 3), dtype=np.uint8)
            txt = f"{ref.clip_name[:44]}  frame {fi0}  box={','.join(sorted(gt))}  person=#{','.join(sorted(tracks)) or '?'}"
            cv2.putText(head, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            out = np.vstack([head, strip])

            name = f"{ref.clip_name[:30]}_f{fi0}.jpg"
            name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
            cv2.imwrite(str(out_root / name), out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            rows.append({
                "file": name, "record": ref.record_id, "clip": ref.clip_name,
                "frame": fi0, "tokens": sorted(gt), "tracks": sorted(tracks),
            })
        print(f"[ok] {ref.clip_name[:44]}  {len(mine)} 个")

    (out_root / "index.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n共 {len(rows)} 张 → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
