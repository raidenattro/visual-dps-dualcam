#!/usr/bin/env python3
"""渲染误报事件的画面（峰值帧前后各两帧），用来判断是漏标的真拣货还是真误报。

红框=被误报的货框，灰框=其他。只读 collector；产物写本仓 output/viz/。
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
from scripts.render_single_frame_segments import OFFSETS, _tile


def main() -> int:
    ap = argparse.ArgumentParser(description="渲染误报事件画面")
    ap.add_argument("--events", default=str(ROOT / "output/sweep/v5/no_pick_fp.json"))
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v2.json"))
    ap.add_argument("--out", default=str(ROOT / "output/viz/fp_no_pick"))
    ap.add_argument("--top", type=int, default=12, help="按峰值取前 N 个")
    args = ap.parse_args()

    paths = load_paths()
    evs = json.loads(Path(args.events).read_text(encoding="utf-8"))
    evs = sorted(evs, key=lambda e: -float(e["peak"]))[: args.top]
    by_rid: dict[str, list[dict]] = {}
    for e in evs:
        by_rid.setdefault(str(e["record_id"]), []).append(e)

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
            print(f"[skip] 无视频 {ref.clip_name}")
            continue
        iw = int(record.meta.get("infer_width") or ref.infer_width or 0)
        ih = int(record.meta.get("infer_height") or ref.infer_height or 0)

        for e in mine:
            f0 = int(e["frame"])
            tok = str(e["token"])
            tiles = []
            for off in OFFSETS:
                fi = f0 + off
                img = _read_frame(video, fi)
                if img is None:
                    continue
                if iw and ih and (img.shape[1] != iw or img.shape[0] != ih):
                    img = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA)
                _draw_boxes(img, record.boxes, gt={tok}, touched=set())
                for p in (by_f.get(fi) or {}).get("persons") or []:
                    if isinstance(p, dict):
                        _draw_person(img, p, is_main=True, wrist_min=0.15)
                tiles.append(
                    _tile(img, marked=(off == 0), label=f"f{fi}" + (" 峰值帧" if off == 0 else ""))
                )
            if not tiles:
                continue
            hh = max(t.shape[0] for t in tiles)
            tiles = [
                cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                for t in tiles
            ]
            strip = np.hstack(tiles)
            head = np.zeros((26, strip.shape[1], 3), dtype=np.uint8)
            txt = (
                f"误报 {tok}  峰值{e['peak']}  持续{e['n']}帧 f{e['span'][0]}-{e['span'][1]}  "
                f"{ref.clip_name[:36]}"
            )
            cv2.putText(head, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            img_out = np.vstack([head, strip])
            name = f"p{e['peak']}_{ref.clip_name[:24]}_{tok}_f{f0}.jpg"
            name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
            cv2.imwrite(str(out_root / name), img_out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            rows.append({"file": name, **e})
        print(f"[ok] {ref.clip_name[:40]}  {len(mine)} 个")

    (out_root / "index.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n共 {len(rows)} 张 → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
