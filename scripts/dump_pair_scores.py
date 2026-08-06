#!/usr/bin/env python3
"""落每帧每个「人-货框」对的打分，供 sweep_policy.py 离线重放判定策略。

分数只依赖模型与平滑，与阈值/连续帧/互斥无关，所以跑一次就能扫遍所有策略组合。
只读 collector；产物写本仓 output/。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_paths
from adapters.frame_sampling import sample_frame_indices
from adapters.record_reader import list_records_from_manifest, load_record
from pipeline.runner import PickStatePipeline, load_pipeline_config
from scripts.export_manifest28 import baseline_frame_indices


def main() -> int:
    ap = argparse.ArgumentParser(description="导出人-货框配对分数")
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v1.json"))
    ap.add_argument("--split-role", default="val")
    ap.add_argument("--out", required=True, help="输出目录（建议 output/scores/<run>）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--sample-fps",
        type=float,
        default=0.0,
        help="模拟线上抽帧的处理帧率（如 15）；0 表示逐帧不抽",
    )
    args = ap.parse_args()

    paths = load_paths()
    config = load_pipeline_config(args.config)
    man = Path(args.manifest)
    if not man.is_absolute():
        man = ROOT / man
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = list_records_from_manifest(man, split_role=args.split_role.strip() or None)
    if args.limit:
        refs = refs[: args.limit]

    t0 = time.time()
    n_pairs = 0
    for i, ref in enumerate(refs, 1):
        record = load_record(ref, paths)
        pipeline = PickStatePipeline(config)
        frames: list[list] = []

        def collect(frame_idx: int, result, _frames=frames) -> None:
            pairs = []
            for d in result.pick_decisions:
                # pairwise 模式下 person_track_id 是 "track|token"
                track, _, token = str(d.person_track_id).partition("|")
                if not token:
                    continue
                pairs.append([track, token, round(d.score_raw, 5), round(d.score_smooth, 5)])
            _frames.append([int(frame_idx), pairs])

        indices = baseline_frame_indices(paths, ref.file_name)
        if args.sample_fps > 0:
            base = indices if indices else {
                int(f.get("source_frame_idx") or f.get("frame_idx") or 0) for f in record.frames
            }
            indices = set(
                sample_frame_indices(
                    base, source_fps=float(record.fps or 25.0), target_fps=args.sample_fps
                )
            )
        pipeline.run_record(record, frame_indices=indices, on_frame=collect)
        n_pairs += sum(len(p) for _, p in frames)
        (out_dir / ref.file_name).write_text(
            json.dumps(
                {"record_id": ref.record_id, "file": ref.file_name, "frames": frames},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[{i}/{len(refs)}] {ref.record_id.split('/')[-1]} frames={len(frames)}")

    (out_dir / "_meta.json").write_text(
        json.dumps(
            {
                "config": config.get("name"),
                "model_path": ((config.get("pair_state") or {}).get("linear_expert") or {}).get(
                    "model_path"
                ),
                "score_smooth": (config.get("pair_state") or {}).get("score_smooth"),
                "manifest": str(man),
                "split_role": args.split_role,
                "sample_fps": args.sample_fps or None,
                "record_count": len(refs),
                "pair_count": n_pairs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n分数 → {out_dir}  记录 {len(refs)} 配对 {n_pairs}  用时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
