#!/usr/bin/env python3
"""对 28-clip 跑本仓 pipeline，导出与 collector 评估器兼容的 upload JSON。

只读 collector；产物写本仓 output/。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_paths
from adapters.record_reader import list_records, list_records_from_manifest, load_record
from pipeline.runner import PickStatePipeline, load_pipeline_config


def baseline_frame_indices(paths, file_name: str) -> set[int] | None:
    """复用 baseline 导出包的帧号，确保与对照包逐帧对齐。"""
    clip = paths.baseline_manifest.parent / file_name
    if not clip.is_file():
        return None
    rows = json.loads(clip.read_text(encoding="utf-8"))
    return {int(r.get("frame_idx") or 0) for r in rows if isinstance(r, dict)}


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 pick-state pipeline 推测结果")
    ap.add_argument("--config", default=str(ROOT / "configs" / "pipeline.baseline_rule_expert.json"))
    ap.add_argument("--out", required=True, help="输出目录（建议 output/export/<run>）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试）")
    ap.add_argument("--manifest", default="", help="本仓 tagged manifest；指定后按 --split-role 选 record")
    ap.add_argument(
        "--split-role",
        default="val",
        help="配合 --manifest：val=含 val 段的 record；train；空=全部",
    )
    args = ap.parse_args()

    paths = load_paths()
    config = load_pipeline_config(args.config)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest.strip():
        man = Path(args.manifest)
        if not man.is_absolute():
            man = ROOT / man
        role = args.split_role.strip() or None
        refs = list_records_from_manifest(man, split_role=role)
    else:
        refs = list_records(paths)
    if args.limit:
        refs = refs[: args.limit]

    records_meta: list[dict] = []
    errors = 0
    t0 = time.time()

    for i, ref in enumerate(refs, 1):
        try:
            record = load_record(ref, paths)
            indices = baseline_frame_indices(paths, ref.file_name)
            pipeline = PickStatePipeline(config)
            rows = pipeline.run_record(record, frame_indices=indices)
            (out_dir / ref.file_name).write_text(
                json.dumps(rows, ensure_ascii=False), encoding="utf-8"
            )
            picking = sum(1 for r in rows if r["is_picking"])
            records_meta.append(
                {
                    "status": "ok",
                    "record_id": ref.record_id,
                    "clip_name": ref.clip_name,
                    "camera_slug": ref.camera_slug,
                    "file": ref.file_name,
                    "frame_count_exported": len(rows),
                    "picking_frame_count": picking,
                    "box_count": len(record.boxes),
                }
            )
            print(f"[{i}/{len(refs)}] {ref.record_id} frames={len(rows)} picking={picking}")
        except Exception as exc:  # noqa: BLE001 - 单条失败不阻断整批
            errors += 1
            records_meta.append(
                {"status": "error", "record_id": ref.record_id, "file": ref.file_name, "error": str(exc)}
            )
            print(f"[{i}/{len(refs)}] {ref.record_id} ERROR: {exc}")

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(paths.baseline_manifest),
        "params": {
            "baseline_type": "pick_state_pipeline",
            "pipeline_config": config.get("name"),
            "scorer": (config.get("pick_state") or {}).get("scorer"),
            "pose_tier": "rtmpose-m",
            "pose_frame_interval": config.get("pose_frame_interval"),
            "alarm_min_consecutive_frames": (config.get("alarm") or {}).get("min_consecutive_frames"),
            "alarm_cooldown_frames": (config.get("alarm") or {}).get("cooldown_frames"),
            "require_pick_state": (config.get("box_trigger") or {}).get("require_pick_state"),
            "writeback_timeline": False,
            "readonly_collector": True,
        },
        "record_count": len(refs),
        "exported_count": len(refs) - errors,
        "error_count": errors,
        "records": records_meta,
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n导出完成：{out_dir}  ok={len(refs) - errors} err={errors} 用时 {time.time() - t0:.1f}s")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
