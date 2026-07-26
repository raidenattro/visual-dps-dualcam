#!/usr/bin/env python3
"""检查只读 collector 路径与 baseline manifest 是否可读。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_baseline_manifest, load_paths


def main() -> int:
    paths = load_paths()
    missing = paths.ensure_readable()
    print("collector_root:", paths.root)
    print("localdata:", paths.localdata)
    print("baseline_manifest:", paths.baseline_manifest)
    if missing:
        print("MISSING:")
        for m in missing:
            print(" ", m)
        return 1

    manifest = load_baseline_manifest(paths)
    records = manifest.get("records") or manifest.get("record_ids") or []
    # 兼容不同 manifest 形状
    if isinstance(records, list) and records and isinstance(records[0], dict):
        n = len(records)
        sample = records[0].get("record_id") or records[0].get("id")
    else:
        n = len(records)
        sample = records[0] if records else None

    params = manifest.get("params") or {}
    print("manifest_ok: records=", n, "sample=", sample)
    print(
        "params:",
        {
            "pose_frame_interval": params.get("pose_frame_interval"),
            "alarm_min_consecutive_frames": params.get("alarm_min_consecutive_frames"),
            "tags": params.get("tags"),
        },
    )

    pipe_cfg = ROOT / "configs" / "pipeline.baseline_rule_expert.json"
    cfg = json.loads(pipe_cfg.read_text(encoding="utf-8"))
    print("pipeline_config:", cfg.get("name"), "scorer=", (cfg.get("pick_state") or {}).get("scorer"))
    print("OK: read-only collector access verified; visual-dps untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
