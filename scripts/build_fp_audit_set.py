#!/usr/bin/env python3
"""按「误报类型 × 分数段」分层抽样，生成供人工审核的误报清单。

已确认高分区（该货框整条视频没被拣过、分数≥0.5）里 89% 是漏标，需要知道
其余类型和低分区的漏标率才能算出真实精确率。每层记下总数与抽样数，
审完可加权外推。

紧贴真值段边界的误报（BOUNDARY）不进审核：那是标注起止松紧，不是漏标。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KIND_DESC = {
    "NO_PICK": "该货框整条视频都没被标注过被拣",
    "WRONG_TIME": "该货框别的时间被拣过，这次报早/报晚",
    "WRONG_BOX": "同时刻确实有人在拣货，但报的是别的货框",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="生成分层抽样的误报审核清单")
    ap.add_argument("--errors", default=str(ROOT / "output/sweep/v5/errors.json"))
    ap.add_argument("--prev", default=str(ROOT / "output/audit/fp_audit.json"),
                    help="已有判定，按 record+货框+区间 继承")
    ap.add_argument("--out", default=str(ROOT / "output/audit/fp_audit_set.json"))
    ap.add_argument("--high", type=float, default=0.5, help="高低分区分界")
    ap.add_argument("--low-min", type=float, default=0.2, help="低分区下界")
    ap.add_argument("--n-high", type=int, default=40, help="每个类型在高分区抽多少")
    ap.add_argument("--n-low", type=int, default=35, help="每个类型在低分区抽多少")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = json.loads(Path(args.errors).read_text(encoding="utf-8"))
    fps = [f for f in data["false_positives"] if f["kind"] in KIND_DESC]

    prev: dict[tuple, str] = {}
    prev_path = Path(args.prev)
    if prev_path.is_file():
        old = json.loads(prev_path.read_text(encoding="utf-8"))
        for e in old.get("events") or []:
            if e.get("verdict"):
                prev[(e["record_id"], e["token"], int(e["span"][0]), int(e["span"][1]))] = e["verdict"]
        print(f"继承已有判定 {len(prev)} 条")

    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in fps:
        pk = float(f["peak"])
        if pk >= args.high:
            band = "high"
        elif pk >= args.low_min:
            band = "low"
        else:
            continue
        strata[(f["kind"], band)].append(f)

    rng = np.random.default_rng(args.seed)
    events: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []

    for (kind, band), items in sorted(strata.items()):
        items.sort(key=lambda x: -float(x["peak"]))
        want = args.n_high if band == "high" else args.n_low
        # 已判过的一律保留，剩下的随机抽满
        keyed = [
            (it, prev.get((it["record_id"], it["box_token"],
                           int(it["frames"][0]), int(it["frames"][1])), ""))
            for it in items
        ]
        done = [(it, v) for it, v in keyed if v]
        rest = [it for it, v in keyed if not v]
        take = max(0, want - len(done))
        if take and len(rest) > take:
            idx = rng.choice(len(rest), size=take, replace=False)
            picked = [rest[int(i)] for i in sorted(idx)]
        else:
            picked = rest[:take]

        chosen = [(it, v) for it, v in done] + [(it, "") for it in picked]
        chosen.sort(key=lambda x: -float(x[0]["peak"]))
        for it, verdict in chosen:
            events.append({
                "record_id": it["record_id"],
                "token": it["box_token"],
                "span": [int(it["frames"][0]), int(it["frames"][1])],
                "frame": int(it["peak_frame"]),
                "n": int(it["n_frames"]),
                "peak": float(it["peak"]),
                "kind": kind,
                "kind_desc": KIND_DESC[kind],
                "band": band,
                "person_track_id": str(it.get("person_track_id") or ""),
                "person_track_ids": [str(t) for t in (it.get("person_track_ids") or [])],
                "someone_picking_elsewhere": bool(it.get("someone_picking_elsewhere")),
                "verdict": verdict,
            })
        meta.append({
            "kind": kind, "band": band,
            "population": len(items), "sampled": len(chosen),
            "already_judged": len(done),
        })
        print(
            f"  {kind:11s} {band:4s}  总体 {len(items):5d}  抽样 {len(chosen):3d}"
            f"（已判 {len(done)}）"
        )

    events.sort(key=lambda e: (e["band"] != "high", -e["peak"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"strata": meta, "n_events": len(events), "events": events},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    n_todo = sum(1 for e in events if not e["verdict"])
    print(f"\n共 {len(events)} 个待审（其中 {n_todo} 个还没判）→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
