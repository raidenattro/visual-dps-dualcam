#!/usr/bin/env python3
"""在 dump_pair_scores.py 的分数上离线重放判定策略，扫出召回/误报取舍表。

三个维度：
  阈值        分数到多少算这只手在拿货
  连续帧      连续命中多少帧才报警
  互斥        一个人同帧碰到多个框时，是否只报分数最高的那个

不重跑推理，因此可以一次扫遍全部组合。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.alarm import AlarmTracker
from scripts.eval_tagged_val import eval_record


def replay(
    dump: dict[str, Any], *, threshold: float, min_frames: int, exclusive: bool
) -> list[dict[str, Any]]:
    tracker = AlarmTracker(min_consecutive_frames=min_frames, cooldown_frames=0)
    rid = dump["record_id"]
    out: list[dict[str, Any]] = []
    for frame_idx, pairs in dump["frames"]:
        if exclusive and pairs:
            best: dict[str, list] = {}
            for p in pairs:
                cur = best.get(p[0])
                if cur is None or p[3] > cur[3]:
                    best[p[0]] = p
            pairs = list(best.values())
        tokens = sorted({p[1] for p in pairs if p[3] >= threshold})
        alarms = tracker.step(tokens, int(frame_idx))
        out.append(
            {
                "record_id": rid,
                "frame_idx": int(frame_idx),
                "is_picking": bool(alarms),
                "rule_alarm_collisions": alarms,
            }
        )
    return out


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="离线扫描判定策略")
    ap.add_argument("--scores", required=True, help="dump_pair_scores.py 的输出目录")
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v1.json"))
    ap.add_argument("--thresholds", default="0.21,0.24,0.27,0.30,0.35,0.40,0.45,0.50")
    ap.add_argument("--min-frames", default="2,3,4,5")
    ap.add_argument("--exclusive", default="both", choices=["no", "yes", "both"])
    ap.add_argument("--out-dir", default=str(ROOT / "output/sweep"))
    args = ap.parse_args()

    scores_dir = Path(args.scores)
    if not scores_dir.is_absolute():
        scores_dir = ROOT / scores_dir
    man_path = Path(args.manifest)
    if not man_path.is_absolute():
        man_path = ROOT / man_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    man = json.loads(man_path.read_text(encoding="utf-8"))
    by_rid: dict[str, list[dict[str, Any]]] = {}
    for s in man.get("segments") or []:
        by_rid.setdefault(str(s["record_id"]), []).append(s)

    dumps = []
    for p in sorted(scores_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        segs = by_rid.get(str(d.get("record_id"))) or []
        if not any(s.get("split") == "val" for s in segs):
            continue
        dumps.append((d, [s for s in segs if s.get("split") == "val"], segs))
    if not dumps:
        raise SystemExit(f"{scores_dir} 里没有含 val 段的记录")

    excl_opts = {"no": [False], "yes": [True], "both": [False, True]}[args.exclusive]
    rows: list[dict[str, Any]] = []
    for threshold in _floats(args.thresholds):
        for min_frames in _ints(args.min_frames):
            for exclusive in excl_opts:
                gt = det = miss = fp = fp_a = fp_b = 0
                for dump, val_segs, all_segs in dumps:
                    frames = replay(
                        dump, threshold=threshold, min_frames=min_frames, exclusive=exclusive
                    )
                    m = eval_record(frames, val_segs, all_segs)
                    gt += m["gt_segments"]
                    det += m["detected"]
                    miss += m["missed"]
                    fp += m["fp_events"]
                    fp_a += m["fp_events_algorithm"]
                    fp_b += m["fp_events_boundary"]
                rows.append(
                    {
                        "threshold": threshold,
                        "min_frames": min_frames,
                        "exclusive": exclusive,
                        "gt_segments": gt,
                        "detected": det,
                        "missed": miss,
                        "recall": round(det / gt, 4) if gt else None,
                        "fp_events": fp,
                        "fp_events_algorithm": fp_a,
                        "fp_events_boundary": fp_b,
                    }
                )
                print(
                    f"阈值 {threshold:.2f}  连续帧 {min_frames}  互斥 {'是' if exclusive else '否'}"
                    f"   召回 {det / gt:6.2%}  漏段 {miss:3}  误报 {fp:4}（算法 {fp_a}）"
                )

    (out_dir / "sweep.json").write_text(
        json.dumps({"scores": str(scores_dir), "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# 判定策略扫描（{scores_dir.name}）",
        "",
        f"- val 段总数：{rows[0]['gt_segments']}",
        "",
        "| 阈值 | 连续帧 | 互斥 | 召回 | 漏段 | 误报事件 | 其中算法 |",
        "|------|--------|------|------|------|----------|----------|",
    ]
    for r in sorted(rows, key=lambda r: (r["fp_events_algorithm"], -(r["recall"] or 0))):
        lines.append(
            f"| {r['threshold']:.2f} | {r['min_frames']} | {'是' if r['exclusive'] else '否'} "
            f"| {r['recall']:.2%} | {r['missed']} | {r['fp_events']} | {r['fp_events_algorithm']} |"
        )
    lines.append("")
    (out_dir / "sweep.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n扫描表 → {out_dir / 'sweep.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
