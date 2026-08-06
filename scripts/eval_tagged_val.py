#!/usr/bin/env python3
"""按 tagged manifest 的 val 段评估导出包（段/事件口径）。

召回/漏报只看 val 段；误报在含 val 段的 record 上统计未覆盖告警事件。
只读 collector；产物写导出目录。
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

from scripts.eval_export import (
    GtSegment,
    _extract_alarms,
    _fp_events,
    _pct,
    _token_match,
)


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def eval_record(
    frames: list[dict[str, Any]],
    val_segs: list[dict[str, Any]],
    all_segs: list[dict[str, Any]],
) -> dict[str, Any]:
    """val_segs 用于召回；all_segs 用于判定告警是否被任一真值段覆盖（含 train，避免把 train 真值当误报）。"""
    alarms = _extract_alarms(frames)
    gt = [
        GtSegment(
            gt_tokens=tuple(s.get("gt_tokens") or []),
            frame_start=int(s["frame_start"]),
            frame_end=int(s["frame_end"]),
            entry_count=int(s.get("entry_count") or 0),
        )
        for s in val_segs
    ]
    cover = [
        GtSegment(
            gt_tokens=tuple(s.get("gt_tokens") or []),
            frame_start=int(s["frame_start"]),
            frame_end=int(s["frame_end"]),
            entry_count=int(s.get("entry_count") or 0),
        )
        for s in all_segs
    ]

    detected = missed = 0
    missed_samples: list[dict[str, Any]] = []
    for seg in gt:
        toks = list(seg.gt_tokens)
        tracks = None  # 告警侧无 track；检出仍按同框
        ok = any(
            seg.frame_start <= frame <= seg.frame_end and any(_token_match(tok, g) for g in toks)
            for frame, tok in alarms
        )
        # 若段带 track，仍只要求框命中（导出告警不带 person）
        _ = tracks
        if ok:
            detected += 1
        else:
            missed += 1
            if len(missed_samples) < 30:
                missed_samples.append(
                    {
                        "frame_start": seg.frame_start,
                        "frame_end": seg.frame_end,
                        "gt_tokens": toks,
                    }
                )

    uncovered: list[tuple[int, str]] = []
    for frame, tok in alarms:
        covered = any(
            seg.frame_start <= frame <= seg.frame_end
            and any(_token_match(tok, g) for g in seg.gt_tokens)
            for seg in cover
        )
        if not covered:
            uncovered.append((frame, tok))

    fp_ev = _fp_events(uncovered, cover)
    total = len(gt)
    algo = fp_ev["fp_events_algorithm"]
    return {
        "gt_segments": total,
        "detected": detected,
        "missed": missed,
        "recall": round(detected / total, 4) if total else None,
        "fp_events": fp_ev["fp_events"],
        "fp_events_algorithm": algo,
        "fp_events_boundary": fp_ev["fp_events_boundary"],
        "missed_segments": missed_samples,
        "fp_event_samples": fp_ev["fp_event_samples"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="tagged val 段评估")
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v1.json"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pkg = Path(args.pkg)
    if not pkg.is_absolute():
        pkg = ROOT / pkg
    man_path = Path(args.manifest)
    if not man_path.is_absolute():
        man_path = ROOT / man_path
    man = _load_manifest(man_path)

    by_rid: dict[str, list[dict[str, Any]]] = {}
    for s in man.get("segments") or []:
        by_rid.setdefault(str(s["record_id"]), []).append(s)

    clips: list[dict[str, Any]] = []
    for p in sorted(pkg.glob("*.json")):
        if p.name.startswith("_") or p.name.startswith("eval_") or p.name.startswith("accuracy_"):
            continue
        frames = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(frames, list) or not frames:
            continue
        rid = str(frames[0].get("record_id") or "")
        segs = by_rid.get(rid) or []
        val_segs = [s for s in segs if s.get("split") == "val"]
        if not val_segs:
            continue
        metrics = eval_record(frames, val_segs, segs)
        clips.append(
            {
                "status": "ok",
                "upload_file": p.name,
                "record_id": rid,
                "n_val_segments": len(val_segs),
                "n_all_segments": len(segs),
                "segment_val": metrics,
            }
        )

    ok = [c for c in clips if c.get("status") == "ok"]
    gt = sum(c["segment_val"]["gt_segments"] for c in ok)
    det = sum(c["segment_val"]["detected"] for c in ok)
    miss = sum(c["segment_val"]["missed"] for c in ok)
    fp = sum(c["segment_val"]["fp_events"] for c in ok)
    fp_a = sum(c["segment_val"]["fp_events_algorithm"] for c in ok)
    fp_b = sum(c["segment_val"]["fp_events_boundary"] for c in ok)
    summary = {
        "evaluated": len(ok),
        "gt_segments": gt,
        "detected": det,
        "missed": miss,
        "recall": round(det / gt, 4) if gt else None,
        "fp_events": fp,
        "fp_events_algorithm": fp_a,
        "fp_events_boundary": fp_b,
        "manifest": str(man_path),
        "split": "val_segments",
    }
    result = {"summary": summary, "clips": clips}
    out = Path(args.out) if args.out else pkg / "eval_tagged_val.json"
    if not out.is_absolute():
        out = ROOT / out
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"【val段】召回 {_pct(summary.get('recall'))}  "
        f"漏段 {miss}/{gt}  "
        f"误报事件 {fp}（算法 {fp_a} / 边界 {fp_b}）"
    )
    print(f"报告: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
