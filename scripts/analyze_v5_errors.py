#!/usr/bin/env python3
"""在指定工作点上归因漏报和误报，回答「该往哪投入」。

漏报分三类：上游没给出任何人-货框配对（NO_PAIR）、给了但分数够不到阈值（LOW_SCORE）、
分数够但连续帧数不足（SHORT_RUN）。
误报分四类：标注边界（BOUNDARY）、同时刻有人在拣别的框（WRONG_BOX）、
同一货框在别的时间被拣过（WRONG_TIME）、完全无人拣货（NO_PICK）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.alarm import AlarmTracker
from scripts.eval_export import EVENT_MERGE_GAP, _token_match

BOUNDARY_TOL = 4


def _runs(frames: list[int], gap: int) -> list[list[int]]:
    out: list[list[int]] = []
    for f in sorted(set(frames)):
        if out and f - out[-1][-1] <= gap:
            out[-1].append(f)
        else:
            out.append([f])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="v5 错误归因")
    ap.add_argument("--scores", default=str(ROOT / "output/scores/v5"))
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v2.json"))
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--min-frames", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "output/sweep/v5/errors.json"))
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    val_by_rid: dict[str, list[dict]] = {}
    all_by_rid: dict[str, list[dict]] = {}
    for s in man["segments"]:
        all_by_rid.setdefault(s["record_id"], []).append(s)
        if s["split"] == "val":
            val_by_rid.setdefault(s["record_id"], []).append(s)

    missed: list[dict[str, Any]] = []
    fps: list[dict[str, Any]] = []
    n_val = n_det = 0

    for p in sorted(Path(args.scores).glob("*.json")):
        if p.name.startswith("_"):
            continue
        dump = json.loads(p.read_text(encoding="utf-8"))
        rid = dump["record_id"]
        vsegs = val_by_rid.get(rid) or []
        asegs = all_by_rid.get(rid) or []
        if not vsegs:
            continue

        # 帧 -> 该帧所有配对（token, raw, smooth, person_track_id）
        by_frame: dict[int, list[tuple[str, float, float, str]]] = {}
        for frame_idx, pairs in dump["frames"]:
            by_frame[int(frame_idx)] = [
                (str(x[1]), float(x[2]), float(x[3]), str(x[0])) for x in pairs
            ]

        tracker = AlarmTracker(min_consecutive_frames=args.min_frames, cooldown_frames=0)
        alarms: list[tuple[int, str]] = []
        for frame_idx, pairs in dump["frames"]:
            toks = sorted({str(x[1]) for x in pairs if float(x[3]) >= args.threshold})
            for tok in tracker.step(toks, int(frame_idx)):
                alarms.append((int(frame_idx), tok))

        # ---- 漏报归因 ----
        for seg in vsegs:
            n_val += 1
            a, b = int(seg["frame_start"]), int(seg["frame_end"])
            toks = seg["gt_tokens"]
            hit = any(a <= f <= b and any(_token_match(t, g) for g in toks) for f, t in alarms)
            if hit:
                n_det += 1
                continue
            best_smooth = best_raw = -1.0
            n_pair = n_over = 0
            for f in range(a, b + 1):
                for tok, raw, sm, _tr in by_frame.get(f) or []:
                    if not any(_token_match(tok, g) for g in toks):
                        continue
                    n_pair += 1
                    best_smooth = max(best_smooth, sm)
                    best_raw = max(best_raw, raw)
                    if sm >= args.threshold:
                        n_over += 1
            if n_pair == 0:
                cause = "NO_PAIR"
            elif n_over == 0:
                cause = "LOW_SCORE"
            else:
                cause = "SHORT_RUN"
            missed.append({
                "record_id": rid, "seg_id": seg["seg_id"], "frames": [a, b],
                "span": b - a + 1, "entry_count": seg["entry_count"],
                "gt_tokens": toks, "person_track_ids": seg.get("person_track_ids") or [],
                "cause": cause, "n_pair_frames": n_pair,
                "n_frames_over_threshold": n_over,
                "best_smooth": round(best_smooth, 3), "best_raw": round(best_raw, 3),
            })

        # ---- 误报归因 ----
        uncovered: list[tuple[int, str]] = []
        for f, tok in alarms:
            if not any(
                int(s["frame_start"]) <= f <= int(s["frame_end"])
                and any(_token_match(tok, g) for g in s["gt_tokens"])
                for s in asegs
            ):
                uncovered.append((f, tok))

        by_tok: dict[str, list[int]] = {}
        for f, tok in uncovered:
            by_tok.setdefault(tok, []).append(f)
        for tok, flist in by_tok.items():
            own = [
                (int(s["frame_start"]), int(s["frame_end"]))
                for s in asegs
                if any(_token_match(tok, g) for g in s["gt_tokens"])
            ]
            for run in _runs(flist, EVENT_MERGE_GAP):
                s0, s1 = run[0], run[-1]

                def _best(f: int, tk: str = tok) -> tuple[float, str]:
                    """该帧里这个货框分数最高的配对，返回 (分数, 人物ID)。"""
                    cands = [(sm, tr) for t2, _r, sm, tr in by_frame.get(f) or [] if t2 == tk]
                    return max(cands, default=(0.0, ""))

                peak_frame = max(run, key=lambda f: _best(f)[0])
                peak, peak_track = _best(peak_frame)
                tracks = sorted({tr for f in run if (tr := _best(f)[1])})
                near = min(
                    (max(0, max(a - s1, s0 - b)) for a, b in own), default=10**9
                )
                # 同时刻是否有人在拣别的框
                other = any(
                    int(s["frame_start"]) <= s1 and s0 <= int(s["frame_end"])
                    and not any(_token_match(tok, g) for g in s["gt_tokens"])
                    for s in asegs
                )
                if near <= BOUNDARY_TOL:
                    kind = "BOUNDARY"
                elif other:
                    kind = "WRONG_BOX"
                elif own:
                    kind = "WRONG_TIME"
                else:
                    kind = "NO_PICK"
                fps.append({
                    "record_id": rid, "box_token": tok, "frames": [s0, s1],
                    "n_frames": len(run), "kind": kind,
                    "gap_to_gt": None if near >= 10**9 else near,
                    "peak": round(peak, 3), "peak_frame": peak_frame,
                    "person_track_id": peak_track, "person_track_ids": tracks,
                    "someone_picking_elsewhere": other,
                })

    # ---- 汇总 ----
    print(f"工作点：阈值 {args.threshold}  连续帧 {args.min_frames}")
    print(f"召回 {n_det}/{n_val} = {n_det/n_val:.2%}   漏 {len(missed)} 段\n")

    print("漏报原因：")
    for k, v in Counter(m["cause"] for m in missed).most_common():
        desc = {
            "NO_PAIR": "上游根本没给出手进这个框的配对（姿态/跟踪问题）",
            "LOW_SCORE": "有配对但模型分数够不到阈值（模型问题）",
            "SHORT_RUN": "分数够了但连续帧不足（策略问题）",
        }[k]
        print(f"  {v:3d}  {k:10s} {desc}")

    print(f"\n漏报明细（{len(missed)} 段）：")
    for m in sorted(missed, key=lambda x: x["cause"]):
        print(
            f"  {m['cause']:10s} {m['record_id'].split('/')[-1][:30]:32s} "
            f"f{m['frames'][0]}-{m['frames'][1]} ({m['span']:3d}帧) {','.join(m['gt_tokens']):20s} "
            f"配对{m['n_pair_frames']:3d}帧 最高分{m['best_smooth']:.2f}"
        )

    print(f"\n误报 {len(fps)} 次，原因：")
    for k, v in Counter(f["kind"] for f in fps).most_common():
        desc = {
            "BOUNDARY": "紧贴真值段边界（标注起止松紧问题）",
            "WRONG_BOX": "同时刻确实有人在拣货，但报错了货框",
            "WRONG_TIME": "该货框别的时间被拣过，这次报早/报晚了",
            "NO_PICK": "这个货框整条视频都没被拣过，纯误报",
        }[k]
        print(f"  {v:5d} ({v/len(fps):5.1%})  {k:10s} {desc}")

    lens = Counter(min(f["n_frames"], 10) for f in fps)
    print("\n误报事件长度（帧，10=10及以上）：")
    for k in sorted(lens):
        print(f"  {k:2d}帧 {lens[k]:5d} ({lens[k]/len(fps):5.1%})")
    short = sum(v for k, v in lens.items() if k <= 3)
    print(f"  ≤3帧的短脉冲共 {short} 次（{short/len(fps):.1%}）")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "threshold": args.threshold, "min_frames": args.min_frames,
            "n_val_segments": n_val, "detected": n_det,
            "missed": missed, "false_positives": fps,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
