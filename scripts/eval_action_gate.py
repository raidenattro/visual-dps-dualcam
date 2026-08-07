#!/usr/bin/env python3
"""离线验证动作门控（方案 A）：先判人是否在做拣货动作，再允许货框告警。

训练标签：该帧该人是否落在任一真值拣货段内（不看碰的是哪个框）。
特征：过去 W 帧骨架序列统计量（复用 eval_temporal_action），不看货框。
评估：
  1) GroupKFold OOF AUC
  2) 门控后段级召回 / 误报
  3) 对人工审核过的误报：能压掉多少「没拣货」、会误伤多少「报错框」
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_paths
from pipeline.alarm import AlarmTracker
from scripts.eval_export import EVENT_MERGE_GAP, _token_match
from scripts.eval_temporal_action import collect_sequences, oof, window_features


def _load_gt(manifest: Path) -> dict[str, list[dict[str, Any]]]:
    man = json.loads(manifest.read_text(encoding="utf-8"))
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in man.get("segments") or []:
        by[str(s["record_id"])].append(s)
    return dict(by)


def _person_positive(segs: list[dict[str, Any]], frame: int, track: str) -> bool:
    for s in segs:
        if not (int(s["frame_start"]) <= frame <= int(s["frame_end"])):
            continue
        tracks = [str(t) for t in (s.get("person_track_ids") or []) if t]
        if not tracks or track in tracks:
            return True
    return False


def _fp_runs(
    alarms: list[tuple[int, str]], cover: list[dict[str, Any]]
) -> list[tuple[str, int, int]]:
    unc = [
        (f, t)
        for f, t in alarms
        if not any(
            int(s["frame_start"]) <= f <= int(s["frame_end"])
            and any(_token_match(t, g) for g in s["gt_tokens"])
            for s in cover
        )
    ]
    by: dict[str, list[int]] = defaultdict(list)
    for f, t in unc:
        by[t].append(f)
    out: list[tuple[str, int, int]] = []
    for t, fl in by.items():
        fl = sorted(set(fl))
        run = [fl[0]]
        for f in fl[1:]:
            if f - run[-1] <= EVENT_MERGE_GAP:
                run.append(f)
            else:
                out.append((t, run[0], run[-1]))
                run = [f]
        out.append((t, run[0], run[-1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="动作门控离线验证")
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v2.json"))
    ap.add_argument("--scores", default=str(ROOT / "output/scores/v5"))
    ap.add_argument("--audit", default=str(ROOT / "output/audit/fp_audit.json"))
    ap.add_argument("--cache", default=str(ROOT / "output/train/action_sequences_v2.json"))
    ap.add_argument("--window", type=int, default=30, help="过去窗口源帧数（≈2s@15fps）")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pair-threshold", type=float, default=0.20)
    ap.add_argument("--min-frames", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "output/sweep/v5_action_gate/report.md"))
    args = ap.parse_args()

    paths = load_paths()
    man_path = Path(args.manifest)
    gt = _load_gt(man_path)
    cache = Path(args.cache)
    if cache.is_file():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        seqs = {
            r: {int(t): {int(f): v for f, v in fr.items()} for t, fr in tr.items()}
            for r, tr in raw.items()
        }
        print(f"[..] 复用序列 {cache}")
    else:
        seqs = collect_sequences(man_path, paths)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(seqs, ensure_ascii=False), encoding="utf-8")
        print(f"[ok] 序列 → {cache}")

    # 人-帧样本：来自分数文件（有碰撞才需要门控）
    rows: list[dict[str, Any]] = []
    dumps: list[dict[str, Any]] = []
    for p in sorted(Path(args.scores).glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rid = d["record_id"]
        segs = gt.get(rid) or []
        if not segs:
            continue
        dumps.append(d)
        seen: set[tuple[int, str]] = set()
        for frame_idx, pairs in d["frames"]:
            fi = int(frame_idx)
            for x in pairs:
                track = str(x[0])
                key = (fi, track)
                if key in seen:
                    continue
                seen.add(key)
                frames = (seqs.get(rid) or {}).get(int(track)) or {}
                feat, cov = window_features(frames, fi, args.window, args.step)
                rows.append(
                    {
                        "record_id": rid,
                        "frame": fi,
                        "track": track,
                        "y": int(_person_positive(segs, fi, track)),
                        "feat": feat + [cov],
                    }
                )

    X = np.array([r["feat"] for r in rows], dtype=np.float64)
    y = np.array([r["y"] for r in rows], dtype=int)
    g = np.array([r["record_id"] for r in rows])
    print(f"人-帧样本 {len(y)}  正 {int(y.sum())}（{y.mean():.1%}）  窗口 {args.window} 步长 {args.step}")

    p_lr = oof(X, y, g, model="lr", folds=args.folds)
    p_gb = oof(X, y, g, model="gbdt", folds=args.folds)
    auc_lr = roc_auc_score(y, p_lr)
    auc_gb = roc_auc_score(y, p_gb)
    use = "gbdt" if auc_gb >= auc_lr else "lr"
    p_act = p_gb if use == "gbdt" else p_lr
    print(f"OOF AUC  逻辑回归 {auc_lr:.4f}  GBDT {auc_gb:.4f}  → 用 {use}")

    act_of = {
        (r["record_id"], r["frame"], r["track"]): float(p_act[i]) for i, r in enumerate(rows)
    }

    # 审核误报在峰值帧的动作分
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    by_verdict: dict[str, list[float]] = defaultdict(list)
    for e in audit.get("events") or []:
        if not e.get("verdict"):
            continue
        key = (e["record_id"], int(e["frame"]), str(e.get("person_track_id") or ""))
        sc = act_of.get(key)
        if sc is None:
            # 峰值帧可能无碰撞分数字典键——用区间内最大动作分
            sc = max(
                (
                    act_of[k]
                    for k in act_of
                    if k[0] == e["record_id"]
                    and k[2] == str(e.get("person_track_id") or "")
                    and int(e["span"][0]) <= k[1] <= int(e["span"][1])
                ),
                default=float("nan"),
            )
        if not np.isnan(sc):
            by_verdict[e["verdict"]].append(sc)

    print("\n审核误报的动作分（中位）：")
    for k in ("not_pick", "wrong_box", "missing"):
        a = by_verdict.get(k) or []
        if a:
            print(f"  {k:12s} n={len(a):3d}  中位 {np.median(a):.3f}  "
                  f"p25 {np.percentile(a,25):.3f}  p75 {np.percentile(a,75):.3f}")

    # 门控扫描：动作分 < t_act 则该人该帧所有框告警作废
    val_by = {
        rid: [s for s in segs if s.get("split") == "val"] for rid, segs in gt.items()
    }
    lines = [
        "# 动作门控离线验证",
        "",
        f"- 窗口 {args.window} 帧 / step {args.step}，模型 {use}",
        f"- OOF AUC 逻辑回归 {auc_lr:.4f} / GBDT {auc_gb:.4f}",
        f"- 配对阈值 {args.pair_threshold}，连续帧 {args.min_frames}",
        "",
        "| 动作门槛 | 召回 | 漏段 | 误报 | 审核没拣货被压 | 审核报错框被压 | 审核漏标被压 |",
        "|---------|------|------|------|----------------|----------------|--------------|",
    ]
    print(f"\n配对阈值 {args.pair_threshold} / 连续 {args.min_frames}，扫动作门槛：")
    print(f"{'动作门槛':>8} {'召回':>8} {'漏':>4} {'误报':>6}  "
          f"{'压没拣货':>8} {'压报错框':>8} {'压漏标':>6}")

    # 预计算每帧每人是否过配对阈值的 token
    best_rows = []
    for t_act in [0.0, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        n_val = n_det = 0
        n_fp = 0
        killed = Counter()
        judged_hits = {v: 0 for v in ("not_pick", "wrong_box", "missing")}
        judged_total = {v: len(by_verdict.get(v) or []) for v in judged_hits}

        for d in dumps:
            rid = d["record_id"]
            cover = gt.get(rid) or []
            vsegs = val_by.get(rid) or []
            tr = AlarmTracker(min_consecutive_frames=args.min_frames, cooldown_frames=0)
            alarms: list[tuple[int, str]] = []
            for frame_idx, pairs in d["frames"]:
                fi = int(frame_idx)
                toks: set[str] = set()
                for x in pairs:
                    track, tok, sm = str(x[0]), str(x[1]), float(x[3])
                    if sm < args.pair_threshold:
                        continue
                    a = act_of.get((rid, fi, track), 1.0)
                    if a < t_act:
                        continue
                    toks.add(tok)
                for tok in tr.step(sorted(toks), fi):
                    alarms.append((fi, tok))

            for s in vsegs:
                n_val += 1
                a, b = int(s["frame_start"]), int(s["frame_end"])
                if any(
                    a <= f <= b and any(_token_match(t, g) for g in s["gt_tokens"])
                    for f, t in alarms
                ):
                    n_det += 1
            n_fp += len(_fp_runs(alarms, cover))

        # 审核事件：峰值帧动作分 < t_act 算被压掉
        for e in audit.get("events") or []:
            v = e.get("verdict")
            if v not in judged_hits:
                continue
            key = (e["record_id"], int(e["frame"]), str(e.get("person_track_id") or ""))
            sc = act_of.get(key)
            if sc is None:
                sc = max(
                    (
                        act_of[k]
                        for k in act_of
                        if k[0] == e["record_id"]
                        and k[2] == str(e.get("person_track_id") or "")
                        and int(e["span"][0]) <= k[1] <= int(e["span"][1])
                    ),
                    default=1.0,
                )
            if sc < t_act:
                judged_hits[v] += 1

        recall = n_det / n_val if n_val else 0.0
        def pct(v: str) -> str:
            tot = judged_total[v]
            return f"{judged_hits[v]}/{tot}" if tot else "-"

        print(
            f"{t_act:8.2f} {recall:8.2%} {n_val-n_det:4d} {n_fp:6d}  "
            f"{pct('not_pick'):>8} {pct('wrong_box'):>8} {pct('missing'):>6}"
        )
        lines.append(
            f"| {t_act:.2f} | {recall:.2%} | {n_val-n_det} | {n_fp} | "
            f"{pct('not_pick')} | {pct('wrong_box')} | {pct('missing')} |"
        )
        best_rows.append(
            {
                "t_act": t_act,
                "recall": recall,
                "missed": n_val - n_det,
                "fp": n_fp,
                "kill_not_pick": judged_hits["not_pick"],
                "kill_wrong_box": judged_hits["wrong_box"],
                "kill_missing": judged_hits["missing"],
            }
        )

    # 正样本动作分分位 → 能滤掉多少负样本
    lines += ["", "## 按保住真拣货比例看过滤力", "", "| 保住正样本 | 动作门槛 | 滤掉负样本 |", "|-----------|---------|-----------|"]
    print("\n按保住真拣货比例：")
    for keep in (0.99, 0.97, 0.95, 0.90, 0.85):
        t = float(np.quantile(p_act[y == 1], 1 - keep))
        filt = float((p_act[y == 0] < t).mean())
        print(f"  保住 {keep:.0%} 正样本 → 门槛 {t:.3f} → 滤掉 {filt:.1%} 负样本")
        lines.append(f"| {keep:.0%} | {t:.3f} | {filt:.1%} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out.parent / "rows.json").write_text(
        json.dumps(
            {
                "auc_lr": auc_lr,
                "auc_gbdt": auc_gb,
                "model": use,
                "window": args.window,
                "audit_action_scores": {k: {
                    "n": len(v),
                    "median": float(np.median(v)),
                    "p25": float(np.percentile(v, 25)),
                    "p75": float(np.percentile(v, 75)),
                } for k, v in by_verdict.items() if v},
                "rows": best_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
