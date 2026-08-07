#!/usr/bin/env python3
"""离线验证邻框消歧（方案 B）：用进框深度 / 到框心距离卡住「手擦过邻框」。

审核显示 wrong_box 里 92% 同帧只碰一个框（互斥无效），主因是人在拣 A、手腕落在邻框 B。
正样本 vs 报错框：depth_ratio 中位 0.58 vs 0.32，center_dist_norm 0.90 vs 1.55。

可选叠加速度作门控（复用 action_sequences + 同窗口 GBDT OOF）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_paths
from adapters.record_reader import list_records_from_manifest, load_record
from features.box_geometry import compute_pair_features
from pipeline.alarm import AlarmTracker
from pipeline.box_trigger import BoxTrigger
from scripts.eval_action_gate import _fp_runs, _load_gt, _person_positive
from scripts.eval_export import _token_match
from scripts.eval_temporal_action import window_features


def _oof_gbdt(X: np.ndarray, y: np.ndarray, g: np.ndarray, folds: int = 5) -> np.ndarray:
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, g):
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_depth=6, random_state=0
        )
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="邻框消歧离线验证")
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v2.json"))
    ap.add_argument("--scores", default=str(ROOT / "output/scores/v5"))
    ap.add_argument("--audit", default=str(ROOT / "output/audit/fp_audit.json"))
    ap.add_argument("--seq-cache", default=str(ROOT / "output/train/action_sequences_v2.json"))
    ap.add_argument("--feat-cache", default=str(ROOT / "output/sweep/v5_box_gate/pair_feats.json"))
    ap.add_argument("--pair-threshold", type=float, default=0.20)
    ap.add_argument("--min-frames", type=int, default=1)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--with-action", action="store_true", help="叠加速度作门控")
    ap.add_argument("--out", default=str(ROOT / "output/sweep/v5_box_gate/report.md"))
    args = ap.parse_args()

    paths = load_paths()
    man_path = Path(args.manifest)
    gt = _load_gt(man_path)
    audit_ev = [e for e in json.loads(Path(args.audit).read_text())["events"] if e.get("verdict")]

    dumps = []
    need_rids = set()
    for p in sorted(Path(args.scores).glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["record_id"] not in gt:
            continue
        dumps.append(d)
        need_rids.add(d["record_id"])

    # ---- 配对几何特征缓存 ----
    feat_path = Path(args.feat_cache)
    if feat_path.is_file():
        feat_of = json.loads(feat_path.read_text(encoding="utf-8"))
        print(f"[..] 复用特征缓存 {feat_path}  ({len(feat_of)} 条)")
    else:
        recs = {}
        for ref in list_records_from_manifest(man_path):
            if ref.record_id not in need_rids:
                continue
            rec = load_record(ref, paths)
            recs[ref.record_id] = {
                "byf": {
                    int(f.get("source_frame_idx") or f.get("frame_idx") or 0): f
                    for f in rec.frames
                },
                "trigger": BoxTrigger(rec.boxes, wrist_score_min=0.15),
                "box": {b.token: b for b in rec.boxes},
                "ih": int(rec.meta.get("infer_height") or ref.infer_height or 1),
            }
            print(f"[load] {ref.clip_name[:40]}")

        feat_of: dict[str, dict[str, float]] = {}
        for d in dumps:
            rid = d["record_id"]
            R = recs[rid]
            for frame_idx, pairs in d["frames"]:
                fi = int(frame_idx)
                fr = R["byf"].get(fi) or {}
                persons = {
                    str(p.get("person_track_id")): p
                    for p in (fr.get("persons") or [])
                    if isinstance(p, dict)
                }
                for x in pairs:
                    track, tok = str(x[0]), str(x[1])
                    person = persons.get(track)
                    if person is None:
                        continue
                    hit = next(
                        (h for h in R["trigger"].hits_for_person(person) if h["token"] == tok),
                        None,
                    )
                    box = R["box"].get(tok)
                    if hit is None or box is None:
                        continue
                    pf = compute_pair_features(person, hit, box, infer_height=R["ih"])
                    key = f"{rid}|{fi}|{track}|{tok}"
                    feat_of[key] = {
                        "depth_ratio": float(pf.get("depth_ratio") or 0.0),
                        "center_dist_norm": float(pf.get("center_dist_norm") or 99.0),
                    }
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        feat_path.write_text(json.dumps(feat_of), encoding="utf-8")
        print(f"[ok] 特征 → {feat_path}  ({len(feat_of)} 条)")

    # ---- 可选动作分 ----
    act_of: dict[tuple[str, int, str], float] = {}
    if args.with_action:
        seqs_raw = json.loads(Path(args.seq_cache).read_text(encoding="utf-8"))
        seqs = {
            r: {int(t): {int(f): v for f, v in fr.items()} for t, fr in tr.items()}
            for r, tr in seqs_raw.items()
        }
        rows = []
        for d in dumps:
            rid = d["record_id"]
            segs = gt[rid]
            seen: set[tuple[int, str]] = set()
            for frame_idx, pairs in d["frames"]:
                fi = int(frame_idx)
                for x in pairs:
                    track = str(x[0])
                    if (fi, track) in seen:
                        continue
                    seen.add((fi, track))
                    frames = (seqs.get(rid) or {}).get(int(track)) or {}
                    feat, cov = window_features(frames, fi, args.window, args.step)
                    rows.append(
                        {
                            "rid": rid,
                            "fi": fi,
                            "track": track,
                            "y": int(_person_positive(segs, fi, track)),
                            "feat": feat + [cov],
                        }
                    )
        X = np.array([r["feat"] for r in rows], dtype=np.float64)
        y = np.array([r["y"] for r in rows], dtype=int)
        g = np.array([r["rid"] for r in rows])
        p = _oof_gbdt(X, y, g)
        act_of = {(r["rid"], r["fi"], r["track"]): float(p[i]) for i, r in enumerate(rows)}
        print(f"[ok] 动作分 OOF 样本 {len(y)}")

    val_by = {rid: [s for s in segs if s.get("split") == "val"] for rid, segs in gt.items()}

    def audit_killed(t_act: float, d_min: float, c_max: float) -> dict[str, tuple[int, int]]:
        out = {v: [0, 0] for v in ("not_pick", "wrong_box", "missing")}
        for e in audit_ev:
            v = e["verdict"]
            if v not in out:
                continue
            out[v][1] += 1
            rid, fi = e["record_id"], int(e["frame"])
            track, tok = str(e.get("person_track_id") or ""), e["token"]
            fk = f"{rid}|{fi}|{track}|{tok}"
            feat = feat_of.get(fk) or {}
            dr = feat.get("depth_ratio", 0.0)
            cd = feat.get("center_dist_norm", 99.0)
            a = act_of.get((rid, fi, track), 1.0)
            if a < t_act or dr < d_min or cd > c_max:
                out[v][0] += 1
        return {k: (a, b) for k, (a, b) in out.items()}

    def eval_policy(t_act: float, d_min: float, c_max: float) -> tuple[int, int, int]:
        n_val = n_det = n_fp = 0
        for d in dumps:
            rid = d["record_id"]
            cover = gt[rid]
            tr = AlarmTracker(min_consecutive_frames=args.min_frames, cooldown_frames=0)
            alarms: list[tuple[int, str]] = []
            for frame_idx, pairs in d["frames"]:
                fi = int(frame_idx)
                toks: set[str] = set()
                for x in pairs:
                    track, tok, sm = str(x[0]), str(x[1]), float(x[3])
                    if sm < args.pair_threshold:
                        continue
                    if act_of and act_of.get((rid, fi, track), 1.0) < t_act:
                        continue
                    feat = feat_of.get(f"{rid}|{fi}|{track}|{tok}") or {}
                    if feat.get("depth_ratio", 0.0) < d_min:
                        continue
                    if feat.get("center_dist_norm", 99.0) > c_max:
                        continue
                    toks.add(tok)
                for tok in tr.step(sorted(toks), fi):
                    alarms.append((fi, tok))
            for s in val_by.get(rid) or []:
                n_val += 1
                a, b = int(s["frame_start"]), int(s["frame_end"])
                if any(
                    a <= f <= b and any(_token_match(t, g) for g in s["gt_tokens"])
                    for f, t in alarms
                ):
                    n_det += 1
            n_fp += len(_fp_runs(alarms, cover))
        return n_det, n_val, n_fp

    # 扫描网格
    depth_mins = [0.0, 0.25, 0.35, 0.40, 0.45, 0.50]
    dist_maxs = [99.0, 2.0, 1.6, 1.4, 1.2, 1.0]
    act_ts = [0.0, 0.30] if args.with_action else [0.0]

    lines = [
        "# 邻框消歧（B）" + (" + 动作门控（A）" if args.with_action else ""),
        "",
        f"配对阈值 {args.pair_threshold} / 连续帧 {args.min_frames}",
        "",
        "| 动作门槛 | depth≥ | 距心≤ | 召回 | 漏段 | 误报 | 压没拣货 | 压报错框 | 压漏标 |",
        "|---------|--------|-------|------|------|------|----------|----------|--------|",
    ]
    print(f"{'动作':>6} {'depth≥':>7} {'距心≤':>6} {'召回':>8} {'漏':>4} {'误报':>6}  "
          f"{'没拣货':>8} {'报错框':>8} {'漏标':>6}")

    rows_out = []
    for t_act in act_ts:
        for d_min in depth_mins:
            for c_max in dist_maxs:
                # 无约束的重复组合跳过一部分
                if d_min == 0.0 and c_max == 99.0 and t_act == 0.0:
                    pass
                n_det, n_val, n_fp = eval_policy(t_act, d_min, c_max)
                killed = audit_killed(t_act, d_min, c_max)
                recall = n_det / n_val if n_val else 0.0
                def fmt(v: str) -> str:
                    a, b = killed[v]
                    return f"{a}/{b}"
                print(
                    f"{t_act:6.2f} {d_min:7.2f} {c_max:6.1f} {recall:8.2%} "
                    f"{n_val-n_det:4d} {n_fp:6d}  {fmt('not_pick'):>8} "
                    f"{fmt('wrong_box'):>8} {fmt('missing'):>6}"
                )
                lines.append(
                    f"| {t_act:.2f} | {d_min:.2f} | {c_max:.1f} | {recall:.2%} | "
                    f"{n_val-n_det} | {n_fp} | {fmt('not_pick')} | {fmt('wrong_box')} | "
                    f"{fmt('missing')} |"
                )
                rows_out.append(
                    {
                        "t_act": t_act,
                        "depth_min": d_min,
                        "center_max": c_max,
                        "recall": recall,
                        "missed": n_val - n_det,
                        "fp": n_fp,
                        "kill_not_pick": killed["not_pick"][0],
                        "kill_wrong_box": killed["wrong_box"][0],
                        "kill_missing": killed["missing"][0],
                    }
                )

    # Pareto：漏段尽量少，同漏段下误报最少
    lines += ["", "## 漏报优先 Pareto（漏段升序，同漏段取误报最少）", ""]
    best: dict[int, dict] = {}
    for r in rows_out:
        m = r["missed"]
        if m not in best or r["fp"] < best[m]["fp"]:
            best[m] = r
    print("\nPareto（漏报优先）：")
    for m in sorted(best):
        r = best[m]
        print(
            f"  漏{m:2d} 召回{r['recall']:.2%} 误报{r['fp']:4d}  "
            f"动作≥{r['t_act']:.2f} depth≥{r['depth_min']:.2f} 距心≤{r['center_max']:.1f}  "
            f"压报错框{r['kill_wrong_box']}/120 压没拣{r['kill_not_pick']}/120"
        )
        lines.append(
            f"- 漏 {m}（召回 {r['recall']:.2%}）：误报 {r['fp']}，"
            f"动作≥{r['t_act']:.2f} depth≥{r['depth_min']:.2f} 距心≤{r['center_max']:.1f}，"
            f"压报错框 {r['kill_wrong_box']}/120、没拣货 {r['kill_not_pick']}/120"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out.parent / "rows.json").write_text(
        json.dumps({"with_action": args.with_action, "rows": rows_out}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
