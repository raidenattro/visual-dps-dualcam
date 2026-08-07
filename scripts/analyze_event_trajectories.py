#!/usr/bin/env python3
"""只读诊断：真值拣货段 vs 算法误报事件的多维轨迹形态是否可分。

不做连续帧门槛，看 score / depth_ratio / center_dist_norm 随时间的相位与斜率。
产物写 output/sweep/event_confirm/。
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

from scripts.eval_export import EVENT_BOUNDARY_TOL, EVENT_MERGE_GAP, _interval_gap, _token_match
from scripts.sweep_policy import replay

DIMS = ("score", "depth_ratio", "center_dist_norm")
N_BINS = 12


def _load_pair_index(scores_dir: Path, feat_of: dict[str, dict[str, float]]) -> dict[str, dict[int, list[dict]]]:
    """rid -> frame -> list[{track, token, score, depth, center}]"""
    by: dict[str, dict[int, list[dict]]] = {}
    for p in sorted(scores_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rid = d["record_id"]
        frames: dict[int, list[dict]] = {}
        for frame_idx, pairs in d["frames"]:
            fi = int(frame_idx)
            rows = []
            for x in pairs:
                track, tok, _raw, sm = str(x[0]), str(x[1]), float(x[2]), float(x[3])
                feat = feat_of.get(f"{rid}|{fi}|{track}|{tok}") or {}
                rows.append(
                    {
                        "track": track,
                        "token": tok,
                        "score": sm,
                        "depth_ratio": float(feat.get("depth_ratio") or np.nan),
                        "center_dist_norm": float(feat.get("center_dist_norm") or np.nan),
                    }
                )
            frames[fi] = rows
        by[rid] = frames
    return by


def _series_for_pair(
    frame_map: dict[int, list[dict]],
    *,
    token: str,
    track: str | None,
    start: int,
    end: int,
) -> dict[str, np.ndarray]:
    fs = sorted(f for f in frame_map if start <= f <= end)
    out = {k: [] for k in DIMS}
    out_f = []
    for f in fs:
        cands = [
            r
            for r in frame_map[f]
            if _token_match(r["token"], token) and (track is None or str(r["track"]) == str(track))
        ]
        if not cands and track is not None:
            # 跟踪抖动：同框任意 track
            cands = [r for r in frame_map[f] if _token_match(r["token"], token)]
        if not cands:
            continue
        best = max(cands, key=lambda r: r["score"])
        out_f.append(f)
        for k in DIMS:
            out[k].append(best[k])
    return {"frames": np.asarray(out_f, dtype=int), **{k: np.asarray(out[k], dtype=float) for k in DIMS}}


def _pick_track(frame_map: dict[int, list[dict]], token: str, start: int, end: int, preferred: list[str]) -> str | None:
    scores: dict[str, float] = defaultdict(float)
    for f, rows in frame_map.items():
        if not (start <= f <= end):
            continue
        for r in rows:
            if _token_match(r["token"], token):
                scores[str(r["track"])] += float(r["score"])
    if not scores:
        return None
    for t in preferred:
        if str(t) in scores:
            return str(t)
    return max(scores, key=scores.get)


def _resample(series: dict[str, np.ndarray], n_bins: int = N_BINS) -> dict[str, np.ndarray] | None:
    frames = series["frames"]
    if len(frames) < 2:
        return None
    t0, t1 = float(frames[0]), float(frames[-1])
    if t1 <= t0:
        return None
    grid = np.linspace(0.0, 1.0, n_bins)
    src = (frames - t0) / (t1 - t0)
    out = {}
    for k in DIMS:
        y = series[k]
        ok = np.isfinite(y)
        if ok.sum() < 2:
            return None
        out[k] = np.interp(grid, src[ok], y[ok])
    return out


def _shape_feats(series: dict[str, np.ndarray]) -> dict[str, float] | None:
    frames = series["frames"]
    if len(frames) < 3:
        return None
    span = float(frames[-1] - frames[0] + 1)
    n = len(frames)
    mid = n // 2
    feats: dict[str, float] = {"n_hit": float(n), "span": span, "density": n / max(span, 1.0)}

    sc = series["score"]
    dep = series["depth_ratio"]
    cen = series["center_dist_norm"]
    if not (np.isfinite(sc).sum() >= 3 and np.isfinite(dep).sum() >= 3):
        return None

    def slope(y: np.ndarray) -> float:
        x = np.arange(len(y), dtype=float)
        m = np.isfinite(y)
        if m.sum() < 2:
            return float("nan")
        return float(np.polyfit(x[m], y[m], 1)[0])

    feats["score_peak"] = float(np.nanmax(sc))
    feats["score_peak_pos"] = float(np.nanargmax(sc) / max(n - 1, 1))  # 0=开头 1=结尾
    feats["score_slope"] = slope(sc)
    feats["score_slope_1st"] = slope(sc[: max(mid, 2)])
    feats["score_slope_2nd"] = slope(sc[mid:])
    feats["depth_mean"] = float(np.nanmean(dep))
    feats["depth_peak"] = float(np.nanmax(dep))
    feats["depth_slope"] = slope(dep)
    feats["depth_slope_1st"] = slope(dep[: max(mid, 2)])
    feats["depth_slope_2nd"] = slope(dep[mid:])
    feats["depth_rise_then_flat"] = float(
        feats["depth_slope_1st"] - max(feats["depth_slope_2nd"], 0.0)
    )
    feats["center_mean"] = float(np.nanmean(cen))
    feats["center_slope"] = slope(cen)
    # 协调：depth 升时 center 应降
    m = np.isfinite(dep) & np.isfinite(cen) & np.isfinite(sc)
    if m.sum() >= 4:
        feats["corr_depth_neg_center"] = float(np.corrcoef(dep[m], -cen[m])[0, 1])
        feats["corr_score_depth"] = float(np.corrcoef(sc[m], dep[m])[0, 1])
        feats["corr_score_neg_center"] = float(np.corrcoef(sc[m], -cen[m])[0, 1])
    else:
        feats["corr_depth_neg_center"] = float("nan")
        feats["corr_score_depth"] = float("nan")
        feats["corr_score_neg_center"] = float("nan")
    return feats


def _auc(y: np.ndarray, s: np.ndarray) -> float | None:
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) < 10 or y.min() == y.max():
        return None
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # P(pos > neg) + 0.5 P(eq)
    wins = sum(float((neg < p).sum() + 0.5 * (neg == p).sum()) for p in pos)
    return float(wins / (len(pos) * len(neg)))


def _mean_curve(curves: list[dict[str, np.ndarray]], key: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.stack([c[key] for c in curves], axis=0)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _fmt_curve(mu: np.ndarray) -> str:
    return " ".join(f"{v:.2f}" for v in mu)


def collect_fp_events(
    dumps: list[dict],
    by_rid_segs: dict[str, list[dict]],
    pair_index: dict[str, dict[int, list[dict]]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    events = []
    for d in dumps:
        rid = d["record_id"]
        segs = by_rid_segs[rid]
        frames = replay(d, threshold=threshold, min_frames=1, exclusive=False)
        # uncovered alarm frames with best track
        cover = segs
        unc: list[tuple[int, str, str, float]] = []  # f, tok, track, score
        fmap = pair_index.get(rid) or {}
        for fr in frames:
            if not fr["is_picking"]:
                continue
            fi = int(fr["frame_idx"])
            for tok in fr["rule_alarm_collisions"]:
                covered = any(
                    int(s["frame_start"]) <= fi <= int(s["frame_end"])
                    and any(_token_match(tok, g) for g in (s.get("gt_tokens") or []))
                    for s in cover
                )
                if covered:
                    continue
                rows = [r for r in (fmap.get(fi) or []) if _token_match(r["token"], tok) and r["score"] >= threshold]
                if not rows:
                    continue
                best = max(rows, key=lambda r: r["score"])
                unc.append((fi, tok, best["track"], best["score"]))
        by_tok: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
        for fi, tok, track, sc in unc:
            by_tok[tok].append((fi, track, sc))
        for tok, items in by_tok.items():
            items = sorted(items, key=lambda x: x[0])
            run = [items[0]]
            for it in items[1:]:
                if it[0] - run[-1][0] <= EVENT_MERGE_GAP:
                    run.append(it)
                else:
                    events.append(_fp_pack(rid, tok, run, cover, fmap))
                    run = [it]
            events.append(_fp_pack(rid, tok, run, cover, fmap))
    return [e for e in events if e is not None]


def _fp_pack(rid, tok, run, cover, fmap):
    span = (run[0][0], run[-1][0])
    own = [
        (int(s["frame_start"]), int(s["frame_end"]))
        for s in cover
        if any(_token_match(tok, g) for g in (s.get("gt_tokens") or []))
    ]
    near = min((_interval_gap(span, s) for s in own), default=10**9)
    if near <= EVENT_BOUNDARY_TOL:
        return None  # 只看算法误报
    # 主 track = 出现最多
    tracks = [t for _, t, _ in run]
    track = max(set(tracks), key=tracks.count)
    series = _series_for_pair(fmap, token=tok, track=track, start=span[0], end=span[1])
    # 上下文窗（仅用于曲线；形态特征优先用事件本体，太短再退回上下文）
    ctx = _series_for_pair(fmap, token=tok, track=track, start=span[0] - 6, end=span[1] + 6)
    shape = _shape_feats(series) or _shape_feats(ctx)
    curve = _resample(ctx if len(ctx["frames"]) >= 2 else series)
    if shape is None:
        return None
    return {
        "kind": "fp",
        "record_id": rid,
        "token": tok,
        "track": track,
        "frame_start": span[0],
        "frame_end": span[1],
        "n_alarm": float(len(run)),
        "shape": shape,
        "curve": curve,
    }


def collect_tp_events(
    by_rid_segs: dict[str, list[dict]],
    pair_index: dict[str, dict[int, list[dict]]],
) -> list[dict[str, Any]]:
    out = []
    for rid, segs in by_rid_segs.items():
        fmap = pair_index.get(rid) or {}
        for s in segs:
            if s.get("split") != "val":
                continue
            toks = s.get("gt_tokens") or []
            if not toks:
                continue
            tok = toks[0]
            preferred = [str(t) for t in (s.get("person_track_ids") or [])]
            a, b = int(s["frame_start"]), int(s["frame_end"])
            track = _pick_track(fmap, tok, a, b, preferred)
            series = _series_for_pair(fmap, token=tok, track=track, start=a, end=b)
            shape = _shape_feats(series)
            curve = _resample(series)
            if shape is None or curve is None:
                continue
            out.append(
                {
                    "kind": "tp",
                    "record_id": rid,
                    "token": tok,
                    "track": track,
                    "frame_start": a,
                    "frame_end": b,
                    "shape": shape,
                    "curve": curve,
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="事件轨迹形态诊断")
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v4.json"))
    ap.add_argument("--scores", default=str(ROOT / "output/scores/v5_on_v4"))
    ap.add_argument("--feat-cache", default=str(ROOT / "output/sweep/v5_box_gate/pair_feats_v4.json"))
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--out-dir", default=str(ROOT / "output/sweep/event_confirm"))
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    by_rid: dict[str, list[dict]] = defaultdict(list)
    for s in man.get("segments") or []:
        by_rid[str(s["record_id"])].append(s)

    feat_of = json.loads(Path(args.feat_cache).read_text(encoding="utf-8"))
    scores_dir = Path(args.scores)
    print("[..] 索引配对分数与几何特征")
    pair_index = _load_pair_index(scores_dir, feat_of)

    dumps = []
    for p in sorted(scores_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["record_id"] in by_rid:
            dumps.append(d)

    print("[..] 抽取 TP / FP 事件轨迹")
    tps = collect_tp_events(by_rid, pair_index)
    fps = collect_fp_events(dumps, by_rid, pair_index, threshold=args.threshold)
    print(f"[ok] TP={len(tps)}  FP算法={len(fps)}")

    # 形态特征可分性
    shape_keys = [
        "n_hit",
        "span",
        "density",
        "score_peak",
        "score_peak_pos",
        "score_slope",
        "score_slope_1st",
        "score_slope_2nd",
        "depth_mean",
        "depth_peak",
        "depth_slope",
        "depth_slope_1st",
        "depth_slope_2nd",
        "depth_rise_then_flat",
        "center_mean",
        "center_slope",
        "corr_depth_neg_center",
        "corr_score_depth",
        "corr_score_neg_center",
    ]
    rows = []
    for e in tps + fps:
        y = 1 if e["kind"] == "tp" else 0
        rows.append((y, e["shape"]))
    y = np.array([r[0] for r in rows])
    auc_rows = []
    for k in shape_keys:
        s = np.array([r[1].get(k, np.nan) for r in rows], dtype=float)
        # 方向：取 max(auc, 1-auc)
        a = _auc(y, s)
        a2 = _auc(y, -s)
        if a is None and a2 is None:
            continue
        if a is None or (a2 is not None and a2 > a):
            best, direction = float(a2), "-"
        else:
            best, direction = float(a), "+"
        med_tp = float(np.nanmedian(s[y == 1]))
        med_fp = float(np.nanmedian(s[y == 0]))
        auc_rows.append((best, direction, k, med_tp, med_fp))
    auc_rows.sort(reverse=True)

    # 水平量 vs 纯形态：便于判断「像不像」是否独立于「高不高」
    level_keys = {"score_peak", "depth_mean", "depth_peak", "center_mean", "span", "n_hit", "density"}
    shape_only = [r for r in auc_rows if r[2] not in level_keys]
    level_only = [r for r in auc_rows if r[2] in level_keys]

    # 平均曲线（需成功重采样）
    tp_curves = [e["curve"] for e in tps if e.get("curve") is not None]
    fp_curves = [e["curve"] for e in fps if e.get("curve") is not None]
    tps = [e for e in tps if e.get("curve") is not None]  # 保持与曲线统计一致
    # FP 允许无曲线但有 shape；曲线统计用子集

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 事件轨迹形态诊断",
        "",
        f"manifest `{args.manifest}`，scores `{args.scores}`，阈值 {args.threshold}",
        f"可用轨迹：真值段(TP) **{len(tps)}**，算法误报事件(FP) **{len(fps)}**",
        f"时间归一化到 {N_BINS} 个相位点（事件起点→终点）。",
        "",
        "## 1. 平均轨迹（相位 0→1）",
        "",
    ]
    for dim in DIMS:
        mu_t, sd_t = _mean_curve(tp_curves, dim)
        mu_f, sd_f = _mean_curve(fp_curves, dim)
        lines += [
            f"### {dim}",
            "",
            f"- TP mean: `{_fmt_curve(mu_t)}`",
            f"- FP mean: `{_fmt_curve(mu_f)}`",
            f"- 末段-首段 ΔTP={mu_t[-1] - mu_t[0]:+.3f}  ΔFP={mu_f[-1] - mu_f[0]:+.3f}",
            f"- 中段(半程) TP={mu_t[N_BINS // 2]:.3f} FP={mu_f[N_BINS // 2]:.3f}",
            "",
        ]
        print(f"\n{dim}:")
        print("  TP", _fmt_curve(mu_t))
        print("  FP", _fmt_curve(mu_f))

    lines += [
        "## 2. 事件级特征可分性（AUC，TP vs FP）",
        "",
        "### 2a. 水平 / 长度（更容易，但也更像「分数高、段更长」）",
        "",
        "| AUC | 方向 | 特征 | TP中位 | FP中位 |",
        "|-----|------|------|--------|--------|",
    ]
    print("\nLevel features:")
    for best, direction, k, med_tp, med_fp in level_only:
        lines.append(f"| {best:.3f} | {direction} | `{k}` | {med_tp:.3f} | {med_fp:.3f} |")
        print(f"  {best:.3f} {direction} {k:28} tp={med_tp:.3f} fp={med_fp:.3f}")
    lines += [
        "",
        "### 2b. 纯形态（斜率 / 相位 / 维间相关）",
        "",
        "| AUC | 方向 | 特征 | TP中位 | FP中位 |",
        "|-----|------|------|--------|--------|",
    ]
    print("\nShape-only features:")
    for best, direction, k, med_tp, med_fp in shape_only:
        lines.append(f"| {best:.3f} | {direction} | `{k}` | {med_tp:.3f} | {med_fp:.3f} |")
        print(f"  {best:.3f} {direction} {k:28} tp={med_tp:.3f} fp={med_fp:.3f}")

    # 简单组合：depth 前半升 + corr + span
    combo = []
    for e in tps + fps:
        sh = e["shape"]
        # 启发式：像拣货 = 较长 + depth 前半上升 + depth/center 负相关 + score-depth 正相关
        score = (
            0.15 * np.log1p(sh["span"])
            + 2.0 * max(sh.get("depth_slope_1st") or 0.0, 0.0)
            + 0.5 * (sh.get("corr_depth_neg_center") or 0.0)
            + 0.5 * (sh.get("corr_score_depth") or 0.0)
            - 0.3 * abs(sh.get("center_mean") or 0.0)
        )
        combo.append((1 if e["kind"] == "tp" else 0, score))
    y_c = np.array([c[0] for c in combo])
    s_c = np.array([c[1] for c in combo])
    a_c = _auc(y_c, s_c)
    lines += [
        "",
        "## 3. 粗糙启发式轨迹分（未训练）",
        "",
        f"`0.15*log(span) + 2*max(depth_slope_1st,0) + 0.5*corr(depth,-center) "
        f"+ 0.5*corr(score,depth) - 0.3*|center_mean|`",
        f"",
        f"- AUC = **{a_c:.3f}**" if a_c is not None else "- AUC = n/a",
        "",
        "## 4. 解读",
        "",
    ]
    if level_only:
        lines.append(
            f"- 最强信号在**水平/长度**：`{level_only[0][2]}` AUC {level_only[0][0]:.3f}"
            f"（TP中位 {level_only[0][3]:.3f} vs FP {level_only[0][4]:.3f}）"
        )
    if shape_only:
        lines.append(
            f"- 最强**纯形态**信号：`{shape_only[0][2]}` AUC {shape_only[0][0]:.3f}"
            f"；若 <0.60，说明仅靠斜率/相关还不够，需更丰富的轨迹维或序列模型"
        )
    mu_td, _ = _mean_curve(tp_curves, "depth_ratio")
    mu_fd, _ = _mean_curve(fp_curves, "depth_ratio")
    lines.append(
        f"- depth 平均曲线：TP 中段 {mu_td[N_BINS // 2]:.2f}，FP 中段 {mu_fd[N_BINS // 2]:.2f}；"
        f"两者都有「先升后回落」痕迹，差异更多是幅度"
    )
    mu_ts, _ = _mean_curve(tp_curves, "score")
    mu_fs, _ = _mean_curve(fp_curves, "score")
    lines.append(
        f"- score 包络：TP 中段 {mu_ts[N_BINS // 2]:.2f} vs FP {mu_fs[N_BINS // 2]:.2f}，分离很强"
    )
    lines += [
        "- **结论倾向**：当前三维（score/depth/center）上，「像不像」尚未明显独立于「高不高、长不长」；",
        "  下一步应把腕速/臂角等也拉成事件轨迹，或对高分 FP 做分数匹配后再看形态",
        "",
    ]

    report = out_dir / "trajectory_morphology.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # save curves json for later plots
    payload = {
        "n_tp": len(tps),
        "n_fp": len(fps),
        "bins": N_BINS,
        "mean_curves": {
            "tp": {k: _mean_curve(tp_curves, k)[0].tolist() for k in DIMS},
            "fp": {k: _mean_curve(fp_curves, k)[0].tolist() for k in DIMS},
        },
        "auc": [
            {"auc": a, "direction": d, "feature": k, "tp_median": t, "fp_median": f}
            for a, d, k, t, f in auc_rows
        ],
        "heuristic_auc": a_c,
    }
    (out_dir / "trajectory_morphology.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nheuristic AUC={a_c}")
    print(f"→ {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
