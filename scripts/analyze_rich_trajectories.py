#!/usr/bin/env python3
"""更富轨迹诊断：在事件窗内叠腕速 / 臂角 / 动作分相位，看是否独立于「分高段长」。

复用 action_sequences + action_gate_v1；产物写 output/sweep/event_confirm/。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.action_temporal import window_features
from scripts.analyze_event_trajectories import (
    N_BINS,
    _auc,
    _fmt_curve,
    _load_pair_index,
    _mean_curve,
    _pick_track,
    _series_for_pair,
    collect_fp_events,
    collect_tp_events,
)
from scripts.eval_export import _token_match

# SEQ_KEYS: 5 person + wl_x,wl_y,wr_x,wr_y
IDX_ARM = 1
IDX_ELBOW = 2
IDX_WRIST_ELEV = 3
IDX_WL = 5
IDX_WR = 7

RICH_DIMS = (
    "wrist_speed",
    "arm_torso_angle",
    "elbow_angle",
    "wrist_elev_angle",
    "action_score",
)
ALL_CURVE_DIMS = ("score", "depth_ratio", "center_dist_norm", *RICH_DIMS)


def _load_seqs(path: Path) -> dict[str, dict[int, dict[int, list[float]]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        r: {int(t): {int(f): v for f, v in fr.items()} for t, fr in tr.items()}
        for r, tr in raw.items()
    }


def _wrist_mid(vec: list[float]) -> np.ndarray | None:
    pts = []
    for base in (IDX_WL, IDX_WR):
        x, y = vec[base], vec[base + 1]
        if x == x and y == y:  # not nan
            pts.append((float(x), float(y)))
    if not pts:
        return None
    return np.mean(pts, axis=0)


def _person_at(
    seq_track: dict[int, list[float]], frame: int
) -> dict[str, float]:
    """单帧人级量；腕速用相对上一可用帧的位移/Δframe。"""
    out = {k: float("nan") for k in RICH_DIMS if k != "action_score"}
    vec = seq_track.get(frame)
    if vec is None:
        return out
    out["arm_torso_angle"] = float(vec[IDX_ARM]) if vec[IDX_ARM] == vec[IDX_ARM] else float("nan")
    out["elbow_angle"] = float(vec[IDX_ELBOW]) if vec[IDX_ELBOW] == vec[IDX_ELBOW] else float("nan")
    out["wrist_elev_angle"] = (
        float(vec[IDX_WRIST_ELEV]) if vec[IDX_WRIST_ELEV] == vec[IDX_WRIST_ELEV] else float("nan")
    )
    cur = _wrist_mid(vec)
    prev_fs = [f for f in seq_track if f < frame]
    if cur is not None and prev_fs:
        pf = max(prev_fs)
        prev = _wrist_mid(seq_track[pf])
        if prev is not None:
            out["wrist_speed"] = float(np.linalg.norm(cur - prev) / max(frame - pf, 1))
    return out


def _batch_action_scores(
    keys: list[tuple[str, int, str]],
    seqs: dict,
    model,
    *,
    window: int,
    step: int,
) -> dict[tuple[str, int, str], float]:
    """keys: (rid, frame, track) -> action_score"""
    uniq = sorted(set(keys))
    if not uniq:
        return {}
    Xs = []
    for rid, fi, track in uniq:
        frames = (seqs.get(rid) or {}).get(int(track)) or {}
        feat, cov = window_features(frames, fi, window, step)
        Xs.append(feat + [cov])
    X = np.asarray(Xs, dtype=np.float64)
    p = model.predict_proba(X)[:, 1]
    return {uniq[i]: float(p[i]) for i in range(len(uniq))}


def _attach_rich(
    series: dict[str, np.ndarray],
    *,
    rid: str,
    track: str | None,
    seqs: dict,
    act_of: dict[tuple[str, int, str], float],
) -> dict[str, np.ndarray]:
    frames = series["frames"]
    rich = {k: [] for k in RICH_DIMS}
    seq_track = (seqs.get(rid) or {}).get(int(track)) if track is not None else None
    seq_track = seq_track or {}
    for fi in frames:
        fi = int(fi)
        pers = _person_at(seq_track, fi) if seq_track else {}
        for k in RICH_DIMS:
            if k == "action_score":
                rich[k].append(act_of.get((rid, fi, str(track)), float("nan")))
            else:
                rich[k].append(pers.get(k, float("nan")))
    out = dict(series)
    for k, vals in rich.items():
        out[k] = np.asarray(vals, dtype=float)
    return out


def _resample_any(series: dict[str, np.ndarray], dims: tuple[str, ...]) -> dict[str, np.ndarray] | None:
    frames = series["frames"]
    if len(frames) < 2:
        return None
    t0, t1 = float(frames[0]), float(frames[-1])
    if t1 <= t0:
        return None
    grid = np.linspace(0.0, 1.0, N_BINS)
    src = (frames - t0) / (t1 - t0)
    out = {}
    ok_any = False
    for k in dims:
        y = series.get(k)
        if y is None:
            continue
        m = np.isfinite(y)
        if m.sum() < 2:
            out[k] = np.full(N_BINS, np.nan)
            continue
        out[k] = np.interp(grid, src[m], y[m])
        ok_any = True
    return out if ok_any else None


def _slope(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=float)
    m = np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(x[m], y[m], 1)[0])


def _shape_rich(series: dict[str, np.ndarray]) -> dict[str, float] | None:
    frames = series["frames"]
    if len(frames) < 3 or np.isfinite(series["score"]).sum() < 3:
        return None
    n = len(frames)
    mid = n // 2
    span = float(frames[-1] - frames[0] + 1)
    feats: dict[str, float] = {
        "n_hit": float(n),
        "span": span,
        "density": n / max(span, 1.0),
        "score_peak": float(np.nanmax(series["score"])),
        "score_peak_pos": float(np.nanargmax(series["score"]) / max(n - 1, 1)),
    }
    for dim in ALL_CURVE_DIMS:
        y = series.get(dim)
        if y is None or np.isfinite(y).sum() < 3:
            continue
        feats[f"{dim}_mean"] = float(np.nanmean(y))
        feats[f"{dim}_peak"] = float(np.nanmax(y))
        feats[f"{dim}_slope"] = _slope(y)
        feats[f"{dim}_slope_1st"] = _slope(y[: max(mid, 2)])
        feats[f"{dim}_slope_2nd"] = _slope(y[mid:])
        peak_i = int(np.nanargmax(y))
        feats[f"{dim}_peak_pos"] = float(peak_i / max(n - 1, 1))
        # 半程差：后半均值 - 前半均值
        feats[f"{dim}_half_delta"] = float(np.nanmean(y[mid:]) - np.nanmean(y[:mid]))

    sc, dep, act = series["score"], series["depth_ratio"], series.get("action_score")
    ws = series.get("wrist_speed")
    arm = series.get("arm_torso_angle")
    m = np.isfinite(sc) & np.isfinite(dep)
    if m.sum() >= 4:
        feats["corr_score_depth"] = float(np.corrcoef(sc[m], dep[m])[0, 1])
    if act is not None:
        m2 = np.isfinite(sc) & np.isfinite(act)
        if m2.sum() >= 4:
            feats["corr_score_action"] = float(np.corrcoef(sc[m2], act[m2])[0, 1])
        m3 = np.isfinite(dep) & np.isfinite(act)
        if m3.sum() >= 4:
            feats["corr_depth_action"] = float(np.corrcoef(dep[m3], act[m3])[0, 1])
    if ws is not None and act is not None:
        m4 = np.isfinite(ws) & np.isfinite(act)
        if m4.sum() >= 4:
            feats["corr_wrist_action"] = float(np.corrcoef(ws[m4], act[m4])[0, 1])
    if arm is not None and dep is not None:
        m5 = np.isfinite(arm) & np.isfinite(dep)
        if m5.sum() >= 4:
            feats["corr_arm_depth"] = float(np.corrcoef(arm[m5], dep[m5])[0, 1])
    # 动作「先起后稳」：前半 action 升、后半不掉
    if act is not None and np.isfinite(act).sum() >= 4:
        feats["action_rise_hold"] = float(
            max(_slope(act[: max(mid, 2)]), 0.0) - max(-_slope(act[mid:]), 0.0)
        )
    return feats


def _auc_table(events: list[dict], keys: list[str]) -> list[tuple]:
    y = np.array([1 if e["kind"] == "tp" else 0 for e in events])
    rows = []
    for k in keys:
        s = np.array([e["shape"].get(k, np.nan) for e in events], dtype=float)
        a, a2 = _auc(y, s), _auc(y, -s)
        if a is None and a2 is None:
            continue
        if a is None or (a2 is not None and a2 > a):
            best, direction = float(a2), "-"
        else:
            best, direction = float(a), "+"
        rows.append(
            (
                best,
                direction,
                k,
                float(np.nanmedian(s[y == 1])),
                float(np.nanmedian(s[y == 0])),
            )
        )
    rows.sort(reverse=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="更富事件轨迹诊断")
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v4.json"))
    ap.add_argument("--scores", default=str(ROOT / "output/scores/v5_on_v4"))
    ap.add_argument("--feat-cache", default=str(ROOT / "output/sweep/v5_box_gate/pair_feats_v4.json"))
    ap.add_argument("--seq-cache", default=str(ROOT / "output/train/action_sequences_v4.json"))
    ap.add_argument("--action-model", default=str(ROOT / "output/train/action_gate_v1/model.joblib"))
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--out-dir", default=str(ROOT / "output/sweep/event_confirm"))
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    by_rid: dict[str, list[dict]] = defaultdict(list)
    for s in man.get("segments") or []:
        by_rid[str(s["record_id"])].append(s)

    print("[..] 索引 / 序列 / 动作模型")
    feat_of = json.loads(Path(args.feat_cache).read_text(encoding="utf-8"))
    pair_index = _load_pair_index(Path(args.scores), feat_of)
    seqs = _load_seqs(Path(args.seq_cache))
    model = joblib.load(args.action_model)

    dumps = []
    for p in sorted(Path(args.scores).glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["record_id"] in by_rid:
            dumps.append(d)

    print("[..] 抽取基线事件")
    # 先用旧逻辑拿事件壳，再丢掉 shape/curve 用富轨迹重算
    tps_raw = collect_tp_events(by_rid, pair_index)
    fps_raw = collect_fp_events(dumps, by_rid, pair_index, threshold=args.threshold)
    print(f"[..] 壳 TP={len(tps_raw)} FP={len(fps_raw)}，重建富轨迹")

    # 重建 series
    shells = []
    for e in tps_raw:
        fmap = pair_index.get(e["record_id"]) or {}
        series = _series_for_pair(
            fmap,
            token=e["token"],
            track=e["track"],
            start=e["frame_start"],
            end=e["frame_end"],
        )
        shells.append({**e, "series": series})
    for e in fps_raw:
        fmap = pair_index.get(e["record_id"]) or {}
        series = _series_for_pair(
            fmap,
            token=e["token"],
            track=e["track"],
            start=e["frame_start"],
            end=e["frame_end"],
        )
        if len(series["frames"]) < 3:
            series = _series_for_pair(
                fmap,
                token=e["token"],
                track=e["track"],
                start=e["frame_start"] - 6,
                end=e["frame_end"] + 6,
            )
        shells.append({**e, "series": series})

    need_keys: list[tuple[str, int, str]] = []
    for e in shells:
        tr = e.get("track")
        if tr is None:
            continue
        for fi in e["series"]["frames"]:
            need_keys.append((e["record_id"], int(fi), str(tr)))
    print(f"[..] 批量动作分 {len(set(need_keys))} 点")
    act_of = _batch_action_scores(
        need_keys, seqs, model, window=args.window, step=args.step
    )

    events = []
    for e in shells:
        series = _attach_rich(
            e["series"],
            rid=e["record_id"],
            track=e.get("track"),
            seqs=seqs,
            act_of=act_of,
        )
        shape = _shape_rich(series)
        curve = _resample_any(series, ALL_CURVE_DIMS)
        if shape is None or curve is None:
            continue
        events.append(
            {
                "kind": e["kind"],
                "record_id": e["record_id"],
                "token": e["token"],
                "track": e.get("track"),
                "frame_start": e["frame_start"],
                "frame_end": e["frame_end"],
                "shape": shape,
                "curve": curve,
            }
        )

    tps = [e for e in events if e["kind"] == "tp"]
    fps = [e for e in events if e["kind"] == "fp"]
    print(f"[ok] 富轨迹 TP={len(tps)} FP={len(fps)}")

    shape_keys = sorted({k for e in events for k in e["shape"]})
    auc_rows = _auc_table(events, shape_keys)

    level_keys = {
        "score_peak",
        "score_mean",
        "depth_ratio_mean",
        "depth_ratio_peak",
        "center_dist_norm_mean",
        "action_score_mean",
        "action_score_peak",
        "arm_torso_angle_mean",
        "elbow_angle_mean",
        "wrist_elev_angle_mean",
        "wrist_speed_mean",
        "wrist_speed_peak",
        "span",
        "n_hit",
        "density",
    }
    # 也把 *_mean/*_peak 里非斜率的算水平
    level_only = [
        r
        for r in auc_rows
        if r[2] in level_keys or r[2].endswith("_mean") or r[2].endswith("_peak")
    ]
    shape_only = [
        r
        for r in auc_rows
        if r[2] not in {x[2] for x in level_only}
    ]
    # 去重 level：按 auc 排序已有
    level_only = sorted(level_only, reverse=True)
    rich_shape = [
        r
        for r in shape_only
        if any(
            x in r[2]
            for x in (
                "wrist_speed",
                "arm_torso",
                "elbow",
                "wrist_elev",
                "action_score",
                "action_rise",
                "corr_score_action",
                "corr_depth_action",
                "corr_wrist_action",
                "corr_arm_depth",
            )
        )
    ]

    tp_curves = [e["curve"] for e in tps]
    fp_curves = [e["curve"] for e in fps]

    # 分数匹配：FP score_peak 落在 TP IQR
    tp_peaks = np.array([e["shape"]["score_peak"] for e in tps])
    q1, q3 = np.percentile(tp_peaks, [25, 75])
    fps_m = [e for e in fps if q1 <= e["shape"]["score_peak"] <= q3]
    matched = tps + fps_m
    matched_auc = _auc_table(matched, shape_keys) if fps_m else []
    matched_rich = [
        r
        for r in matched_auc
        if any(
            x in r[2]
            for x in (
                "wrist_speed",
                "arm_torso",
                "elbow",
                "wrist_elev",
                "action",
                "corr_",
                "slope",
                "half_delta",
                "peak_pos",
            )
        )
        and not r[2].endswith("_mean")
        and not r[2].endswith("_peak")
        and r[2] not in ("span", "n_hit", "density", "score_peak")
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 更富事件轨迹诊断（腕速 / 臂角 / 动作分）",
        "",
        f"TP={len(tps)}  FP={len(fps)}  分数匹配FP(peak∈[{q1:.2f},{q3:.2f}])={len(fps_m)}",
        "",
        "## 1. 平均相位曲线",
        "",
    ]
    for dim in ALL_CURVE_DIMS:
        if dim not in tp_curves[0]:
            continue
        # skip if all nan
        if not np.isfinite(_mean_curve(tp_curves, dim)[0]).any():
            continue
        mu_t, _ = _mean_curve(tp_curves, dim)
        mu_f, _ = _mean_curve(fp_curves, dim)
        lines += [
            f"### {dim}",
            f"- TP: `{_fmt_curve(mu_t)}`",
            f"- FP: `{_fmt_curve(mu_f)}`",
            f"- 中段 TP={mu_t[N_BINS // 2]:.3f} FP={mu_f[N_BINS // 2]:.3f}  "
            f"Δ末-首 TP={mu_t[-1] - mu_t[0]:+.3f} FP={mu_f[-1] - mu_f[0]:+.3f}",
            "",
        ]
        print(f"\n{dim}:")
        print("  TP", _fmt_curve(mu_t))
        print("  FP", _fmt_curve(mu_f))

    def _dump_table(title: str, rows: list, n: int = 20) -> None:
        nonlocal lines
        lines += [f"## {title}", "", "| AUC | 方向 | 特征 | TP中位 | FP中位 |", "|-----|------|------|--------|--------|"]
        print(f"\n{title}")
        for best, d, k, t, f in rows[:n]:
            lines.append(f"| {best:.3f} | {d} | `{k}` | {t:.3f} | {f:.3f} |")
            print(f"  {best:.3f} {d} {k:32} tp={t:.3f} fp={f:.3f}")
        lines.append("")

    _dump_table("2. 水平量 Top", level_only, 15)
    _dump_table("3. 纯形态 Top（含旧三维）", shape_only, 20)
    _dump_table("4. 富轨迹相关形态（腕速/角/动作）", rich_shape, 20)
    if matched_rich:
        _dump_table(
            f"5. 分数匹配后形态（压制 score_peak 混淆，n_fp={len(fps_m)}）",
            matched_rich,
            20,
        )

    # 启发式：动作分抬升保持 + 腕速前半高后半低 + arm-depth 相关
    combo = []
    for e in events:
        sh = e["shape"]
        score = (
            1.5 * (sh.get("action_score_mean") or 0.0)
            + 1.0 * max(sh.get("action_rise_hold") or 0.0, 0.0)
            + 0.8 * (sh.get("corr_score_action") or 0.0)
            + 0.5 * (sh.get("corr_arm_depth") or 0.0)
            - 0.5 * (sh.get("wrist_speed_mean") or 0.0)  # 真拣货腕速中位可能更稳？待看符号
            + 0.1 * np.log1p(sh.get("span") or 0.0)
        )
        combo.append((1 if e["kind"] == "tp" else 0, score))
    a_c = _auc(np.array([c[0] for c in combo]), np.array([c[1] for c in combo]))

    # 看腕速符号：若 TP 腕速更高则改启发式
    ws_tp = np.nanmedian([e["shape"].get("wrist_speed_mean", np.nan) for e in tps])
    ws_fp = np.nanmedian([e["shape"].get("wrist_speed_mean", np.nan) for e in fps])
    act_tp = np.nanmedian([e["shape"].get("action_score_mean", np.nan) for e in tps])
    act_fp = np.nanmedian([e["shape"].get("action_score_mean", np.nan) for e in fps])

    best_rich = rich_shape[0] if rich_shape else None
    best_matched = matched_rich[0] if matched_rich else None
    lines += [
        "## 6. 解读",
        "",
        f"- action_score 中位：TP={act_tp:.3f} FP={act_fp:.3f}",
        f"- wrist_speed 中位：TP={ws_tp:.4f} FP={ws_fp:.4f}",
        f"- 粗糙富轨迹启发式 AUC={a_c:.3f}" if a_c else "- 启发式 n/a",
    ]
    if best_rich:
        lines.append(
            f"- 全量上最强富形态：`{best_rich[2]}` AUC {best_rich[0]:.3f}"
        )
    if best_matched:
        lines.append(
            f"- 分数匹配后最强形态：`{best_matched[2]}` AUC {best_matched[0]:.3f}"
            f"（≥0.65 才值得做事件轨迹门控）"
        )
    elif fps_m:
        lines.append("- 分数匹配后无明显形态信号")
    else:
        lines.append("- 高分 FP 太少，分数匹配不可靠")
    lines += [
        "",
        "判定：若第4/5节富形态 AUC 仍 <0.65，则「更富轨迹」单独做门控希望有限，",
        "应改事件级序列分类器（把整段曲线喂小模型）或回到几何/站位先验。",
        "",
    ]

    report = out_dir / "trajectory_rich.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "n_tp": len(tps),
        "n_fp": len(fps),
        "n_fp_score_matched": len(fps_m),
        "score_peak_iqr": [float(q1), float(q3)],
        "mean_curves": {
            "tp": {k: _mean_curve(tp_curves, k)[0].tolist() for k in ALL_CURVE_DIMS if k in tp_curves[0]},
            "fp": {k: _mean_curve(fp_curves, k)[0].tolist() for k in ALL_CURVE_DIMS if k in fp_curves[0]},
        },
        "auc_rich_shape": [
            {"auc": a, "direction": d, "feature": k, "tp_median": t, "fp_median": f}
            for a, d, k, t, f in rich_shape
        ],
        "auc_score_matched_shape": [
            {"auc": a, "direction": d, "feature": k, "tp_median": t, "fp_median": f}
            for a, d, k, t, f in matched_rich
        ],
        "heuristic_auc": a_c,
        "action_median": {"tp": act_tp, "fp": act_fp},
        "wrist_speed_median": {"tp": ws_tp, "fp": ws_fp},
    }
    (out_dir / "trajectory_rich.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nheuristic AUC={a_c}")
    print(f"→ {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
