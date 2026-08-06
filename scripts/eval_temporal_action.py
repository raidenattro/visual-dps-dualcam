"""时序动作判别：用一段骨架序列（只看过去）判断这个人在不在拣货，不看货框。

逐帧姿态的上限是 AUC 0.813（scripts/eval_action_only.py）。这里检验把时间结构补上
能否突破——「伸手取物再收回」和「手搭在货架上不动」的单帧姿态可能一样，时间曲线不同。

窗口只取 [t-W, t]，与线上实时约束一致，不用未来帧。
先把序列压成统计量而非直接上序列模型：只有 26 条 record，深度模型会过拟合，
统计量足以回答「时序里有没有信息」。
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_paths
from adapters.record_reader import list_records_from_manifest, load_record
from features.bank import FeatureBank
from features.geometry import WRIST_LEFT, WRIST_RIGHT, read_xy
from train.build_pair_dataset import PERSON_FEATURE_KEYS

# 序列里逐帧记录的量：5 维人体姿态 + 两只手腕的归一化坐标
SEQ_KEYS = [*PERSON_FEATURE_KEYS, "wl_x", "wl_y", "wr_x", "wr_y"]


def collect_sequences(manifest: Path, paths) -> dict[str, dict[int, dict[int, list[float]]]]:
    """→ {record_id: {track: {frame_idx: [SEQ_KEYS...]}}}，逐帧全量，不只是碰撞帧。"""
    out: dict[str, dict[int, dict[int, list[float]]]] = {}
    for ref in list_records_from_manifest(manifest, split_role=None):
        record = load_record(ref, paths)
        infer_h = int(record.meta.get("infer_height") or ref.infer_height or 1)
        infer_w = int(record.meta.get("infer_width") or ref.infer_width or 1)
        bank = FeatureBank(
            infer_width=infer_w, infer_height=infer_h, video_fps=float(record.fps or 15.0)
        )
        per_track: dict[int, dict[int, list[float]]] = defaultdict(dict)
        for fr in sorted(
            record.frames, key=lambda f: int(f.get("source_frame_idx") or f.get("frame_idx") or 0)
        ):
            key = int(fr.get("source_frame_idx") or fr.get("frame_idx") or 0)
            for row in bank.rows_for_frame(fr):
                person = row.get("_person")
                if not isinstance(person, dict):
                    continue
                track = int(row.get("person_track_id") or person.get("person_track_id") or 0)
                vec = [row.get(k) for k in PERSON_FEATURE_KEYS]
                for idx in (WRIST_LEFT, WRIST_RIGHT):
                    pt = read_xy(person, idx)
                    vec += [None, None] if pt is None else [pt[0] / infer_h, pt[1] / infer_h]
                per_track[track][key] = [np.nan if v is None else float(v) for v in vec]
        out[ref.record_id] = dict(per_track)
        print(f"[ok] {ref.record_id.split('/')[-1][:38]} tracks={len(per_track)}")
    return out


def _stats(seq: np.ndarray) -> list[float]:
    """把一维时间序列压成统计量。全 NaN 时返回 NaN，交给下游填充。"""
    v = seq[~np.isnan(seq)]
    if len(v) == 0:
        return [np.nan] * 7
    last = v[-1]
    if len(v) == 1:
        return [last, last, 0.0, last, last, 0.0, 0.0]
    slope = float(np.polyfit(np.arange(len(v)), v, 1)[0])
    half = len(v) // 2
    return [
        last,
        float(v.mean()),
        float(v.std()),
        float(v.min()),
        float(v.max()),
        slope,
        float(v[half:].mean() - v[:half].mean()),
    ]


def window_features(
    frames: dict[int, list[float]], t: int, win: int, step: int
) -> tuple[list[float], float]:
    """取 [t-win, t] 的序列并压成特征。返回（特征, 窗口内有骨架的帧占比）。"""
    keys = [k for k in range(t - win, t + 1, step) if k in frames]
    n_slots = len(range(t - win, t + 1, step))
    if not keys:
        return [np.nan] * (len(SEQ_KEYS) * 7 + 4), 0.0
    seq = np.array([frames[k] for k in keys], dtype=np.float64)

    feats: list[float] = []
    for j in range(len(SEQ_KEYS)):
        feats += _stats(seq[:, j])

    # 手腕轨迹的运动学量：走了多少路、是否原地折返、静止占比
    traj: list[float] = []
    for base in (5, 7):  # wl_x/wl_y、wr_x/wr_y 在 SEQ_KEYS 中的起始列
        xy = seq[:, base : base + 2]
        ok = ~np.isnan(xy).any(axis=1)
        if ok.sum() < 2:
            traj += [np.nan, np.nan]
            continue
        pts = xy[ok]
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        path = float(d.sum())
        net = float(np.linalg.norm(pts[-1] - pts[0]))
        traj += [path, net / path if path > 1e-6 else 0.0]
    feats += traj
    return feats, len(keys) / max(1, n_slots)


def oof(X: np.ndarray, y: np.ndarray, g: np.ndarray, *, model: str, folds: int) -> np.ndarray:
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, g):
        if model == "gbdt":
            clf = HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.06, max_depth=6, random_state=0
            )
            clf.fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        else:
            med = np.nanmedian(X[tr], axis=0)
            med = np.where(np.isnan(med), 0.0, med)
            Xtr = np.where(np.isnan(X[tr]), med, X[tr])
            Xte = np.where(np.isnan(X[te]), med, X[te])
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=4000, class_weight="balanced")
            clf.fit(sc.transform(Xtr), y[tr])
            p[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v1.json"))
    ap.add_argument("--samples", default=str(ROOT / "output/train/pairs_v4_wscore.samples.json"))
    ap.add_argument("--cache", default=str(ROOT / "output/train/action_sequences.npz"))
    ap.add_argument("--windows", default="12,25,50", help="窗口长度（源帧数，25fps 下 ≈0.5/1/2 秒）")
    ap.add_argument("--step", type=int, default=2, help="窗口内取帧步长")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "output/viz/clusters/temporal_action.md"))
    args = ap.parse_args()

    paths = load_paths()
    cache = Path(args.cache)
    if cache.exists():
        blob = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
        seqs = {
            r: {int(t): {int(f): v for f, v in fr.items()} for t, fr in tr.items()}
            for r, tr in blob.items()
        }
        print(f"[..] 复用序列缓存 {cache.with_suffix('.json')}")
    else:
        seqs = collect_sequences(Path(args.manifest), paths)
        cache.with_suffix(".json").write_text(
            json.dumps(seqs, ensure_ascii=False), encoding="utf-8"
        )
        cache.touch()
        print(f"[ok] 序列缓存 → {cache.with_suffix('.json')}")

    meta = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    # 人级去重，与 eval_action_only.py 同口径，AUC 可直接对比
    bucket: dict[tuple[str, int, str], int] = {}
    for m in meta:
        k = (m["record_id"], int(m["frame_idx"]), str(m["person_track_id"]))
        bucket[k] = max(bucket.get(k, 0), int(m["y"]))
    items = sorted(bucket.items())
    print(f"\n人级样本 {len(items)}（正 {sum(bucket.values())}）")

    lines = [
        "# 时序动作判别（只看过去的骨架序列，不看货框）",
        "",
        "对照：逐帧姿态 AUC 0.813（`eval_action_only.py`，同口径人级样本）。",
        "",
        "| 窗口 | 维数 | 骨架覆盖率 | AUC(逻辑回归) | AUC(GBDT) |",
        "|------|------|-----------|--------------|-----------|",
    ]
    results = []
    for win in [int(w) for w in args.windows.split(",") if w.strip()]:
        X, y, g, cov = [], [], [], []
        for (rid, t, track), label in items:
            frames = (seqs.get(rid) or {}).get(int(track)) or {}
            f, c = window_features(frames, t, win, args.step)
            X.append(f)
            y.append(label)
            g.append(rid)
            cov.append(c)
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=int)
        g = np.array(g)
        X = np.c_[X, np.array(cov)]

        a_lr = roc_auc_score(y, oof(X, y, g, model="lr", folds=args.folds))
        a_gb = roc_auc_score(y, oof(X, y, g, model="gbdt", folds=args.folds))
        mean_cov = float(np.mean(cov))
        print(f"窗口 {win:3d} 帧（{win / 25:.1f}秒）  {X.shape[1]:3d}维  "
              f"覆盖 {mean_cov:.1%}  逻辑回归 {a_lr:.4f}  GBDT {a_gb:.4f}")
        lines.append(
            f"| {win} 帧（{win / 25:.1f} 秒） | {X.shape[1]} | {mean_cov:.1%} "
            f"| {a_lr:.4f} | {a_gb:.4f} |"
        )
        results.append((win, a_lr, a_gb, X, y, g))

    # 最好的那档拿来看当第一道闸的过滤能力
    win, a_lr, a_gb, X, y, g = max(results, key=lambda r: max(r[1], r[2]))
    p = oof(X, y, g, model="gbdt" if a_gb >= a_lr else "lr", folds=args.folds)
    lines += [
        "",
        f"## 最好一档（{win} 帧）当第一道闸",
        "",
        "| 保留的真拣货 | 能滤掉的非拣货 |",
        "|------------|--------------|",
    ]
    print("\n当第一道闸：")
    for keep in (0.99, 0.97, 0.95, 0.90):
        t = np.quantile(p[y == 1], 1 - keep)
        filtered = (p[y == 0] < t).mean()
        print(f"  保住 {keep:.0%} 真拣货 → 滤掉 {filtered:.1%} 非拣货")
        lines.append(f"| {keep:.0%} | {filtered:.1%} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
