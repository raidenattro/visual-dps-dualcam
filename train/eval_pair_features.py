"""配对数据集上的特征消融：只看几何特征带来多少增益。

对照组是「现有 5 维人体特征」，在同一份配对样本上跑，保证可比。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KIND_NEG_NEIGHBOR = 1
KIND_NEG_OUTSIDE = 2


def _oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, C: float, folds: int) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=min(folds, len(set(groups.tolist()))))
    for tr, te in gkf.split(X, y, groups):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(scaler.transform(X[te]))[:, 1]
    return oof


def _sweep(
    y: np.ndarray,
    oof: np.ndarray,
    seg_ids: np.ndarray,
    n_segments: int,
    thresholds: list[float],
) -> list[dict[str, float]]:
    """段级召回 + 配对级误报，口径贴近 collector 评估器（未含告警连续帧平滑）。"""
    out = []
    for thr in thresholds:
        fired = oof >= thr
        hit_segs = {s for s, f in zip(seg_ids.tolist(), fired) if f and s}
        out.append(
            {
                "threshold": thr,
                "seg_recall": round(len(hit_segs) / max(1, n_segments), 4),
                "seg_detected": len(hit_segs),
                "fp_pairs": int((fired & (y == 0)).sum()),
                "tp_pairs": int((fired & (y == 1)).sum()),
            }
        )
    return out


def _subset_auc(y: np.ndarray, kind: np.ndarray, oof: np.ndarray, neg_kind: int) -> float:
    """只保留正样本 + 指定类型的负样本，看模型在这一类负样本上的区分力。"""
    mask = (y == 1) | (kind == neg_kind)
    if len(set(y[mask].tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], oof[mask]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "output/train/pairs_v1.npz"))
    ap.add_argument("--out-dir", default=str(ROOT / "output/train/pair_ablation"))
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(int)
    kind = data["kind"].astype(int)
    groups = data["groups"]
    keys = [str(k) for k in data["feature_keys"].tolist()]
    person_keys = [str(k) for k in data["person_feature_keys"].tolist()]
    pair_keys = [str(k) for k in data["pair_feature_keys"].tolist()]
    idx = {k: i for i, k in enumerate(keys)}

    # 站位代理 = 免标定的「人离相机多远」；框内 = 手腕在框里的位置
    stance_keys = ["stance_gap_norm", "person_height_norm", "box_bottom_y_norm", "stance_valid"]
    inbox_keys = ["depth_ratio", "center_dist_norm", "wrist_bearing_x", "wrist_bearing_y"]
    # 上线的 11 维 = 全部 − 站位代理
    prod_keys = [k for k in keys if k not in stance_keys]

    variants = {
        "人体特征（对照基线）": person_keys,
        "几何特征": pair_keys,
        "人体 + 站位代理": person_keys + stance_keys,
        "人体 + 框内位置": person_keys + inbox_keys,
        "上线 11 维": prod_keys,
        "全部 − 框内位置": [k for k in keys if k not in inbox_keys],
        "全部": keys,
    }

    results = []
    oofs = {}
    for name, subset in variants.items():
        cols = [idx[k] for k in subset]
        oof = _oof(X[:, cols], y, groups, C=args.C, folds=args.folds)
        oofs[name] = oof
        results.append(
            {
                "variant": name,
                "n_features": len(subset),
                "oof_auc": round(float(roc_auc_score(y, oof)), 4),
                "oof_ap": round(float(average_precision_score(y, oof)), 4),
                "auc_vs_neighbor": round(_subset_auc(y, kind, oof, KIND_NEG_NEIGHBOR), 4),
                "auc_vs_outside": round(_subset_auc(y, kind, oof, KIND_NEG_OUTSIDE), 4),
            }
        )

    # 全特征模型的系数
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=2000, solver="lbfgs")
    clf.fit(Xs, y)
    coefs = sorted(
        (
            {"feature": k, "coef_std": round(float(c), 4)}
            for k, c in zip(keys, clf.coef_.ravel())
        ),
        key=lambda r: abs(r["coef_std"]),
        reverse=True,
    )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_nb = int((kind == KIND_NEG_NEIGHBOR).sum())
    n_out = int((kind == KIND_NEG_OUTSIDE).sum())
    lines = [
        "# 配对特征消融（GroupKFold by record，28 组）",
        "",
        f"样本 {len(y)}：正 {int(y.sum())} / 负-邻框 {n_nb} / 负-段外 {n_out}",
        "",
        "| 特征组 | 维数 | OOF AUC | OOF AP | AUC(正 vs 邻框负) | AUC(正 vs 段外负) |",
        "|--------|------|---------|--------|------------------|------------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['variant']} | {r['n_features']} | {r['oof_auc']:.4f} | {r['oof_ap']:.4f} | "
            f"{r['auc_vs_neighbor']:.4f} | {r['auc_vs_outside']:.4f} |"
        )
    # 阈值扫描：对照基线 vs 推荐特征组，看同召回下误报差多少
    seg_ids = data["seg_ids"] if "seg_ids" in data else np.asarray([""] * len(y))
    n_segments = len({s for s in seg_ids.tolist() if s})
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    sweep_names = [n for n in ("人体特征（对照基线）", "上线 11 维", "全部") if n in oofs]
    sweeps = {name: _sweep(y, oofs[name], seg_ids, n_segments, thresholds) for name in sweep_names}
    lines += [
        "",
        f"## 阈值扫描（真值段 {n_segments}，配对级误报）",
        "",
        "| 阈值 | " + " | ".join(f"{n} 召回 / FP" for n in sweeps) + " |",
        "|------|" + "|".join(["------"] * len(sweeps)) + "|",
    ]
    for i, thr in enumerate(thresholds):
        cells = [f"{sweeps[n][i]['seg_recall']:.1%} / {sweeps[n][i]['fp_pairs']}" for n in sweeps]
        lines.append(f"| {thr:.2f} | " + " | ".join(cells) + " |")

    lines += ["", "## 全特征模型系数（标准化空间）", "", "| 特征 | coef |", "|------|------|"]
    for c in coefs:
        lines.append(f"| `{c['feature']}` | {c['coef_std']:+.4f} |")
    lines.append("")
    text = "\n".join(lines)
    (out_dir / "ablation.md").write_text(text, encoding="utf-8")
    (out_dir / "ablation.json").write_text(
        json.dumps(
            {"variants": results, "coefs": coefs, "sweeps": sweeps}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(text)
    print(f"→ {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
