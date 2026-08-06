"""L2 logistic：record 分组 CV + 全量重拟合，导出权重与重要性。"""

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


def _fit(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(
        penalty="l2",
        C=C,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
    )
    clf.fit(Xs, y)
    return scaler, clf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "output/train/dataset_v1.npz"))
    ap.add_argument("--out-dir", default=str(ROOT / "output/train/logistic_v1"))
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--name", default="logistic_v1")
    ap.add_argument(
        "--features",
        default="",
        help="逗号分隔的特征子集；留空用数据集全部特征",
    )
    args = ap.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(int)
    groups = data["groups"]
    feature_keys = [str(x) for x in data["feature_keys"].tolist()]
    impute = (
        np.asarray(data["impute"], dtype=np.float64)
        if "impute" in data.files
        else np.zeros(len(feature_keys))
    )

    if args.features.strip():
        wanted = [k.strip() for k in args.features.split(",") if k.strip()]
        unknown = [k for k in wanted if k not in feature_keys]
        if unknown:
            raise SystemExit(f"数据集中不存在这些特征：{unknown}")
        cols = [feature_keys.index(k) for k in wanted]
        X = X[:, cols]
        impute = impute[cols]
        feature_keys = wanted

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_splits = min(args.folds, len(set(groups.tolist())))
    gkf = GroupKFold(n_splits=n_splits)
    fold_rows = []
    oof = np.full(len(y), np.nan)

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        scaler, clf = _fit(X[tr], y[tr], C=args.C)
        prob = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        oof[te] = prob
        auc = roc_auc_score(y[te], prob) if len(set(y[te].tolist())) > 1 else float("nan")
        ap = average_precision_score(y[te], prob) if y[te].sum() > 0 else float("nan")
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "pos_test": int(y[te].sum()),
                "auc": round(float(auc), 4),
                "ap": round(float(ap), 4),
            }
        )
        print(f"fold {fold}: AUC={auc:.4f} AP={ap:.4f} pos_test={int(y[te].sum())}/{len(te)}")

    # 全量重拟合
    scaler, clf = _fit(X, y, C=args.C)
    coef = clf.coef_.ravel()
    importance = []
    for name, c, mean, scale in zip(feature_keys, coef, scaler.mean_, scaler.scale_):
        importance.append(
            {
                "feature": name,
                "coef_std": round(float(c), 4),  # 标准化空间系数
                "abs_coef": round(abs(float(c)), 4),
                "mean": round(float(mean), 6),
                "scale": round(float(scale), 6),
                "coef_raw": round(float(c) / float(scale), 6) if scale else None,
                "direction": "↑提高拣货分" if c > 0 else "↓降低拣货分",
            }
        )
    importance.sort(key=lambda r: r["abs_coef"], reverse=True)

    model = {
        "name": args.name,
        "feature_keys": feature_keys,
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
        "impute": [float(x) for x in impute],
        "coef": [float(x) for x in coef],
        "intercept": float(clf.intercept_.ravel()[0]),
        "C": args.C,
        "class_weight": "balanced",
        "n_samples": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
    }
    (out_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "feature_importance.json").write_text(
        json.dumps(importance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "cv_metrics.json").write_text(
        json.dumps({"folds": fold_rows, "oof_auc": round(float(roc_auc_score(y, oof)), 4)}, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"# {args.name} 特征权重（标准化空间 |coef| 排序）",
        "",
        f"- 样本：{len(y)}（正 {int(y.sum())} / 负 {int(len(y) - y.sum())}）",
        f"- GroupKFold OOF AUC：{roc_auc_score(y, oof):.4f}",
        "",
        "| 特征 | coef(标准化) | 方向 |",
        "|------|-------------|------|",
    ]
    for row in importance:
        lines.append(f"| `{row['feature']}` | {row['coef_std']:+.4f} | {row['direction']} |")
    lines.append("")
    (out_dir / "feature_importance.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n模型 → {out_dir / 'model.json'}")
    print((out_dir / "feature_importance.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
