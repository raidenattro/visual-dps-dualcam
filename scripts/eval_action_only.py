"""验证两阶段思路：先判「这个人在不在做拣货动作」（不看框），再判碰哪个框。

第一阶段可行的前提是动作本身可分。这里把配对样本按（切片, 帧, 人）聚合成人级样本，
比较三组特征的判别力：纯人体动作 / 人体+时序 / 全部（含框内位置）。
同时用 GBDT 对照 logistic，看是不是线性模型限制了上限。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 与框的几何关系无关的特征。side 系列依赖「哪只手进了框」，算半个框相关，单列一组。
BOX_FREE = [
    "ankle_max_speed_norm",
    "arm_torso_angle_max",
    "elbow_angle_mean",
    "wrist_elevation_angle_max",
    "shoulder_hip_knee_angle_min",
    "person_height_norm",
    "stance_gap_norm",
    "stance_valid",
]
TEMPORAL_BOX_FREE = ["wrist_speed_norm", "wrist_speed_win_mean"]
SIDE = ["arm_torso_angle_side", "elbow_angle_side", "wrist_elevation_angle_side"]


def oof(X: np.ndarray, y: np.ndarray, g: np.ndarray, *, model: str, folds: int) -> np.ndarray:
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, g):
        if model == "gbdt":
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, max_depth=6, random_state=0
            )
            clf.fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        else:
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=3000, class_weight="balanced")
            clf.fit(sc.transform(X[tr]), y[tr])
            p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "output/train/pairs_v4_wscore.npz"))
    ap.add_argument("--samples", default=str(ROOT / "output/train/pairs_v4_wscore.samples.json"))
    ap.add_argument("--out", default=str(ROOT / "output/viz/clusters/action_only.md"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    d = np.load(args.dataset, allow_pickle=True)
    X = np.where(np.isnan(d["X"]), d["impute"], d["X"])
    y, groups = d["y"].astype(int), d["groups"]
    keys = [str(k) for k in d["feature_keys"]]
    meta = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    if len(meta) != len(y):
        print("[!!] samples.json 与数据集不一致")
        return 1

    idx = {k: i for i, k in enumerate(keys)}
    groups_of = lambda names: [idx[n] for n in names if n in idx]  # noqa: E731

    # 人级聚合：同一（切片, 帧, 人）的多个配对合成一条，只要有一个是真拣货就算在拣货。
    # 特征取各配对的均值——纯人体特征在同一人同一帧本来就相同，均值不改变取值。
    bucket: dict[tuple[str, int, str], list[int]] = {}
    for i, m in enumerate(meta):
        bucket.setdefault(
            (m["record_id"], int(m["frame_idx"]), str(m["person_track_id"])), []
        ).append(i)
    rows = sorted(bucket.items())
    Xp = np.stack([X[ix].mean(axis=0) for _, ix in rows])
    yp = np.array([int(y[ix].max()) for _, ix in rows])
    gp = np.array([k[0] for k, _ in rows])
    npairs = np.array([len(ix) for _, ix in rows])

    print(f"配对样本 {len(y)} → 人级样本 {len(yp)}（正 {yp.sum()} / 负 {(1 - yp).sum()}）")
    print(f"平均每人每帧命中 {npairs.mean():.2f} 个货框\n")

    combos = [
        ("纯人体动作（完全不看框）", groups_of(BOX_FREE)),
        ("人体动作 + 手腕速度", groups_of(BOX_FREE + TEMPORAL_BOX_FREE)),
        ("再加触发手角度（半依赖框）", groups_of(BOX_FREE + TEMPORAL_BOX_FREE + SIDE)),
        ("全部特征（含框内位置）", list(range(len(keys)))),
    ]

    lines = [
        "# 两阶段可行性：只看动作能不能判断在拣货",
        "",
        f"人级样本 {len(yp)}（正 {yp.sum()} / 负 {(1 - yp).sum()}），"
        f"由 {len(y)} 个「人-货框」配对聚合而成。",
        "",
        "| 特征组 | 维数 | AUC(逻辑回归) | AUC(GBDT) |",
        "|--------|------|--------------|-----------|",
    ]
    for name, cols in combos:
        if not cols:
            continue
        a_lr = roc_auc_score(yp, oof(Xp[:, cols], yp, gp, model="lr", folds=args.folds))
        a_gb = roc_auc_score(yp, oof(Xp[:, cols], yp, gp, model="gbdt", folds=args.folds))
        print(f"{name:<26} {len(cols):>3}维  逻辑回归 {a_lr:.4f}  GBDT {a_gb:.4f}")
        lines.append(f"| {name} | {len(cols)} | {a_lr:.4f} | {a_gb:.4f} |")

    # 第一阶段作为前置过滤能滤掉多少：在保住绝大多数真拣货的前提下看能砍掉多少负样本
    cols = groups_of(BOX_FREE + TEMPORAL_BOX_FREE)
    p = oof(Xp[:, cols], yp, gp, model="gbdt", folds=args.folds)
    lines += [
        "",
        "## 纯动作模型当第一道闸的效果（GBDT，人体+速度）",
        "",
        "| 保留的真拣货 | 对应阈值 | 能滤掉的非拣货 |",
        "|------------|---------|--------------|",
    ]
    print("\n纯动作模型当前置过滤：")
    for keep in (0.99, 0.97, 0.95, 0.90):
        t = np.quantile(p[yp == 1], 1 - keep)
        filtered = (p[yp == 0] < t).mean()
        print(f"  保住 {keep:.0%} 真拣货 → 滤掉 {filtered:.1%} 非拣货")
        lines.append(f"| {keep:.0%} | {t:.3f} | {filtered:.1%} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
