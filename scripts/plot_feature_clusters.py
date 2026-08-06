"""特征空间可分性诊断：拣货 / 拣隔壁框 / 无人拣货 三类到底分不分得开。

产出（output/viz/clusters/）：
  1_tsne.png        t-SNE 降维散点，看簇结构
  2_score_dist.png  OOF 分数分布，看判别力（交叉验证外样本，不是训练集内预测）
  3_features.png    判别力最强的单特征分布
  summary.md        两两 AUC 与重叠度
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KIND_POS, KIND_NEG_NEIGHBOR, KIND_NEG_OUTSIDE = 0, 1, 2
CLASSES = [
    (KIND_POS, "拣这个框（正）", "#d62728"),
    (KIND_NEG_NEIGHBOR, "同时刻在拣别的框", "#ff7f0e"),
    (KIND_NEG_OUTSIDE, "无人拣货", "#1f77b4"),
]


def _use_cjk_font() -> None:
    for name in ("Noto Sans CJK SC", "Noto Sans CJK TC", "WenQuanYi Zen Hei", "SimHei"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int) -> np.ndarray:
    """分组交叉验证的外样本预测。用训练集内预测会让分离度虚高。"""
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        out[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return out


def plot_tsne(X: np.ndarray, kind: np.ndarray, out: Path, *, sample: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    emb = TSNE(
        n_components=2, perplexity=30, init="pca", random_state=seed, max_iter=1000
    ).fit_transform(StandardScaler().fit_transform(X[idx]))
    k = kind[idx]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].scatter(
        *emb[k != KIND_POS].T, s=4, c="#1f77b4", alpha=0.35, label="非拣货（两类合并）"
    )
    axes[0].scatter(*emb[k == KIND_POS].T, s=4, c="#d62728", alpha=0.35, label="拣这个框")
    axes[0].set_title("拣货 vs 非拣货", fontsize=13)
    for kk, name, color in CLASSES:
        axes[1].scatter(*emb[k == kk].T, s=4, c=color, alpha=0.35, label=name)
    axes[1].set_title("负样本拆成两类看", fontsize=13)
    for ax in axes:
        ax.legend(markerscale=4, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"特征空间 t-SNE（抽样 {len(idx)} 个「人-货框」配对）", fontsize=15)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_scores(p: np.ndarray, kind: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    bins = np.linspace(0, 1, 51)
    for kk, name, color in CLASSES:
        axes[0].hist(
            p[kind == kk], bins=bins, alpha=0.55, color=color, label=name, density=True
        )
    axes[0].set_title("模型打分分布（交叉验证外样本）", fontsize=13)
    axes[0].set_xlabel("拣货概率")
    axes[0].set_ylabel("密度")
    axes[0].legend(fontsize=10)

    for kk, name, color in CLASSES:
        v = np.sort(p[kind == kk])
        axes[1].plot(v, np.linspace(0, 1, len(v)), color=color, lw=2, label=name)
    axes[1].set_title("累积分布：任一阈值下各类被判为拣货的比例", fontsize=13)
    axes[1].set_xlabel("阈值")
    axes[1].set_ylabel("低于阈值的占比")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_features(
    X: np.ndarray, y: np.ndarray, kind: np.ndarray, keys: list[str], out: Path, *, top: int
) -> list[tuple[str, float]]:
    aucs = sorted(
        ((k, roc_auc_score(y, X[:, i])) for i, k in enumerate(keys)),
        key=lambda kv: abs(kv[1] - 0.5),
        reverse=True,
    )
    picked = aucs[:top]
    ncol = 3
    nrow = (len(picked) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow))
    for ax, (key, auc) in zip(np.ravel(axes), picked):
        col = X[:, keys.index(key)]
        lo, hi = np.percentile(col, [1, 99])
        bins = np.linspace(lo, hi, 45)
        for kk, name, color in CLASSES:
            ax.hist(col[kind == kk], bins=bins, alpha=0.5, color=color, label=name, density=True)
        ax.set_title(f"{key}（单特征 AUC {auc:.3f}）", fontsize=11)
        ax.set_yticks([])
    for ax in np.ravel(axes)[len(picked) :]:
        ax.axis("off")
    np.ravel(axes)[0].legend(fontsize=9)
    fig.suptitle("判别力最强的单个特征：三类的分布重叠情况", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="output/train/pairs_v4_wscore.npz")
    ap.add_argument("--out-dir", default="output/viz/clusters")
    ap.add_argument("--sample", type=int, default=6000, help="t-SNE 抽样点数")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--top-features", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _use_cjk_font()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.dataset, allow_pickle=True)
    X, y, kind, groups = d["X"], d["y"].astype(int), d["kind"], d["groups"]
    keys = [str(k) for k in d["feature_keys"]]
    X = np.where(np.isnan(X), d["impute"], X)

    print(f"样本 {len(y)}：正 {(kind == KIND_POS).sum()}、"
          f"邻框负 {(kind == KIND_NEG_NEIGHBOR).sum()}、时段外负 {(kind == KIND_NEG_OUTSIDE).sum()}")

    p = oof_scores(X, y, groups, args.folds)
    m_pos, m_nb, m_out = kind == KIND_POS, kind == KIND_NEG_NEIGHBOR, kind == KIND_NEG_OUTSIDE
    auc_all = roc_auc_score(y, p)
    auc_nb = roc_auc_score(
        np.r_[np.ones(m_pos.sum()), np.zeros(m_nb.sum())], np.r_[p[m_pos], p[m_nb]]
    )
    auc_out = roc_auc_score(
        np.r_[np.ones(m_pos.sum()), np.zeros(m_out.sum())], np.r_[p[m_pos], p[m_out]]
    )

    print(f"OOF AUC 总体 {auc_all:.4f} | 正 vs 拣隔壁框 {auc_nb:.4f} | 正 vs 无人拣货 {auc_out:.4f}")

    plot_tsne(X, kind, out_dir / "1_tsne.png", sample=args.sample, seed=args.seed)
    plot_scores(p, kind, out_dir / "2_score_dist.png")
    top = plot_features(X, y, kind, keys, out_dir / "3_features.png", top=args.top_features)

    lines = [
        "# 特征空间可分性诊断",
        "",
        f"- 样本 {len(y)}：正 {m_pos.sum()}、拣隔壁框 {m_nb.sum()}、无人拣货 {m_out.sum()}",
        "",
        "## 两两可分性（OOF AUC，0.5=完全分不开，1.0=完美分开）",
        "",
        "| 对比 | AUC |",
        "|------|-----|",
        f"| 全部正 vs 全部负 | {auc_all:.4f} |",
        f"| 拣这个框 vs **同时刻在拣别的框** | **{auc_nb:.4f}** |",
        f"| 拣这个框 vs 无人拣货 | {auc_out:.4f} |",
        "",
        "## 阈值处的分错比例",
        "",
        "| 阈值 | 漏掉的真拣货 | 误判的邻框 | 误判的无人时段 |",
        "|------|------------|-----------|--------------|",
    ]
    for t in (0.1, 0.2, 0.3, 0.5, 0.7):
        lines.append(
            f"| {t:.1f} | {(p[m_pos] < t).mean():.1%} | "
            f"{(p[m_nb] >= t).mean():.1%} | {(p[m_out] >= t).mean():.1%} |"
        )
    lines += ["", "## 判别力最强的单特征", "", "| 特征 | 单特征 AUC |", "|------|-----------|"]
    lines += [f"| `{k}` | {a:.4f} |" for k, a in top]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"→ {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
