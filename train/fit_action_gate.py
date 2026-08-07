#!/usr/bin/env python3
"""训练并导出动作门控模型（GBDT）。

样本：分数文件里出现过碰撞的人-帧；标签=该人是否落在任一真值拣货段内。
特征：过去窗口骨架统计量（features.action_temporal）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_paths
from features.action_temporal import window_features
from scripts.eval_action_gate import _load_gt, _person_positive
from scripts.eval_temporal_action import collect_sequences


def main() -> int:
    ap = argparse.ArgumentParser(description="训练动作门控模型")
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v2.json"))
    ap.add_argument("--scores", default=str(ROOT / "output/scores/v5"))
    ap.add_argument("--seq-cache", default=str(ROOT / "output/train/action_sequences_v2.json"))
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out-dir", default=str(ROOT / "output/train/action_gate_v1"))
    ap.add_argument("--name", default="action_gate_v1")
    args = ap.parse_args()

    paths = load_paths()
    man_path = Path(args.manifest)
    gt = _load_gt(man_path)
    cache = Path(args.seq_cache)
    if cache.is_file():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        seqs = {
            r: {int(t): {int(f): v for f, v in fr.items()} for t, fr in tr.items()}
            for r, tr in raw.items()
        }
        print(f"[..] 复用序列 {cache}")
    else:
        seqs = collect_sequences(man_path, paths)
        cache.write_text(json.dumps(seqs, ensure_ascii=False), encoding="utf-8")

    rows = []
    for p in sorted(Path(args.scores).glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rid = d["record_id"]
        segs = gt.get(rid) or []
        if not segs:
            continue
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
                        "y": int(_person_positive(segs, fi, track)),
                        "g": rid,
                        "feat": feat + [cov],
                    }
                )

    X = np.array([r["feat"] for r in rows], dtype=np.float64)
    y = np.array([r["y"] for r in rows], dtype=int)
    g = np.array([r["g"] for r in rows])
    print(f"样本 {len(y)}  正 {int(y.sum())}（{y.mean():.1%}）")

    # OOF 报告
    p_oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=args.folds).split(X, y, g):
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_depth=6, random_state=0
        )
        clf.fit(X[tr], y[tr])
        p_oof[te] = clf.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, p_oof))
    print(f"GroupKFold OOF AUC: {auc:.4f}")

    # 全量重训导出
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6, random_state=0
    )
    clf.fit(X, y)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.joblib"
    joblib.dump(clf, model_path)
    meta = {
        "name": args.name,
        "model_path": str(model_path.relative_to(ROOT)),
        "window_frames": args.window,
        "step": args.step,
        "oof_auc": auc,
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "manifest": str(args.manifest),
        "scores": str(args.scores),
        "default_threshold": 0.30,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {model_path}\n→ {out / 'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
