"""渲染「标注说没人拣货、模型却高度确信在拣货」的帧，用于判断是标注漏标还是真误报。

样本级 meta 由 train.build_pair_dataset.build() 产出（内部保留帧号与 token，只是平时不写盘），
首次运行会缓存到 <dataset>.samples.json。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_paths
from adapters.record_reader import list_records_from_manifest, load_record, record_dir
from scripts.render_fn_fp_frames import _read_frame, _resolve_video
from scripts.render_missed_segments import _banner, _draw_boxes, _draw_person
from train.build_pair_dataset import build

KIND_NAME = {0: "拣这个框", 1: "拣隔壁框", 2: "无人拣货"}


def load_samples(dataset: Path, manifest: Path, wrist_min: float, paths) -> list[dict[str, Any]]:
    cache = dataset.with_suffix(".samples.json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    print("[..] 首次运行，重建样本级 meta（几分钟）")
    ds = build(paths=paths, manifest_path=manifest, wrist_score_min=wrist_min)
    meta = ds["meta"]
    cache.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] 缓存 → {cache}")
    return meta


def oof_scores(dataset: Path) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    d = np.load(dataset, allow_pickle=True)
    X = np.where(np.isnan(d["X"]), d["impute"], d["X"])
    y, groups = d["y"].astype(int), d["groups"]
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def pick_frames(rows: list[dict[str, Any]], *, gap: int, per_group: int) -> list[dict[str, Any]]:
    """按时间聚成连续片段，每片取分数最高的几帧，避免渲染一堆相邻重复画面。"""
    rows = sorted(rows, key=lambda r: r["frame_idx"])
    out: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    for r in rows:
        if cur and r["frame_idx"] - cur[-1]["frame_idx"] > gap:
            out.extend(sorted(cur, key=lambda x: -x["score"])[:per_group])
            cur = []
        cur.append(r)
    if cur:
        out.extend(sorted(cur, key=lambda x: -x["score"])[:per_group])
    return sorted(out, key=lambda r: r["frame_idx"])


def main() -> int:
    ap = argparse.ArgumentParser(description="渲染高分负样本")
    ap.add_argument("--dataset", default=str(ROOT / "output/train/pairs_v4_wscore.npz"))
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v1.json"))
    ap.add_argument("--wrist-min", type=float, default=0.15)
    ap.add_argument("--kind", type=int, default=2, help="2=无人拣货时段，1=拣隔壁框")
    ap.add_argument("--min-score", type=float, default=0.8)
    ap.add_argument("--record", default="", help="只渲染 record_id 含该子串的切片")
    ap.add_argument("--gap", type=int, default=40, help="帧号间隔超过它就切成新片段")
    ap.add_argument("--per-group", type=int, default=1)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "output/viz/high_score_neg"))
    args = ap.parse_args()

    dataset, manifest = Path(args.dataset), Path(args.manifest)
    paths = load_paths()
    meta = load_samples(dataset, manifest, args.wrist_min, paths)
    scores = oof_scores(dataset)
    if len(meta) != len(scores):
        print(f"[!!] meta {len(meta)} 与数据集 {len(scores)} 不一致，重建缓存后重试")
        return 1

    rows = [
        {**m, "score": float(s)}
        for m, s in zip(meta, scores)
        if m["kind"] == args.kind
        and s >= args.min_score
        and (not args.record or args.record in m["record_id"])
    ]
    if not rows:
        print("[--] 没有符合条件的样本")
        return 0

    by_rid: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_rid.setdefault(r["record_id"], []).append(r)
    print(f"命中 {len(rows)} 个配对，分布在 {len(by_rid)} 条切片")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    refs = {r.record_id: r for r in list_records_from_manifest(manifest, split_role=None)}

    n = 0
    for rid, rs in sorted(by_rid.items(), key=lambda kv: -len(kv[1])):
        ref = refs.get(rid)
        if ref is None:
            continue
        picked = pick_frames(rs, gap=args.gap, per_group=args.per_group)[: args.limit - n]
        if not picked:
            continue
        record = load_record(ref, paths)
        by_f = {
            int(f.get("source_frame_idx") or f.get("frame_idx") or 0): f for f in record.frames
        }
        video = _resolve_video(
            rid,
            json.loads((record_dir(ref, paths) / "manifest.json").read_text(encoding="utf-8")),
            paths,
        )
        if video is None:
            print(f"[--] 找不到视频，跳过 {rid}")
            continue
        short = rid.split("/")[-1][:34]

        for r in picked:
            fidx = int(r["frame_idx"])
            img = _read_frame(video, fidx)
            if img is None:
                continue
            frame = by_f.get(fidx) or {}
            persons = frame.get("persons") or []
            _draw_boxes(img, record.boxes, gt=set(), touched={r["token"]})
            for pr in persons:
                _draw_person(
                    img,
                    pr,
                    is_main=str(pr.get("person_track_id")) == str(r["person_track_id"]),
                    wrist_min=args.wrist_min,
                )
            _banner(
                img,
                [
                    f"{short}  frame {fidx}",
                    f"model score {r['score']:.3f}  ->  {r['token']}  person #{r['person_track_id']}",
                    f"annotation says: {KIND_NAME[args.kind]} (this frame is labeled NOT picking)",
                ],
            )
            name = f"{short}_f{fidx:06d}_s{int(r['score'] * 1000):03d}.jpg"
            cv2.imwrite(str(out_root / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            n += 1
        if n >= args.limit:
            break

    print(f"[ok] 渲染 {n} 张 → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
