"""构建「人-货框」配对数据集。

标签口径复刻 collector 的 accuracy_service：verified_true 按连续相同 token 集合并成
区间段，配对落在段内且 token 匹配为正。这与导出包的评估指标同口径。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.box_geometry import PAIR_FEATURE_KEYS, compute_pair_features
from features.depth_geometry import DEPTH_FEATURE_KEYS, compute_depth_features, load_depth_cache
from features.pair_temporal import TEMPORAL_FEATURE_KEYS, PairTemporalTracker

PERSON_FEATURE_KEYS = [
    "ankle_max_speed_norm",
    "arm_torso_angle_max",
    "elbow_angle_mean",
    "wrist_elevation_angle_max",
    "shoulder_hip_knee_angle_min",
]
FEATURE_KEYS = PERSON_FEATURE_KEYS + PAIR_FEATURE_KEYS + TEMPORAL_FEATURE_KEYS

# 负样本细分，用于诊断而非训练
KIND_POS = 0
KIND_NEG_NEIGHBOR = 1  # 段内但框不对 → 邻框归属错
KIND_NEG_OUTSIDE = 2  # 段外 → 非拣货时刻


def build_ground_truth_segments(verified_true: list[dict[str, Any]]) -> list[tuple[frozenset, int, int]]:
    """连续相同 token 集的条目合并成 [frame_start, frame_end]。"""
    entries = sorted(
        (e for e in verified_true if isinstance(e, dict)),
        key=lambda e: int(e.get("frame_idx") or e.get("source_frame_idx") or 0),
    )
    segs: list[list[Any]] = []
    cur: list[Any] | None = None
    for entry in entries:
        tokens = frozenset(entry.get("confirmed_box_tokens") or entry.get("box_tokens") or [])
        if not tokens:
            continue
        fi = int(entry.get("frame_idx") or entry.get("source_frame_idx") or 0)
        if fi <= 0:
            continue
        if cur is not None and cur[0] == tokens:
            cur[2] = max(cur[2], fi)
        else:
            if cur is not None:
                segs.append(cur)
            cur = [tokens, fi, fi]
    if cur is not None:
        segs.append(cur)
    return [(s[0], s[1], s[2]) for s in segs]


def _review_key_map(report_path: Path) -> dict[str, str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for clip in report.get("clips") or []:
        rid = str(clip.get("record_id") or "")
        rk = str(clip.get("review_key") or "")
        if rid and rk:
            out[rid] = rk
    return out


def _baseline_frame_indices(paths, file_name: str) -> set[int] | None:
    clip = paths.baseline_manifest.parent / file_name
    if not clip.is_file():
        return None
    rows = json.loads(clip.read_text(encoding="utf-8"))
    return {int(r.get("frame_idx") or 0) for r in rows if isinstance(r, dict)}


def build(*, paths, report_path: Path, depth_dir: Path | None = None) -> dict[str, Any]:
    from adapters.record_reader import list_records, load_record
    from features.bank import FeatureBank
    from pipeline.box_trigger import BoxTrigger

    review_map = _review_key_map(report_path)
    feature_keys = list(FEATURE_KEYS) + (list(DEPTH_FEATURE_KEYS) if depth_dir else [])

    rows: list[list[float | None]] = []
    y_list: list[int] = []
    kind_list: list[int] = []
    groups: list[str] = []
    seg_ids: list[str] = []
    meta: list[dict[str, Any]] = []
    n_segments_total = 0

    for ref in list_records(paths):
        review_key = review_map.get(ref.record_id)
        if not review_key:
            continue
        review_path = paths.review_dir / review_key / "event_review.json"
        if not review_path.is_file():
            continue
        segs = build_ground_truth_segments(
            json.loads(review_path.read_text(encoding="utf-8")).get("verified_true") or []
        )
        n_segments_total += len(segs)

        record = load_record(ref, paths)
        infer_h = int(record.meta.get("infer_height") or ref.infer_height or 1)
        bank = FeatureBank(
            infer_width=int(record.meta.get("infer_width") or ref.infer_width or 1),
            infer_height=infer_h,
            video_fps=float(record.fps or 15.0),
        )
        trigger = BoxTrigger(record.boxes)
        box_by_token = {b.token: b for b in record.boxes}
        depth_cache = load_depth_cache(depth_dir, ref.record_id) if depth_dir else {}
        temporal = PairTemporalTracker(
            infer_width=int(record.meta.get("infer_width") or ref.infer_width or 1),
            infer_height=infer_h,
            video_fps=float(record.fps or 15.0),
        )

        by_key = {
            int(fr.get("source_frame_idx") or fr.get("frame_idx") or 0): fr for fr in record.frames
        }
        frame_indices = _baseline_frame_indices(paths, ref.file_name)
        keys = sorted(frame_indices) if frame_indices is not None else sorted(by_key)

        for export_key in keys:
            # 无骨架的帧也要推进 tracker，否则停留计数不清零；与 runner 的空帧处理一致
            frame = by_key.get(export_key) or {
                "frame_idx": export_key,
                "source_frame_idx": export_key,
                "timestamp_sec": 0.0,
                "persons": [],
            }
            pending: list[tuple[dict[str, Any], int, dict[str, Any], dict[str, Any], str]] = []
            active_pairs: dict[str, dict[str, Any]] = {}
            # 速度特征依赖逐帧推进，所以整帧都要过一遍 bank
            for row in bank.rows_for_frame(frame):
                person = row.get("_person")
                if not isinstance(person, dict):
                    continue
                track = int(row.get("person_track_id") or 0)
                for hit in trigger.hits_for_person(person):
                    box = box_by_token.get(hit["token"])
                    if box is None:
                        continue
                    pair = compute_pair_features(person, hit, box, infer_height=infer_h)
                    pair_key = f"{track}|{hit['token']}"
                    active_pairs[pair_key] = {
                        "depth_ratio": hit["depth_ratio"],
                        "center_dist_norm": pair.get("center_dist_norm"),
                        "wrist_xy": hit["wrist_xy"],
                    }
                    pending.append((row, track, hit, pair, pair_key))

            temporal_feats = temporal.update(export_key, active_pairs)
            active = [(i, s) for i, s in enumerate(segs) if s[1] <= export_key <= s[2]]

            for row, track, hit, pair, pair_key in pending:
                tmp = temporal_feats.get(pair_key) or {}
                vec: list[float | None] = []
                for key in PERSON_FEATURE_KEYS:
                    v = row.get(key)
                    vec.append(None if v is None else float(v))
                for key in PAIR_FEATURE_KEYS:
                    v = pair.get(key)
                    vec.append(None if v is None else float(v))
                for key in TEMPORAL_FEATURE_KEYS:
                    v = tmp.get(key)
                    vec.append(None if v is None else float(v))
                if depth_dir:
                    dep = compute_depth_features(
                        depth_cache.get((export_key, track, hit["token"]))
                    )
                    for key in DEPTH_FEATURE_KEYS:
                        v = dep.get(key)
                        vec.append(None if v is None else float(v))

                matched = [i for i, s in active if hit["token"] in s[0]]
                if matched:
                    label, kind, seg_id = 1, KIND_POS, matched[0]
                elif active:
                    label, kind, seg_id = 0, KIND_NEG_NEIGHBOR, -1
                else:
                    label, kind, seg_id = 0, KIND_NEG_OUTSIDE, -1

                rows.append(vec)
                y_list.append(label)
                kind_list.append(kind)
                groups.append(ref.record_id)
                seg_ids.append(f"{ref.record_id}#{seg_id}" if seg_id >= 0 else "")
                meta.append(
                    {
                        "record_id": ref.record_id,
                        "frame_idx": export_key,
                        "token": hit["token"],
                        "person_track_id": track,
                        "y": label,
                        "kind": kind,
                        "seg_id": seg_id,
                    }
                )

        print(f"[ok] {ref.record_id.split('/')[-1]} pairs={len(rows)}")

    X = np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=np.float64)
    # 缺失按列中位数补，并记录缺失率供诊断
    missing_rate = np.isnan(X).mean(axis=0) if len(X) else np.zeros(len(feature_keys))
    for j in range(X.shape[1] if len(X) else 0):
        col = X[:, j]
        if np.isnan(col).any():
            fill = np.nanmedian(col)
            col[np.isnan(col)] = 0.0 if np.isnan(fill) else fill

    y = np.asarray(y_list, dtype=np.int32)
    return {
        "X": X,
        "y": y,
        "kind": np.asarray(kind_list, dtype=np.int32),
        "groups": np.asarray(groups),
        "seg_ids": np.asarray(seg_ids),
        "n_segments_total": n_segments_total,
        "feature_keys": feature_keys,
        "person_feature_keys": list(PERSON_FEATURE_KEYS),
        "pair_feature_keys": list(PAIR_FEATURE_KEYS),
        "temporal_feature_keys": list(TEMPORAL_FEATURE_KEYS),
        "depth_feature_keys": list(DEPTH_FEATURE_KEYS) if depth_dir else [],
        "missing_rate": {k: round(float(r), 4) for k, r in zip(feature_keys, missing_rate)},
        "meta": meta,
    }


def main() -> int:
    import argparse

    from adapters.collector_paths import load_paths

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report",
        default=str(ROOT / "output/export/pickstate-logistic-v1-t025-prod-test/accuracy_report.json"),
    )
    ap.add_argument("--out", default=str(ROOT / "output/train/pairs_v1.npz"))
    ap.add_argument(
        "--depth-dir",
        default="",
        help="深度标量缓存目录（scripts/precompute_depth.py 的产物）；留空则不带深度特征",
    )
    args = ap.parse_args()

    depth_dir = Path(args.depth_dir) if args.depth_dir.strip() else None
    if depth_dir is not None and not depth_dir.is_absolute():
        depth_dir = ROOT / depth_dir
    ds = build(paths=load_paths(), report_path=Path(args.report), depth_dir=depth_dir)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=ds["X"],
        y=ds["y"],
        kind=ds["kind"],
        groups=ds["groups"],
        seg_ids=ds["seg_ids"],
        feature_keys=np.asarray(ds["feature_keys"]),
        person_feature_keys=np.asarray(ds["person_feature_keys"]),
        pair_feature_keys=np.asarray(ds["pair_feature_keys"]),
        temporal_feature_keys=np.asarray(ds["temporal_feature_keys"]),
        depth_feature_keys=np.asarray(ds["depth_feature_keys"]),
    )
    kind = ds["kind"]
    summary = {
        "n": int(len(ds["y"])),
        "n_pos": int(ds["y"].sum()),
        "n_neg_neighbor": int((kind == KIND_NEG_NEIGHBOR).sum()),
        "n_neg_outside": int((kind == KIND_NEG_OUTSIDE).sum()),
        "n_records": int(len(set(ds["groups"].tolist()))),
        "n_segments_total": ds["n_segments_total"],
        "n_segments_covered": int(len({s for s in ds["seg_ids"].tolist() if s})),
        "feature_keys": ds["feature_keys"],
        "missing_rate": ds["missing_rate"],
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n配对数据集 → {out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
