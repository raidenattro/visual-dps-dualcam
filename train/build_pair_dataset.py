"""构建「人-货框」配对数据集。

schema2：正样本匹配 (frame, person_track_id, box)。
段级划分：--manifest 时只用 train 段作正样本，val 段时间窗整帧不进训练集。
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

from adapters.frame_sampling import sample_frame_indices
from adapters.review_labels import normalize_verified_true
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

KIND_POS = 0
KIND_NEG_NEIGHBOR = 1
KIND_NEG_OUTSIDE = 2


def build_ground_truth_segments(verified_true: list[dict[str, Any]]) -> list[tuple[frozenset, int, int]]:
    """兼容旧接口：仅按 token 集合并（无 track）。"""
    entries = normalize_verified_true(verified_true)
    segs: list[list[Any]] = []
    cur: list[Any] | None = None
    for entry in entries:
        tokens = frozenset(entry.get("confirmed_box_tokens") or [])
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


def _track_match(seg_tracks: list[str] | tuple[str, ...], person_track: int | str) -> bool:
    """段未标 track 时不过滤人；标了则必须命中。"""
    if not seg_tracks:
        return True
    pt = str(person_track).strip()
    return pt in {str(t).strip() for t in seg_tracks}


def _baseline_frame_indices(paths, file_name: str) -> set[int] | None:
    clip = paths.baseline_manifest.parent / file_name
    if not clip.is_file():
        return None
    rows = json.loads(clip.read_text(encoding="utf-8"))
    return {int(r.get("frame_idx") or 0) for r in rows if isinstance(r, dict)}


def build(
    *,
    paths,
    report_path: Path | None = None,
    manifest_path: Path | None = None,
    depth_dir: Path | None = None,
    sample_fps: float = 0.0,
    wrist_score_min: float = 0.3,
) -> dict[str, Any]:
    from adapters.record_reader import (
        list_records,
        list_records_from_manifest,
        load_record,
        load_segment_split,
        review_key_map_from_manifest,
    )
    from features.bank import FeatureBank
    from pipeline.box_trigger import BoxTrigger

    feature_keys = list(FEATURE_KEYS) + (list(DEPTH_FEATURE_KEYS) if depth_dir else [])
    rows: list[list[float | None]] = []
    y_list: list[int] = []
    kind_list: list[int] = []
    groups: list[str] = []
    seg_ids: list[str] = []
    meta: list[dict[str, Any]] = []
    n_segments_total = 0
    n_train_segments = 0
    n_skipped_val_frames = 0

    if manifest_path is not None:
        refs = list_records_from_manifest(manifest_path)  # 全部 record（段可能跨 split）
        review_map = review_key_map_from_manifest(manifest_path)
        by_seg = load_segment_split(manifest_path)
        use_manifest_segs = True
    else:
        if report_path is None:
            raise ValueError("需要 --manifest 或 --report")
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        review_map = {}
        for clip in report.get("clips") or []:
            rid = str(clip.get("record_id") or "")
            rk = str(clip.get("review_key") or "")
            if rid and rk:
                review_map[rid] = rk
        refs = [r for r in list_records(paths) if r.record_id in review_map]
        by_seg = {}
        use_manifest_segs = False

    for ref in refs:
        review_key = review_map.get(ref.record_id)
        if not review_key:
            continue
        review_path = paths.review_dir / review_key / "event_review.json"
        if not review_path.is_file():
            continue

        if use_manifest_segs:
            all_segs = by_seg.get(ref.record_id) or []
            train_segs = [s for s in all_segs if s.get("split") == "train"]
            val_segs = [s for s in all_segs if s.get("split") == "val"]
            n_segments_total += len(all_segs)
            n_train_segments += len(train_segs)
        else:
            raw_segs = build_ground_truth_segments(
                json.loads(review_path.read_text(encoding="utf-8")).get("verified_true") or []
            )
            train_segs = [
                {
                    "seg_id": f"{ref.record_id}#{i}",
                    "gt_tokens": list(toks),
                    "person_track_ids": [],
                    "frame_start": a,
                    "frame_end": b,
                    "split": "train",
                    "local_index": i,
                }
                for i, (toks, a, b) in enumerate(raw_segs)
            ]
            val_segs = []
            n_segments_total += len(train_segs)
            n_train_segments += len(train_segs)

        record = load_record(ref, paths)
        infer_h = int(record.meta.get("infer_height") or ref.infer_height or 1)
        bank = FeatureBank(
            infer_width=int(record.meta.get("infer_width") or ref.infer_width or 1),
            infer_height=infer_h,
            video_fps=float(record.fps or 15.0),
        )
        trigger = BoxTrigger(record.boxes, wrist_score_min=wrist_score_min)
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
        if sample_fps > 0:
            # 时序特征逐帧推进，训练与线上必须同帧率，否则停留/速度的分布对不上
            keys = sample_frame_indices(
                keys, source_fps=float(record.fps or 25.0), target_fps=sample_fps
            )

        before = len(rows)
        for export_key in keys:
            # val 段时间窗：整帧不进训练集
            in_val = any(
                int(s["frame_start"]) <= export_key <= int(s["frame_end"]) for s in val_segs
            )
            frame = by_key.get(export_key) or {
                "frame_idx": export_key,
                "source_frame_idx": export_key,
                "timestamp_sec": 0.0,
                "persons": [],
            }
            pending: list[tuple[dict[str, Any], int, dict[str, Any], dict[str, Any], str]] = []
            active_pairs: dict[str, dict[str, Any]] = {}
            for row in bank.rows_for_frame(frame):
                person = row.get("_person")
                if not isinstance(person, dict):
                    continue
                track = int(row.get("person_track_id") or person.get("person_track_id") or 0)
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
            if in_val:
                n_skipped_val_frames += 1
                continue

            active_train = [
                s
                for s in train_segs
                if int(s["frame_start"]) <= export_key <= int(s["frame_end"])
            ]

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

                matched = [
                    s
                    for s in active_train
                    if hit["token"] in (s.get("gt_tokens") or [])
                    and _track_match(s.get("person_track_ids") or [], track)
                ]
                if matched:
                    s0 = matched[0]
                    label, kind, seg_id = 1, KIND_POS, str(s0.get("seg_id") or "")
                elif active_train:
                    label, kind, seg_id = 0, KIND_NEG_NEIGHBOR, ""
                else:
                    label, kind, seg_id = 0, KIND_NEG_OUTSIDE, ""

                rows.append(vec)
                y_list.append(label)
                kind_list.append(kind)
                groups.append(ref.record_id)
                seg_ids.append(seg_id)
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

        print(f"[ok] {ref.record_id.split('/')[-1]} +pairs={len(rows) - before}")

    X = np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=np.float64)
    missing_rate = np.isnan(X).mean(axis=0) if len(X) else np.zeros(len(feature_keys))
    impute = np.zeros(len(feature_keys), dtype=np.float64)
    for j in range(X.shape[1] if len(X) else 0):
        col = X[:, j]
        fill = np.nanmedian(col) if not np.isnan(col).all() else np.nan
        impute[j] = 0.0 if np.isnan(fill) else fill
        if np.isnan(col).any():
            col[np.isnan(col)] = impute[j]

    y = np.asarray(y_list, dtype=np.int32)
    return {
        "X": X,
        "y": y,
        "kind": np.asarray(kind_list, dtype=np.int32),
        "groups": np.asarray(groups),
        "seg_ids": np.asarray(seg_ids),
        "impute": impute,
        "n_segments_total": n_segments_total,
        "n_train_segments": n_train_segments,
        "n_skipped_val_frames": n_skipped_val_frames,
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
    ap.add_argument("--report", default="", help="旧 accuracy_report 路径（与 --manifest 二选一）")
    ap.add_argument("--manifest", default="", help="tagged_aug85_v1.json（段级划分）")
    ap.add_argument("--out", default=str(ROOT / "output/train/pairs_tagged_v1.npz"))
    ap.add_argument("--depth-dir", default="")
    ap.add_argument("--sample-fps", type=float, default=0.0, help="模拟线上抽帧的处理帧率；0=逐帧")
    ap.add_argument("--wrist-score-min", type=float, default=0.3, help="手腕触发门槛")
    args = ap.parse_args()

    depth_dir = Path(args.depth_dir) if args.depth_dir.strip() else None
    if depth_dir is not None and not depth_dir.is_absolute():
        depth_dir = ROOT / depth_dir

    man = Path(args.manifest) if args.manifest.strip() else None
    if man is not None and not man.is_absolute():
        man = ROOT / man
    rep = Path(args.report) if args.report.strip() else None
    if rep is not None and not rep.is_absolute():
        rep = ROOT / rep

    ds = build(
        paths=load_paths(),
        report_path=rep,
        manifest_path=man,
        depth_dir=depth_dir,
        sample_fps=args.sample_fps,
        wrist_score_min=args.wrist_score_min,
    )
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
        impute=ds["impute"],
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
        "n_train_segments": ds["n_train_segments"],
        "n_skipped_val_frames": ds["n_skipped_val_frames"],
        "n_segments_covered": int(len({s for s in ds["seg_ids"].tolist() if s})),
        "feature_keys": ds["feature_keys"],
        "missing_rate": ds["missing_rate"],
        "manifest": str(man) if man else "",
        "sample_fps": args.sample_fps or None,
        "wrist_score_min": args.wrist_score_min,
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n配对数据集 → {out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
