"""从 28-clip 构建帧级特征表：y=帧是否落在 verified_true 段内。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURE_KEYS = [
    "ankle_max_speed_norm",
    "arm_torso_angle_max",
    "elbow_angle_mean",
    "wrist_elevation_angle_max",
    "shoulder_hip_knee_angle_min",
]


def _load_review_key_map(report_path: Path) -> dict[str, str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for clip in report.get("clips") or []:
        rid = str(clip.get("record_id") or "")
        rk = str(clip.get("review_key") or "")
        if rid and rk:
            out[rid] = rk
    return out


def _positive_frames(review_path: Path) -> set[int]:
    data = json.loads(review_path.read_text(encoding="utf-8"))
    pos: set[int] = set()
    for entry in data.get("verified_true") or []:
        if not isinstance(entry, dict):
            continue
        fi = int(entry.get("source_frame_idx") or entry.get("frame_idx") or 0)
        if fi > 0:
            pos.add(fi)
    return pos


def _row_vector(row: dict[str, Any]) -> list[float] | None:
    vals: list[float] = []
    for key in FEATURE_KEYS:
        v = row.get(key)
        if v is None:
            return None
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            return None
    return vals


def _baseline_frame_indices(paths, file_name: str) -> set[int] | None:
    clip = paths.baseline_manifest.parent / file_name
    if not clip.is_file():
        return None
    rows = json.loads(clip.read_text(encoding="utf-8"))
    return {int(r.get("frame_idx") or 0) for r in rows if isinstance(r, dict)}


def _stable_bucket(record_id: str, frame_idx: int, mod: int) -> int:
    h = 0
    for ch in f"{record_id}:{frame_idx}":
        h = (h * 131 + ord(ch)) % 2_147_483_647
    return h % mod


def build_dataset(
    *,
    paths,
    report_path: Path,
) -> dict[str, Any]:
    """返回 X, y, groups, meta。只保留特征齐全的人-帧；多人同帧取腕抬升最大者。"""
    from adapters.record_reader import list_records, load_record
    from features.bank import FeatureBank
    from pipeline.box_trigger import BoxTrigger

    review_map = _load_review_key_map(report_path)
    refs = list_records(paths)

    X_list: list[list[float]] = []
    y_list: list[int] = []
    groups: list[str] = []
    frame_meta: list[dict[str, Any]] = []

    for ref in refs:
        review_key = review_map.get(ref.record_id)
        if not review_key:
            continue
        review_path = paths.review_dir / review_key / "event_review.json"
        if not review_path.is_file():
            continue
        pos = _positive_frames(review_path)
        frame_indices = _baseline_frame_indices(paths, ref.file_name)
        record = load_record(ref, paths)
        bank = FeatureBank(
            infer_width=int(record.meta.get("infer_width") or 1),
            infer_height=int(record.meta.get("infer_height") or 1),
            video_fps=float(record.fps or 15.0),
        )
        trigger = BoxTrigger(record.boxes)

        by_key = {
            int(fr.get("source_frame_idx") or fr.get("frame_idx") or 0): fr
            for fr in record.frames
        }
        keys = sorted(frame_indices) if frame_indices is not None else sorted(by_key)

        for export_key in keys:
            frame = by_key.get(export_key) or {
                "frame_idx": export_key,
                "source_frame_idx": export_key,
                "timestamp_sec": 0.0,
                "persons": [],
            }
            rows = bank.rows_for_frame(frame)
            if not rows:
                continue

            # 选一人：优先有货框命中，其次腕抬升最大
            best = None
            best_key = (-1.0, -1.0)
            for row in rows:
                person = row.get("_person") or {}
                hits = trigger.hits_for_person(person) if person else []
                elev = float(row.get("wrist_elevation_angle_max") or -1.0)
                key = (1.0 if hits else 0.0, elev)
                if key > best_key:
                    best_key = key
                    best = row
            if best is None:
                continue
            vec = _row_vector(best)
            if vec is None:
                continue

            label = 1 if export_key in pos else 0
            has_hit = best_key[0] > 0.5
            # 负样本：保留有命中的硬负 + ~8% 无命中，避免全负淹没
            if label == 0 and not has_hit and _stable_bucket(ref.record_id, export_key, 12) != 0:
                continue

            X_list.append(vec)
            y_list.append(label)
            groups.append(ref.record_id)
            frame_meta.append(
                {
                    "record_id": ref.record_id,
                    "frame_idx": export_key,
                    "y": label,
                    "has_box_hit": has_hit,
                }
            )

    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.int32)
    return {
        "X": X,
        "y": y,
        "groups": np.asarray(groups),
        "feature_keys": list(FEATURE_KEYS),
        "meta": frame_meta,
        "n_pos": int(y.sum()) if len(y) else 0,
        "n_neg": int(len(y) - y.sum()) if len(y) else 0,
        "n_records": len(set(groups)),
    }


def main() -> int:
    import argparse

    from adapters.collector_paths import load_paths

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report",
        default=str(ROOT / "output/export/pickstate-rule-v0-prod-test/accuracy_report.json"),
    )
    ap.add_argument("--out", default=str(ROOT / "output/train/dataset_v1.npz"))
    args = ap.parse_args()

    paths = load_paths()
    ds = build_dataset(paths=paths, report_path=Path(args.report))
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=ds["X"],
        y=ds["y"],
        groups=ds["groups"],
        feature_keys=np.asarray(ds["feature_keys"]),
    )
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "n": int(len(ds["y"])),
                "n_pos": ds["n_pos"],
                "n_neg": ds["n_neg"],
                "n_records": ds["n_records"],
                "feature_keys": ds["feature_keys"],
                "pos_rate": (ds["n_pos"] / len(ds["y"])) if len(ds["y"]) else 0.0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"dataset → {out}  n={len(ds['y'])} pos={ds['n_pos']} neg={ds['n_neg']} "
        f"records={ds['n_records']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
