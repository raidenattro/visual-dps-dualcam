#!/usr/bin/env python3
"""从 data.db 三标签生成段级 5:5 train/val manifest（只读 collector）。

标签：8.3新标注 / 8.4新标注 / 8.5新标注
划分单位：一次拣货（同一货框、标注帧间隔 <= gap 帧）
产物：本仓 output/manifests/tagged_aug85_v2.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_paths
from adapters.record_reader import RecordRef, load_boxes, record_dir
from adapters.review_labels import camera_review_aliases, normalize_box_token, normalize_verified_true

TAG_NAMES = ("8.3新标注", "8.4新标注", "8.5新标注")


def _data_db(paths) -> Path:
    return paths.localdata / "data.db"


def _tagged_record_ids(db: Path) -> list[tuple[str, str]]:
    """返回 [(record_id, tag_name), ...]；一条 record 若多标签取第一个命中。"""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = list(
        con.execute(
            """
            SELECT rt.record_id, t.name
            FROM record_tags rt
            JOIN tags t ON t.id = rt.tag_id
            WHERE t.name IN (?, ?, ?)
            ORDER BY t.name, rt.record_id
            """,
            TAG_NAMES,
        )
    )
    con.close()
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for rid, tag in rows:
        rid = str(rid)
        if rid in seen:
            continue
        seen.add(rid)
        out.append((rid, str(tag)))
    return out


def find_review_key(paths, record_id: str) -> tuple[str, dict[str, Any]] | None:
    parts = record_id.split("/")
    cam = parts[1] if len(parts) >= 2 else ""
    stem = parts[-1].replace("_rtmpose_m", "").replace("_rtmpose_t", "")
    for alias in camera_review_aliases(cam):
        base = paths.review_dir / alias
        if not base.is_dir():
            continue
        for p in base.iterdir():
            if not p.is_dir():
                continue
            if stem in p.name and (p / "event_review.json").is_file():
                data = json.loads((p / "event_review.json").read_text(encoding="utf-8"))
                return str(p.relative_to(paths.review_dir)), data
    # 兜底
    for p in paths.review_dir.rglob("event_review.json"):
        if stem in str(p):
            data = json.loads(p.read_text(encoding="utf-8"))
            return str(p.parent.relative_to(paths.review_dir)), data
    return None


def build_gt_segments(verified_true: list[Any], *, gap: int = 15) -> list[dict[str, Any]]:
    """一段 = 一次拣货：同一货框集合、相邻标注帧间隔不超过 gap 帧。

    跟踪 ID 不参与切段。上游跟踪会在同一个人身上来回换 ID（实测同一次拣货中
    #2/#4 逐帧抖动），计入判据会把一次拣货切成几十个单帧段。段内出现过的 ID
    全部记下，供训练标签匹配用。

    按货框分组后再切，避免多人同时拣不同货框时标注交错、互相打断。
    """
    by_tokens: dict[tuple[str, ...], list[tuple[int, set[str]]]] = {}
    for entry in normalize_verified_true(verified_true):
        toks = tuple(sorted(entry.get("confirmed_box_tokens") or []))
        tracks = {str(t) for t in (entry.get("person_track_ids") or []) if t}
        fi = int(entry.get("frame_idx") or entry.get("source_frame_idx") or 0)
        if fi <= 0 or not toks:
            continue
        by_tokens.setdefault(toks, []).append((fi, tracks))

    segs: list[dict[str, Any]] = []
    for toks, items in by_tokens.items():
        items.sort(key=lambda x: x[0])
        cur: dict[str, Any] | None = None
        for fi, tracks in items:
            if cur is not None and fi - cur["frame_end"] <= gap:
                cur["frame_end"] = fi
                cur["_tracks"] |= tracks
                cur["entry_count"] += 1
                continue
            if cur is not None:
                segs.append(cur)
            cur = {
                "gt_tokens": list(toks),
                "_tracks": set(tracks),
                "frame_start": fi,
                "frame_end": fi,
                "entry_count": 1,
            }
        if cur is not None:
            segs.append(cur)

    segs.sort(key=lambda s: (s["frame_start"], s["gt_tokens"]))
    for s in segs:
        s["person_track_ids"] = sorted(s.pop("_tracks"))
    return segs


def main() -> int:
    ap = argparse.ArgumentParser(description="三标签段级 5:5 manifest")
    ap.add_argument("--out", default=str(ROOT / "output/manifests/tagged_aug85_v2.json"))
    ap.add_argument("--name", default="tagged_aug85_v2")
    ap.add_argument(
        "--gap",
        type=int,
        default=15,
        help="同一货框的相邻标注帧间隔超过多少帧就算两次拣货（15fps 下 15 帧=1 秒）",
    )
    ap.add_argument("--val-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = load_paths()
    db = _data_db(paths)
    if not db.is_file():
        print(f"找不到 data.db: {db}", file=sys.stderr)
        return 1

    tagged = _tagged_record_ids(db)
    records: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for record_id, tag in tagged:
        found = find_review_key(paths, record_id)
        if not found:
            skipped.append({"record_id": record_id, "reason": "no_review"})
            continue
        review_key, review = found
        if str(review.get("status") or "").lower() != "completed":
            skipped.append({"record_id": record_id, "reason": f"status={review.get('status')}"})
            continue
        segs = build_gt_segments(review.get("verified_true") or [], gap=args.gap)
        if not segs:
            skipped.append({"record_id": record_id, "reason": "no_segments"})
            continue

        parts = record_id.split("/")
        camera = parts[1] if len(parts) >= 2 else ""
        clip_name = parts[-1]
        ref = RecordRef(
            record_id=record_id,
            clip_name=clip_name,
            camera_slug=camera,
            file_name=f"{clip_name}.json",
            infer_width=0,
            infer_height=0,
        )
        rd = record_dir(ref, paths)
        if not (rd / "skeleton.parquet").is_file():
            skipped.append({"record_id": record_id, "reason": "no_skeleton"})
            continue
        man = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
        boxes = load_boxes(rd)
        box_toks = {b.token for b in boxes}
        gt_toks = {normalize_box_token(t) for s in segs for t in s["gt_tokens"]}
        overlap = len(gt_toks & box_toks)
        if overlap <= 0:
            skipped.append({"record_id": record_id, "reason": "no_token_overlap"})
            continue

        rec_idx = len(records)
        records.append(
            {
                "record_id": record_id,
                "review_key": review_key,
                "camera_slug": camera,
                "clip_name": clip_name,
                "file": f"{clip_name}.json",
                "tag": tag,
                "status": "completed",
                "n_segments": len(segs),
                "n_verified_true": len(normalize_verified_true(review.get("verified_true") or [])),
                "token_overlap": overlap,
                "n_gt_tokens": len(gt_toks),
                "n_boxes": len(boxes),
                "infer_width": int(man.get("infer_width") or 0),
                "infer_height": int(man.get("infer_height") or 0),
                "fps": float(man.get("fps") or 0),
            }
        )
        for j, s in enumerate(segs):
            segments.append(
                {
                    "seg_id": f"{record_id}#{j}",
                    "record_id": record_id,
                    "record_index": rec_idx,
                    "local_index": j,
                    "gt_tokens": s["gt_tokens"],
                    "person_track_ids": s["person_track_ids"],
                    "frame_start": s["frame_start"],
                    "frame_end": s["frame_end"],
                    "entry_count": s["entry_count"],
                    "tag": tag,
                }
            )

    rng = np.random.default_rng(args.seed)
    n = len(segments)
    order = rng.permutation(n)
    n_val = int(round(n * args.val_ratio))
    val_set = set(int(i) for i in order[:n_val])
    for i, seg in enumerate(segments):
        seg["split"] = "val" if i in val_set else "train"

    n_train = sum(1 for s in segments if s["split"] == "train")
    n_val_seg = sum(1 for s in segments if s["split"] == "val")

    # 记录是否含 train/val 段
    for rec in records:
        rid = rec["record_id"]
        rec["n_train_segments"] = sum(1 for s in segments if s["record_id"] == rid and s["split"] == "train")
        rec["n_val_segments"] = sum(1 for s in segments if s["record_id"] == rid and s["split"] == "val")
        if rec["n_val_segments"] and rec["n_train_segments"]:
            rec["split_role"] = "both"
        elif rec["n_val_segments"]:
            rec["split_role"] = "val"
        else:
            rec["split_role"] = "train"

    out = {
        "name": args.name,
        "require_tags": list(TAG_NAMES),
        "split_unit": "segment",
        "merge_gap_frames": args.gap,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "n_records": len(records),
        "n_segments": n,
        "n_train_segments": n_train,
        "n_val_segments": n_val_seg,
        "skipped": skipped,
        "records": records,
        "segments": segments,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"写入 {out_path}\n"
        f"  records={len(records)}  segments={n}  train={n_train} val={n_val_seg}\n"
        f"  skipped={len(skipped)}"
    )
    for s in skipped:
        print(f"  [skip] {s}")
    return 0 if records and segments else 1


if __name__ == "__main__":
    raise SystemExit(main())
