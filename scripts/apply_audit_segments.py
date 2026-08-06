#!/usr/bin/env python3
"""把人工审核确认为「漏标」的误报片段补进 manifest，产出新版 manifest。

补标段一律进 train：
  - 训练侧修正假负样本（这些拣货动作原来被当成负样本喂给模型）
  - 评估侧因为 eval_record 用全部段判定告警是否被覆盖，这些片段自动不再算误报
  - 召回分母仍只含原标注的 val 段，避免用模型自己发现的片段给自己加分

人物 ID 从分数文件里恢复（同帧同货框分数最高的那个人）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MERGE_GAP = 15


def main() -> int:
    ap = argparse.ArgumentParser(description="用审核结果补标 manifest")
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v2.json"))
    ap.add_argument("--audit", default=str(ROOT / "output/audit/fp_audit.json"))
    ap.add_argument("--scores", default=str(ROOT / "output/scores/v5"))
    ap.add_argument("--out", default=str(ROOT / "output/manifests/tagged_aug85_v3.json"))
    ap.add_argument("--name", default="tagged_aug85_v3")
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    missing = [e for e in (audit.get("events") or []) if e.get("verdict") == "missing"]
    if not missing:
        print("审核结果里没有判定为漏标的片段")
        return 1
    print(f"审核确认漏标 {len(missing)} 个片段")

    # 帧 -> {token: (track, score)}，用来给补标段填人物 ID
    track_of: dict[tuple[str, int, str], str] = {}
    for p in sorted(Path(args.scores).glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rid = d["record_id"]
        for frame_idx, pairs in d["frames"]:
            best: dict[str, tuple[float, str]] = {}
            for x in pairs:
                tok, tr, sm = str(x[1]), str(x[0]), float(x[3])
                if tok not in best or sm > best[tok][0]:
                    best[tok] = (sm, tr)
            for tok, (_sm, tr) in best.items():
                track_of[(rid, int(frame_idx), tok)] = tr

    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for e in missing:
        by_key[(str(e["record_id"]), str(e["token"]))].append(e)

    extra: list[dict[str, Any]] = []
    for (rid, tok), items in sorted(by_key.items()):
        items.sort(key=lambda x: int(x["span"][0]))
        cur: dict[str, Any] | None = None
        for e in items:
            a, b = int(e["span"][0]), int(e["span"][1])
            tracks = {
                t for f in (a, int(e["frame"]), b)
                if (t := track_of.get((rid, f, tok)))
            }
            if cur is not None and a - int(cur["frame_end"]) <= MERGE_GAP:
                cur["frame_end"] = max(int(cur["frame_end"]), b)
                cur["_tracks"] |= tracks
                cur["entry_count"] += int(e["n"])
                continue
            if cur is not None:
                extra.append(cur)
            cur = {
                "record_id": rid,
                "gt_tokens": [tok],
                "_tracks": set(tracks),
                "frame_start": a,
                "frame_end": b,
                "entry_count": int(e["n"]),
            }
        if cur is not None:
            extra.append(cur)

    rec_index = {r["record_id"]: i for i, r in enumerate(man["records"])}
    local_next: dict[str, int] = defaultdict(int)
    for s in man["segments"]:
        local_next[s["record_id"]] = max(local_next[s["record_id"]], int(s["local_index"]) + 1)

    added = 0
    for s in extra:
        rid = s["record_id"]
        if rid not in rec_index:
            continue
        j = local_next[rid]
        local_next[rid] += 1
        man["segments"].append({
            "seg_id": f"{rid}#audit{j}",
            "record_id": rid,
            "record_index": rec_index[rid],
            "local_index": j,
            "gt_tokens": s["gt_tokens"],
            "person_track_ids": sorted(s.pop("_tracks")),
            "frame_start": int(s["frame_start"]),
            "frame_end": int(s["frame_end"]),
            "entry_count": int(s["entry_count"]),
            "tag": "audit",
            "split": "train",
            "source": "audit",
        })
        added += 1

    for rec in man["records"]:
        rid = rec["record_id"]
        rec["n_train_segments"] = sum(
            1 for s in man["segments"] if s["record_id"] == rid and s["split"] == "train"
        )
        rec["n_val_segments"] = sum(
            1 for s in man["segments"] if s["record_id"] == rid and s["split"] == "val"
        )
        rec["n_segments"] = rec["n_train_segments"] + rec["n_val_segments"]
        if rec["n_val_segments"] and rec["n_train_segments"]:
            rec["split_role"] = "both"
        elif rec["n_val_segments"]:
            rec["split_role"] = "val"
        else:
            rec["split_role"] = "train"

    man["name"] = args.name
    man["n_segments"] = len(man["segments"])
    man["n_train_segments"] = sum(1 for s in man["segments"] if s["split"] == "train")
    man["n_val_segments"] = sum(1 for s in man["segments"] if s["split"] == "val")
    man["audit_added_segments"] = added
    man["audit_source"] = str(Path(args.audit))

    out = Path(args.out)
    out.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"补标段 {added} 个（{len(missing)} 个片段按 {MERGE_GAP} 帧间隔合并）\n"
        f"写入 {out}\n"
        f"  train {man['n_train_segments']}（原 {man['n_train_segments'] - added}）"
        f"  val {man['n_val_segments']}（不变）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
