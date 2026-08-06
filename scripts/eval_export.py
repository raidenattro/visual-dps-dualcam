#!/usr/bin/env python3
"""评估导出包：只关心段/事件的漏报与误报（不报帧级结论）。

- 漏报：真值段内同框无任何告警
- 误报：未落入真值段的告警，合并为事件；再拆「算法该负责」与「标注边界」
- 主口径 segment_loose（同框连续条目合并，对齐历史 98%/156）
- 对照 segment_gap2；segment_adj1 仅对照 collector 过严规则

只读 collector review；产物写 --out（默认写入导出目录）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_paths
from adapters.review_labels import normalize_box_token, normalize_verified_true


def _norm_token(raw: str) -> str:
    return normalize_box_token(raw)


def _token_match(a: str, b: str) -> bool:
    return _norm_token(a) == _norm_token(b)


def _rid_aliases(record_id: str) -> list[str]:
    """rtmpose-m / rtmpose-t 共用同一路 review，record_id 常只写其中一种。"""
    rid = str(record_id or "").strip()
    out = [rid]
    if "/rtmpose-m/" in f"/{rid}/" or rid.startswith("rtmpose-m/"):
        out.append(rid.replace("rtmpose-m", "rtmpose-t", 1))
    if "/rtmpose-t/" in f"/{rid}/" or rid.startswith("rtmpose-t/"):
        out.append(rid.replace("rtmpose-t", "rtmpose-m", 1))
    # 去重保序
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _load_review(paths, record_id: str) -> tuple[dict[str, Any] | None, str]:
    aliases = set(_rid_aliases(record_id))
    stem = str(record_id or "").strip().split("/")[-1]
    stem_base = stem.replace("_rtmpose_m", "").replace("_rtmpose_t", "")
    fallback: tuple[dict[str, Any], str] | None = None
    for p in paths.review_dir.rglob("event_review.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        er_rid = str(data.get("record_id") or "").strip()
        rk = str(data.get("review_key") or p.parent.relative_to(paths.review_dir))
        if er_rid in aliases:
            return data, rk
        # 次选：同相机同 clip 时间戳前缀（pose 档不同）
        if stem_base and stem_base in rk and fallback is None:
            fallback = (data, rk)
    if fallback is not None:
        return fallback
    return None, ""


def _verified_entries(verified_true: list[dict[str, Any]]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for entry in normalize_verified_true(verified_true):
        fi = int(entry.get("source_frame_idx") or entry.get("frame_idx") or 0)
        if fi <= 0:
            continue
        for tok in entry.get("confirmed_box_tokens") or []:
            if tok:
                out.append((fi, str(tok)))
    return out


def _extract_alarms(frames: list[dict[str, Any]]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for fr in frames:
        if not fr.get("is_picking"):
            continue
        fi = int(fr.get("frame_idx") or 0)
        tokens = list(fr.get("rule_alarm_collisions") or []) or list(fr.get("rule_collisions") or [])
        if tokens:
            for raw in tokens:
                tok = _norm_token(str(raw))
                if tok:
                    out.append((fi, tok))
        else:
            out.append((fi, ""))
    return out


def _alarms_at(alarms: list[tuple[int, str]], frame: int) -> list[str]:
    return [t for f, t in alarms if f == frame and t]


def eval_frame_level(
    verified_true: list[dict[str, Any]],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """帧级严格：标真条目是否在同帧同框有告警；告警帧不在标真 (frame,box) 上即 FP。"""
    entries = _verified_entries(verified_true)
    alarms = _extract_alarms(frames)
    pos_set = set(entries)

    tp = fn = 0
    fn_samples: list[dict[str, Any]] = []
    for fi, tok in entries:
        hit = any(_token_match(t, tok) for t in _alarms_at(alarms, fi))
        if hit:
            tp += 1
        else:
            fn += 1
            if len(fn_samples) < 30:
                fn_samples.append({"frame_idx": fi, "box_token": tok})

    fp = 0
    fp_samples: list[dict[str, Any]] = []
    for fi, tok in alarms:
        if not tok:
            fp += 1
            if len(fp_samples) < 30:
                fp_samples.append({"frame_idx": fi, "box_token": ""})
            continue
        if (fi, tok) not in pos_set and not any(
            _token_match(t, tok) for f, t in pos_set if f == fi
        ):
            fp += 1
            if len(fp_samples) < 30:
                fp_samples.append({"frame_idx": fi, "box_token": tok})

    total = tp + fn
    recall = round(tp / total, 4) if total else None
    prec = round(tp / (tp + fp), 4) if (tp + fp) else None
    return {
        "verified_entries": total,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "recall": recall,
        "miss_rate": round(fn / total, 4) if total else None,
        "precision": prec,
        "alarm_frames": len({f for f, _ in alarms}),
        "fn_samples": fn_samples,
        "fp_samples": fp_samples,
    }


@dataclass
class GtSegment:
    gt_tokens: tuple[str, ...]
    frame_start: int
    frame_end: int
    entry_count: int = 0


# 与 pose_frame_interval=2 对齐：隔帧采样下相邻导出帧间隔为 2，合段容差至少要到 2。
# collector 7/30 改成「必须 +1」会把合法段拆碎，本仓主口径不用那套。
DEFAULT_SEG_GAP = 2


def _build_segments(
    verified_true: list[dict[str, Any]],
    *,
    max_gap: int | None,
) -> list[GtSegment]:
    """段级 GT：同框且帧间隔 ≤ max_gap 则合并；max_gap=None 表示同框条目一律并（旧宽松口径）。"""
    entries = sorted(
        normalize_verified_true(verified_true),
        key=lambda e: int(e.get("frame_idx") or e.get("source_frame_idx") or 0),
    )
    segs: list[GtSegment] = []
    cur: GtSegment | None = None
    prev_frame: int | None = None

    for entry in entries:
        toks = tuple(sorted(str(t) for t in (entry.get("confirmed_box_tokens") or []) if t))
        if not toks:
            continue
        frame = int(entry.get("frame_idx") or entry.get("source_frame_idx") or 0)
        if frame <= 0:
            continue
        if cur is not None and cur.gt_tokens == toks:
            gap_ok = max_gap is None or (prev_frame is not None and frame - prev_frame <= max_gap)
            if gap_ok:
                cur.frame_end = max(cur.frame_end, frame)
                cur.entry_count += 1
                prev_frame = frame
                continue
        if cur is not None:
            segs.append(cur)
        cur = GtSegment(gt_tokens=toks, frame_start=frame, frame_end=frame, entry_count=1)
        prev_frame = frame
    if cur is not None:
        segs.append(cur)
    return segs


# 误报事件合并：同框告警帧间隔 ≤ 此值算同一事件（源帧，含隔帧采样）
EVENT_MERGE_GAP = 6
# 紧贴同框真值段边界的误报事件，视为标注边界误差（不算算法）
EVENT_BOUNDARY_TOL = 4


def _interval_gap(a: tuple[int, int], b: tuple[int, int]) -> int:
    if a[1] < b[0]:
        return b[0] - a[1]
    if b[1] < a[0]:
        return a[0] - b[1]
    return 0


def _fp_events(
    uncovered: list[tuple[int, str]],
    segments: list[GtSegment],
    *,
    merge_gap: int = EVENT_MERGE_GAP,
    boundary_tol: int = EVENT_BOUNDARY_TOL,
) -> dict[str, Any]:
    """未覆盖告警帧 → 事件；再拆「算法该负责」与「标注边界」。"""
    by_tok: dict[str, list[int]] = {}
    for frame, tok in uncovered:
        by_tok.setdefault(tok, []).append(frame)

    events_all = 0
    events_algo = 0
    events_boundary = 0
    samples: list[dict[str, Any]] = []

    for tok, frames_list in by_tok.items():
        frames_list = sorted(set(frames_list))
        if not frames_list:
            continue
        runs: list[list[int]] = [[frames_list[0]]]
        for f in frames_list[1:]:
            if f - runs[-1][-1] <= merge_gap:
                runs[-1].append(f)
            else:
                runs.append([f])
        own = [(s.frame_start, s.frame_end) for s in segments if any(_token_match(tok, g) for g in s.gt_tokens)]
        for run in runs:
            span = (run[0], run[-1])
            events_all += 1
            near = min((_interval_gap(span, s) for s in own), default=10**9)
            if near <= boundary_tol:
                events_boundary += 1
                kind = "boundary"
            else:
                events_algo += 1
                kind = "algorithm"
            if len(samples) < 30:
                samples.append(
                    {
                        "box_token": tok,
                        "frame_start": span[0],
                        "frame_end": span[1],
                        "n_frames": len(run),
                        "kind": kind,
                        "gap_to_gt": near if near < 10**9 else None,
                    }
                )

    return {
        "fp_events": events_all,
        "fp_events_algorithm": events_algo,
        "fp_events_boundary": events_boundary,
        "fp_event_samples": samples,
    }


def eval_segment_level(
    verified_true: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    *,
    max_gap: int | None,
) -> dict[str, Any]:
    """只评段/事件：漏报=未检出的真值段；误报=未落入真值段的告警事件。"""
    segments = _build_segments(verified_true, max_gap=max_gap)
    alarms = _extract_alarms(frames)

    detected = missed = 0
    missed_samples: list[dict[str, Any]] = []
    for seg in segments:
        gt = list(seg.gt_tokens)
        ok = any(
            seg.frame_start <= frame <= seg.frame_end and any(_token_match(tok, g) for g in gt)
            for frame, tok in alarms
        )
        if ok:
            detected += 1
        else:
            missed += 1
            if len(missed_samples) < 20:
                missed_samples.append(
                    {
                        "frame_start": seg.frame_start,
                        "frame_end": seg.frame_end,
                        "gt_tokens": list(seg.gt_tokens),
                        "entry_count": seg.entry_count,
                    }
                )

    uncovered: list[tuple[int, str]] = []
    for frame, tok in alarms:
        covered = any(
            seg.frame_start <= frame <= seg.frame_end and any(_token_match(tok, g) for g in seg.gt_tokens)
            for seg in segments
        )
        if not covered:
            uncovered.append((frame, tok))

    fp_ev = _fp_events(uncovered, segments)
    total = len(segments)
    recall = round(detected / total, 4) if total else None
    algo_fp = fp_ev["fp_events_algorithm"]
    return {
        "gt_segments": total,
        "detected": detected,
        "missed": missed,
        "recall": recall,
        "miss_rate": round(missed / total, 4) if total else None,
        "fp_events": fp_ev["fp_events"],
        "fp_events_algorithm": algo_fp,
        "fp_events_boundary": fp_ev["fp_events_boundary"],
        "precision_proxy": (
            round(detected / (detected + algo_fp), 4) if (detected + algo_fp) else None
        ),
        "missed_segments": missed_samples,
        "fp_event_samples": fp_ev["fp_event_samples"],
        "max_gap": max_gap,
        # 旧字段：不再用帧级 FP 作结论
        "fp": algo_fp,
    }


def eval_clip(paths, export_path: Path) -> dict[str, Any]:
    frames = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(frames, list) or not frames:
        return {"status": "error", "upload_file": export_path.name, "error": "空 JSON"}
    record_id = str(frames[0].get("record_id") or "").strip()
    if not record_id:
        return {"status": "error", "upload_file": export_path.name, "error": "缺少 record_id"}

    review, review_key = _load_review(paths, record_id)
    if not review:
        return {"status": "error", "upload_file": export_path.name, "record_id": record_id, "error": "未找到 review"}
    status = str(review.get("status") or "").strip().lower()
    verified = review.get("verified_true") if isinstance(review.get("verified_true"), list) else []
    if not verified:
        return {"status": "skipped", "upload_file": export_path.name, "record_id": record_id, "error": "无 verified_true"}

    frame_metrics = eval_frame_level(verified, frames)
    # 主口径：gap=2（隔帧采样）；collector_adj1 仅对照（7/30 后的过严规则）
    seg_gap2 = eval_segment_level(verified, frames, max_gap=DEFAULT_SEG_GAP)
    seg_adj1 = eval_segment_level(verified, frames, max_gap=1)
    seg_loose = eval_segment_level(verified, frames, max_gap=None)

    return {
        "status": "ok" if status == "completed" else "excluded",
        "upload_file": export_path.name,
        "record_id": record_id,
        "review_key": review_key,
        "review_status": status or "unknown",
        "verified_entry_count": len(verified),
        "frame_level": frame_metrics,
        "segment_gap2": seg_gap2,
        "segment_adj1": seg_adj1,
        "segment_loose": seg_loose,
        # 兼容旧字段名
        "segment_legacy": seg_loose,
        "segment_gap45": eval_segment_level(verified, frames, max_gap=45),
    }


def _render_md(result: dict[str, Any], pkg: Path) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# 导出包评估（段/事件口径）",
        "",
        f"> 生成时间：{now}  ",
        f"> 目录：`{pkg}`  ",
        "> **只评段/事件**：漏报=未检出真值段；误报=未覆盖告警合并为事件（算法 vs 边界）  ",
        "> **主口径** `segment_loose`；对照 `segment_gap2`  ",
        "",
    ]
    s = result.get("summary") or {}
    lo = s.get("segment_loose") or {}
    g2 = s.get("segment_gap2") or {}
    lines.extend(
        [
            "## 汇总（主·历史对齐）",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 成功评估 | {s.get('evaluated', 0)} |",
            f"| GT 段数 | {lo.get('gt_segments', 0)} |",
            f"| 检出段 | {lo.get('detected', 0)} |",
            f"| 漏段 | {lo.get('missed', 0)} |",
            f"| 段召回 | {_pct(lo.get('recall'))} |",
            f"| 误报事件（算法） | {lo.get('fp_events_algorithm', 0)} |",
            f"| 误报事件（边界） | {lo.get('fp_events_boundary', 0)} |",
            f"| 误报事件合计 | {lo.get('fp_events', 0)} |",
            "",
            "## 对照（gap=2）",
            "",
            f"召回 {_pct(g2.get('recall'))}  漏段 {g2.get('missed')}/{g2.get('gt_segments')}  "
            f"误报事件 {g2.get('fp_events')}（算法 {g2.get('fp_events_algorithm')} / 边界 {g2.get('fp_events_boundary')}）",
            "",
        ]
    )
    lines.extend(
        [
            "## 分片",
            "",
            "| 文件 | 漏段 | GT段 | 召回(loose) | 误报事件(算法/边界) |",
            "|------|------|------|-------------|---------------------|",
        ]
    )
    for c in result.get("clips") or []:
        if c.get("status") != "ok":
            lines.append(f"| {c.get('upload_file')} | — | — | {c.get('status')} | — |")
            continue
        sl = c.get("segment_loose") or {}
        lines.append(
            f"| {c.get('upload_file')} | {sl.get('missed')} | {sl.get('gt_segments')} | "
            f"{_pct(sl.get('recall'))} | {sl.get('fp_events_algorithm')}/{sl.get('fp_events_boundary')} |"
        )
    return "\n".join(lines) + "\n"


def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.2%}"


def main() -> int:
    ap = argparse.ArgumentParser(description="评估导出包（段/事件漏报与误报）")
    ap.add_argument("--pkg", required=True, help="导出目录")
    ap.add_argument("--out", default="", help="报告 JSON 路径（默认 <pkg>/eval_event_report.json）")
    args = ap.parse_args()

    paths = load_paths()
    pkg = Path(args.pkg)
    if not pkg.is_absolute():
        pkg = ROOT / pkg
    if not pkg.is_dir():
        print(f"目录不存在: {pkg}", file=sys.stderr)
        return 1

    clip_files = sorted(
        p for p in pkg.glob("*.json") if p.name != "_manifest.json" and not p.name.startswith("eval_") and not p.name.startswith("accuracy_")
    )
    clips = [eval_clip(paths, p) for p in clip_files]
    ok = [c for c in clips if c.get("status") == "ok"]

    def _sum_seg(key: str) -> dict[str, Any]:
        gt = sum(int((c.get(key) or {}).get("gt_segments") or 0) for c in ok)
        det = sum(int((c.get(key) or {}).get("detected") or 0) for c in ok)
        miss = sum(int((c.get(key) or {}).get("missed") or 0) for c in ok)
        fp_ev = sum(int((c.get(key) or {}).get("fp_events") or 0) for c in ok)
        fp_algo = sum(int((c.get(key) or {}).get("fp_events_algorithm") or 0) for c in ok)
        fp_bound = sum(int((c.get(key) or {}).get("fp_events_boundary") or 0) for c in ok)
        return {
            "gt_segments": gt,
            "detected": det,
            "missed": miss,
            "fp_events": fp_ev,
            "fp_events_algorithm": fp_algo,
            "fp_events_boundary": fp_bound,
            "recall": round(det / gt, 4) if gt else None,
            "precision_proxy": round(det / (det + fp_algo), 4) if (det + fp_algo) else None,
        }

    summary = {
        "evaluated": len(ok),
        "clip_count": len(clips),
        "metric": "segment_event_only",
        "segment_loose": _sum_seg("segment_loose"),
        "segment_gap2": _sum_seg("segment_gap2"),
        "segment_adj1": _sum_seg("segment_adj1"),
    }

    result = {"summary": summary, "clips": clips, "primary": "segment_loose"}
    out_json = Path(args.out) if args.out else pkg / "eval_event_report.json"
    if not out_json.is_absolute():
        out_json = ROOT / out_json
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (pkg / "eval_event_report.md").write_text(_render_md(result, pkg), encoding="utf-8")

    lo = summary["segment_loose"]
    g2 = summary["segment_gap2"]
    print(
        f"【主·历史对齐】召回 {_pct(lo.get('recall'))}  "
        f"漏段 {lo['missed']}/{lo['gt_segments']}  "
        f"误报事件 {lo['fp_events']}（算法 {lo['fp_events_algorithm']} / 边界 {lo['fp_events_boundary']}）"
    )
    print(
        f"【对照·gap=2】  召回 {_pct(g2.get('recall'))}  "
        f"漏段 {g2['missed']}/{g2['gt_segments']}  "
        f"误报事件 {g2['fp_events']}（算法 {g2['fp_events_algorithm']} / 边界 {g2['fp_events_boundary']}）"
    )
    print(f"报告: {out_json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
