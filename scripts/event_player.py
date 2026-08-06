#!/usr/bin/env python3
"""网页事件播放器：浏览器播视频，每个事件（检对/误报/漏报）自动暂停，空格继续。

远程 Cursor 可用：本机起 HTTP 服务，端口转发后打开浏览器即可（不依赖 OpenCV GUI）。

只读 collector 视频 / review；缓存叠加图写本仓 output/viz/。

示例：
  .venv/bin/python scripts/event_player.py \\
    --pkg output/export/exp_v1_t027_val \\
    --port 8765
  # 默认加载 --pkg 下全部 clip；可用 --clips a,b 限定子集
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_paths
from adapters.record_reader import RecordRef, load_boxes, load_skeleton_frames, record_dir
from scripts.eval_export import (
    EVENT_BOUNDARY_TOL,
    EVENT_MERGE_GAP,
    _build_segments,
    _extract_alarms,
    _load_review,
    _token_match,
)
from scripts.render_fn_fp_frames import (
    COLOR_FN,
    COLOR_FP,
    COLOR_GT,
    _draw_boxes,
    _draw_skeleton,
    _persons_at,
    _resolve_video,
)

KIND_LABEL = {
    "tp": "检对",
    "fp": "误报",
    "fn": "漏报",
    "fp_boundary": "误报·边界",
}


def _frame_to_sec(frame_idx: int, fps: float, *, start_pts: float = 0.0) -> float:
    """frame_idx(1-based) → HTML5 currentTime。

    部分 collector 切片 mp4 的媒体时间轴带 start_time（如 1.623s），
    OpenCV 按帧号读不受影响，但 <video>.currentTime 走的是带偏移的时间轴；
    不加 start_pts 会左侧视频比右侧叠加帧慢一截。
    """
    return max(0.0, float(start_pts) + (int(frame_idx) - 1) / max(fps, 1e-6))


def _collect_clip_events(
    export_path: Path,
    paths,
    *,
    merge_gap: int = EVENT_MERGE_GAP,
    boundary_tol: int = EVENT_BOUNDARY_TOL,
) -> dict[str, Any] | None:
    frames = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(frames, list) or not frames:
        return None
    record_id = str(frames[0].get("record_id") or "").strip()
    if not record_id:
        return None

    review, review_key = _load_review(paths, record_id)
    if not review:
        return None
    verified = review.get("verified_true") if isinstance(review.get("verified_true"), list) else []
    if not verified:
        return None

    ref = RecordRef(
        record_id=record_id,
        clip_name="",
        camera_slug="",
        file_name="",
        infer_width=0,
        infer_height=0,
    )
    rd = record_dir(ref, paths)
    man = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    video = _resolve_video(record_id, man, paths)
    if video is None or not video.is_file():
        return None

    fps = float(man.get("fps") or man.get("video_fps") or 25.0)
    start_pts = float(man.get("video_start_pts_sec") or 0.0)
    iw = int(man.get("infer_width") or 0)
    ih = int(man.get("infer_height") or 0)
    segments = _build_segments(verified, max_gap=None)
    alarms = _extract_alarms(frames)

    probs: dict[tuple[int, str], float] = {}
    for row in frames:
        p = row.get("picking_prob")
        if p is None:
            continue
        fi = int(row.get("frame_idx") or 0)
        for tok in row.get("rule_alarm_collisions") or []:
            probs[(fi, str(tok))] = float(p)

    events: list[dict[str, Any]] = []

    for seg in segments:
        hit_frames = [
            f
            for f, tok in alarms
            if seg.frame_start <= f <= seg.frame_end
            and any(_token_match(tok, g) for g in seg.gt_tokens)
        ]
        if hit_frames:
            peak = max(hit_frames, key=lambda f: max(
                (probs.get((f, g), 0.0) for g in seg.gt_tokens), default=0.0
            ))
            kind = "tp"
        else:
            peak = (seg.frame_start + seg.frame_end) // 2
            kind = "fn"
        events.append(
            {
                "kind": kind,
                "label": KIND_LABEL[kind],
                "frame_start": seg.frame_start,
                "frame_end": seg.frame_end,
                "peak_frame": peak,
                "tokens": list(seg.gt_tokens),
                "prob": max((probs.get((peak, g), 0.0) for g in seg.gt_tokens), default=None),
                "t_start": round(_frame_to_sec(seg.frame_start, fps, start_pts=start_pts), 3),
                "t_end": round(_frame_to_sec(seg.frame_end, fps, start_pts=start_pts), 3),
                "t_peak": round(_frame_to_sec(peak, fps, start_pts=start_pts), 3),
            }
        )

    uncovered: list[tuple[int, str]] = []
    for frame, tok in alarms:
        covered = any(
            seg.frame_start <= frame <= seg.frame_end
            and any(_token_match(tok, g) for g in seg.gt_tokens)
            for seg in segments
        )
        if not covered:
            uncovered.append((frame, tok))

    # 按 run 建完整误报事件列表
    by_tok: dict[str, list[int]] = {}
    for frame, tok in uncovered:
        by_tok.setdefault(tok, []).append(frame)
    for tok, flist in by_tok.items():
        flist = sorted(set(flist))
        if not flist:
            continue
        runs: list[list[int]] = [[flist[0]]]
        for f in flist[1:]:
            if f - runs[-1][-1] <= merge_gap:
                runs[-1].append(f)
            else:
                runs.append([f])
        own = [
            (s.frame_start, s.frame_end)
            for s in segments
            if any(_token_match(tok, g) for g in s.gt_tokens)
        ]
        for run in runs:
            span = (run[0], run[-1])
            near = min((_interval_gap(span, s) for s in own), default=10**9)
            is_bound = near <= boundary_tol
            kind = "fp_boundary" if is_bound else "fp"
            peak = max(run, key=lambda f: probs.get((f, tok), 0.0))
            events.append(
                {
                    "kind": kind,
                    "label": KIND_LABEL[kind],
                    "frame_start": span[0],
                    "frame_end": span[1],
                    "peak_frame": peak,
                    "tokens": [tok],
                    "prob": probs.get((peak, tok)),
                    "gap_to_gt": None if near >= 10**9 else near,
                    "t_start": round(_frame_to_sec(span[0], fps, start_pts=start_pts), 3),
                    "t_end": round(_frame_to_sec(span[1], fps, start_pts=start_pts), 3),
                    "t_peak": round(_frame_to_sec(peak, fps, start_pts=start_pts), 3),
                }
            )

    events.sort(key=lambda e: (e["frame_start"], e["kind"], e["tokens"]))
    for i, e in enumerate(events):
        e["index"] = i

    camera = record_id.split("/")[1] if "/" in record_id else ""
    return {
        "id": export_path.stem,
        "upload_file": export_path.name,
        "record_id": record_id,
        "review_key": review_key,
        "camera": camera,
        "fps": fps,
        "video_start_pts_sec": start_pts,
        "infer_width": iw,
        "infer_height": ih,
        "video_path": str(video),
        "record_dir": str(rd),
        "n_events": len(events),
        "counts": {
            "tp": sum(1 for e in events if e["kind"] == "tp"),
            "fn": sum(1 for e in events if e["kind"] == "fn"),
            "fp": sum(1 for e in events if e["kind"] == "fp"),
            "fp_boundary": sum(1 for e in events if e["kind"] == "fp_boundary"),
        },
        "events": events,
    }


def _collect_gt_clips(
    manifest_path: Path, paths, *, gap: int, split: str
) -> list[dict[str, Any]]:
    """拣货事件模式：直接读标注，不依赖导出包。

    每条切片给出两级粒度：
      segments —— manifest 现在的段（同货框同人的标注连续就合并，不看空洞）
      bursts   —— 段内按标注帧间隔 > gap 再切开，一个 burst 应当对应一次拣货
    两者并列展示，用于人工确认「一个事件」该怎么算。
    """
    from adapters.record_reader import list_records_from_manifest
    from adapters.review_labels import normalize_verified_true
    from scripts.build_tagged_manifest import find_review_key

    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_rid: dict[str, list[dict[str, Any]]] = {}
    for s in man.get("segments") or []:
        if split and s.get("split") != split:
            continue
        by_rid.setdefault(str(s["record_id"]), []).append(s)

    refs = {r.record_id: r for r in list_records_from_manifest(manifest_path, split_role=None)}
    clips: list[dict[str, Any]] = []

    for rid, segs in sorted(by_rid.items()):
        ref = refs.get(rid)
        if ref is None:
            continue
        rd = record_dir(ref, paths)
        man_json = rd / "manifest.json"
        if not man_json.is_file():
            continue
        rec_man = json.loads(man_json.read_text(encoding="utf-8"))
        video = _resolve_video(rid, rec_man, paths)
        if video is None or not video.is_file():
            continue

        found = find_review_key(paths, rid)
        if not found:
            continue
        labeled_by_token: dict[str, set[int]] = {}
        for e in normalize_verified_true(found[1].get("verified_true") or []):
            fi = int(e.get("frame_idx") or e.get("source_frame_idx") or 0)
            if fi <= 0:
                continue
            for tok in e.get("confirmed_box_tokens") or []:
                labeled_by_token.setdefault(str(tok), set()).add(fi)

        fps = float(rec_man.get("fps") or rec_man.get("video_fps") or 25.0)
        start_pts = float(rec_man.get("video_start_pts_sec") or 0.0)
        sec = lambda f: round(_frame_to_sec(f, fps, start_pts=start_pts), 3)  # noqa: E731

        out_segs: list[dict[str, Any]] = []
        for s in sorted(segs, key=lambda x: int(x["frame_start"])):
            a, b = int(s["frame_start"]), int(s["frame_end"])
            toks = [str(t) for t in (s.get("gt_tokens") or [])]
            marked = sorted(
                f
                for tok in toks
                for f in labeled_by_token.get(tok, ())
                if a <= f <= b
            )
            marked = sorted(set(marked))
            bursts: list[dict[str, Any]] = []
            if marked:
                run = [marked[0]]
                for f in marked[1:]:
                    if f - run[-1] > gap:
                        bursts.append(run)
                        run = []
                    run.append(f)
                bursts.append(run)
            span = b - a + 1
            out_segs.append(
                {
                    "seg_id": str(s.get("seg_id") or ""),
                    "frame_start": a,
                    "frame_end": b,
                    "tokens": toks,
                    "tracks": [str(t) for t in (s.get("person_track_ids") or [])],
                    "split": str(s.get("split") or ""),
                    "span": span,
                    "n_labeled": len(marked),
                    "fill": round(len(marked) / span, 3) if span else 0.0,
                    "labeled": marked,
                    "t_start": sec(a),
                    "t_end": sec(b),
                    "bursts": [
                        {
                            "frame_start": r[0],
                            "frame_end": r[-1],
                            "n_labeled": len(r),
                            "t_start": sec(r[0]),
                            "t_end": sec(r[-1]),
                        }
                        for r in bursts
                    ],
                }
            )

        n_bursts = sum(len(s["bursts"]) for s in out_segs)
        clips.append(
            {
                "mode": "gt",
                "id": rid.split("/")[-1],
                "record_id": rid,
                "camera": rid.split("/")[1] if "/" in rid else "",
                "fps": fps,
                "video_start_pts_sec": start_pts,
                "infer_width": int(rec_man.get("infer_width") or 0),
                "infer_height": int(rec_man.get("infer_height") or 0),
                "video_path": str(video),
                "record_dir": str(rd),
                "segments": out_segs,
                "n_segments": len(out_segs),
                "n_bursts": n_bursts,
            }
        )
    return clips


def _interval_gap(a: tuple[int, int], b: tuple[int, int]) -> int:
    if a[1] < b[0]:
        return b[0] - a[1]
    if b[1] < a[0]:
        return a[0] - b[1]
    return 0


def _render_overlay(clip: dict[str, Any], event: dict[str, Any], cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{clip['id']}_{event['index']:04d}_{event['kind']}.jpg"
    if out.is_file():
        return out

    frame_idx = int(event["peak_frame"])
    highlight = set(event.get("tokens") or [])
    kind = event["kind"]
    color = COLOR_GT if kind == "tp" else (COLOR_FN if kind == "fn" else COLOR_FP)

    rd = Path(clip["record_dir"])
    boxes = load_boxes(rd)
    skel = load_skeleton_frames(rd)
    persons = _persons_at(skel, frame_idx)

    cap = cv2.VideoCapture(clip["video_path"])
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
        ok, img = cap.read()
        if not ok or img is None:
            # 占位
            img = __import__("numpy").zeros((480, 852, 3), dtype="uint8")
    finally:
        cap.release()

    iw = int(clip.get("infer_width") or 0) or img.shape[1]
    ih = int(clip.get("infer_height") or 0) or img.shape[0]
    if img.shape[1] != iw or img.shape[0] != ih:
        img = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA)

    _draw_boxes(img, boxes, highlight=highlight, highlight_color=color)
    _draw_skeleton(img, persons)
    lines = [
        f"{event['label']}  #{event['index'] + 1}/{clip['n_events']}",
        f"frame {event['frame_start']}-{event['frame_end']}  peak {frame_idx}",
        " ".join(event.get("tokens") or []),
    ]
    if event.get("prob") is not None:
        lines.append(f"prob={float(event['prob']):.3f}")
    y = 22
    for line in lines:
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22

    cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return out


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>事件播放器</title>
<style>
  :root {
    --bg: #0f1218; --panel: #1a2030; --text: #e8ecf4; --muted: #8b93a7;
    --tp: #3dd68c; --fn: #f0a020; --fp: #f07178; --bound: #9aa0b4;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
         background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; }
  header { padding: 10px 16px; background: var(--panel); display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  select, button { background: #243049; color: var(--text); border: 1px solid #33405c;
                   border-radius: 6px; padding: 6px 10px; font-size: 13px; cursor: pointer; }
  button:hover { background: #2d3c5c; }
  .main { flex: 1; display: grid; grid-template-columns: 1.2fr 1fr; gap: 12px; padding: 12px; min-height: 0; }
  @media (max-width: 960px) { .main { grid-template-columns: 1fr; } }
  .pane { background: var(--panel); border-radius: 10px; padding: 10px; min-height: 0;
          display: flex; flex-direction: column; gap: 8px; overflow-y: auto; }
  .pane > .meta, .pane > .now, .pane > #gtbar { flex: none; }
  video, img.overlay { width: 100%; background: #000; border-radius: 8px; max-height: 48vh; object-fit: contain; }
  .meta { font-size: 14px; line-height: 1.5; }
  .meta .label { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
  .label.tp { color: var(--tp); } .label.fn { color: var(--fn); }
  .label.fp { color: var(--fp); } .label.fp_boundary { color: var(--bound); }
  .help { color: var(--muted); font-size: 12px; }
  .list { overflow: auto; flex: 1; font-size: 12px; }
  .list div { padding: 4px 6px; border-radius: 4px; cursor: pointer; display: flex; gap: 8px; }
  .list div:hover { background: #243049; }
  .list div.on { background: #2a3858; }
  .k { width: 52px; font-weight: 600; }
  .k.tp { color: var(--tp); } .k.fn { color: var(--fn); }
  .k.fp { color: var(--fp); } .k.fp_boundary { color: var(--bound); }
  .paused-banner { background: #3a2a10; color: #ffd28a; padding: 8px 10px; border-radius: 6px;
                   font-weight: 600; display: none; }
  .paused-banner.on { display: block; }
  /* 拣货事件模式 */
  canvas#bar { width: 100%; height: 46px; background: #141a26; border-radius: 6px; display: block; }
  .barhelp { color: var(--muted); font-size: 11px; display: flex; gap: 14px; flex-wrap: wrap; }
  .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: -1px; margin-right: 4px; }
  /* .list div 是 pkg 模式的 flex 行规则，这里的两级列表要显式改回块级 */
  .list .seg { display: block; margin-top: 6px; }
  .seg > .hd { padding: 5px 6px; border-radius: 4px; cursor: pointer; background: #202942;
               display: flex; gap: 8px; align-items: baseline;
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .seg > .hd:hover { background: #2a3556; }
  .seg > .hd.on { background: #34426b; }
  .seg .warn { color: var(--fn); font-weight: 600; }
  .seg .ok { color: var(--tp); }
  .burst { padding: 3px 6px 3px 22px; border-radius: 4px; cursor: pointer; display: flex; gap: 8px;
           white-space: nowrap; }
  .burst:hover { background: #243049; }
  .burst.on { background: #2a3858; }
  .burst .n { color: var(--tp); width: 26px; }
  .now { font-size: 13px; padding: 6px 8px; border-radius: 6px; background: #141a26; }
  .now b.yes { color: var(--tp); } .now b.no { color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1 id="title">事件播放器</h1>
  <select id="clip"></select>
  <button id="btnPrev" title="上一个">上一个</button>
  <button id="btnNext" title="空格：播到下一个并停">继续 ▶</button>
  <button id="btnJump">跳到当前</button>
  <span class="help" id="hint">空格=继续播到下一事件并停 · ←/→ 上一个/下一个 · 1..9 选片</span>
</header>
<div class="main">
  <div class="pane">
    <div id="paused" class="paused-banner">已停在事件上 — 按空格继续</div>
    <video id="v" controls preload="auto"></video>
    <div id="gtbar" style="display:none">
      <canvas id="bar" width="1200" height="46"></canvas>
      <div class="barhelp">
        <span><i class="sw" style="background:#3dd68c"></i>该帧有人工标注</span>
        <span><i class="sw" style="background:#f0a020"></i>单次拣货的范围</span>
        <span><i class="sw" style="background:#5b6480"></i>段内空档（无标注）</span>
        <span><i class="sw" style="background:#fff"></i>当前播放位置</span>
      </div>
      <div class="now" id="now"></div>
    </div>
    <div class="meta" id="meta">加载中…</div>
  </div>
  <div class="pane">
    <img class="overlay" id="ov" alt="事件叠加帧"/>
    <div class="help" id="ovhelp">右侧为峰值帧叠加（货框高亮 + 骨架）。左侧为原视频回放。</div>
    <div class="list" id="list"></div>
  </div>
</div>
<script>
let clips = [];
let clip = null;
let idx = 0;
let armNext = false;
let pendingIdx = -1;
let GT = false;              // 拣货事件模式
let segI = 0, burstI = 0;    // 当前段 / 当前单次拣货
let stopAt = -1;             // 播到该时间自动停
const v = document.getElementById('v');
const ov = document.getElementById('ov');
const meta = document.getElementById('meta');
const list = document.getElementById('list');
const sel = document.getElementById('clip');
const paused = document.getElementById('paused');
const bar = document.getElementById('bar');
const now = document.getElementById('now');

async function boot() {
  const r = await fetch('/api/catalog');
  const data = await r.json();
  clips = data.clips;
  GT = clips.length > 0 && clips[0].mode === 'gt';
  if (GT) {
    document.getElementById('title').textContent = '拣货事件播放器';
    document.getElementById('gtbar').style.display = '';
    document.getElementById('ov').style.display = 'none';
    document.getElementById('ovhelp').textContent =
      '左列表：粗体=manifest 现在的「段」，缩进=按标注间隔切出的「单次拣货」。';
    document.getElementById('hint').textContent =
      '空格=播下一次拣货 · ←/→ 上一次/下一次 · A=连续播整段（含空档） · 1..9 选片';
    document.getElementById('btnNext').textContent = '播这次拣货 ▶';
    document.getElementById('btnJump').textContent = '连续播整段';
    sel.innerHTML = clips.map((c,i) =>
      `<option value="${i}">${c.camera} · ${c.id} · 段${c.n_segments} → 单次拣货${c.n_bursts}</option>`
    ).join('');
    sel.onchange = () => loadClipGt(+sel.value);
    document.getElementById('btnNext').onclick = () => playBurst();
    document.getElementById('btnPrev').onclick = () => stepBurst(-1);
    document.getElementById('btnJump').onclick = () => playWholeSegment();
    window.addEventListener('resize', drawBar);
  } else {
    sel.innerHTML = clips.map((c,i) =>
      `<option value="${i}">${c.camera} · ${c.id} · 事件${c.n_events}（对${c.counts.tp}/漏${c.counts.fn}/误${c.counts.fp}）</option>`
    ).join('');
    sel.onchange = () => loadClip(+sel.value);
    document.getElementById('btnNext').onclick = continuePlay;
    document.getElementById('btnPrev').onclick = () => gotoEvent(idx - 1, true);
    document.getElementById('btnJump').onclick = () => gotoEvent(idx, true);
  }
  window.addEventListener('keydown', onKey);
  v.addEventListener('timeupdate', GT ? onTimeGt : onTime);
  if (clips.length) (GT ? loadClipGt : loadClip)(0);
}

function onKey(e) {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  if (e.key >= '1' && e.key <= '9') {
    const i = +e.key - 1;
    if (i < clips.length) { sel.value = String(i); (GT ? loadClipGt : loadClip)(i); }
    return;
  }
  if (GT) {
    if (e.code === 'Space') { e.preventDefault(); playBurst(); }
    else if (e.code === 'ArrowRight') { e.preventDefault(); stepBurst(1); }
    else if (e.code === 'ArrowLeft') { e.preventDefault(); stepBurst(-1); }
    else if (e.key === 'a' || e.key === 'A') { e.preventDefault(); playWholeSegment(); }
    return;
  }
  if (e.code === 'Space') { e.preventDefault(); continuePlay(); }
  else if (e.code === 'ArrowRight') { e.preventDefault(); gotoEvent(idx + 1, true); }
  else if (e.code === 'ArrowLeft') { e.preventDefault(); gotoEvent(idx - 1, true); }
}

/* ---------- 拣货事件模式 ---------- */

function loadClipGt(i) {
  clip = clips[i];
  segI = 0; burstI = 0; stopAt = -1;
  v.src = '/video/' + encodeURIComponent(clip.id);
  v.load();
  renderListGt();
  v.onloadeddata = () => selectBurst(0, 0, true);
}

function segs() { return clip.segments; }
function curSeg() { return clip.segments[segI]; }
function curBurst() { const s = curSeg(); return s && s.bursts[burstI]; }
function frameOf(t) {
  return Math.round((t - clip.video_start_pts_sec) * clip.fps) + 1;
}

function renderListGt() {
  list.innerHTML = segs().map((s, si) => {
    const nb = s.bursts.length;
    const cls = nb > 1 ? 'warn' : 'ok';
    const dur = (s.span / clip.fps).toFixed(1);
    return `<div class="seg">
      <div class="hd" data-s="${si}">
        <span>段#${si+1}</span>
        <span>${dur}s</span>
        <span>标注${s.n_labeled}/${s.span}帧=${(s.fill*100).toFixed(0)}%</span>
        <span class="${cls}">${nb>1 ? '含 '+nb+' 次拣货' : '1 次拣货'}</span>
        <span class="help">${s.tokens.join(',')}</span>
      </div>
      ${s.bursts.map((b, bi) => {
        const bd = ((b.frame_end - b.frame_start + 1) / clip.fps).toFixed(1);
        return `<div class="burst" data-s="${si}" data-b="${bi}">
          <span class="n">${bi+1}</span>
          <span>${bd}s</span><span class="help">f${b.frame_start}-${b.frame_end}（标注${b.n_labeled}帧）</span>
        </div>`;
      }).join('')}
    </div>`;
  }).join('');
  list.querySelectorAll('.hd').forEach(el =>
    el.onclick = () => { selectBurst(+el.dataset.s, 0, true); playWholeSegment(); });
  list.querySelectorAll('.burst').forEach(el =>
    el.onclick = () => selectBurst(+el.dataset.s, +el.dataset.b, true));
}

function selectBurst(si, bi, seek) {
  segI = Math.max(0, Math.min(segs().length - 1, si));
  const s = curSeg();
  burstI = Math.max(0, Math.min(Math.max(0, s.bursts.length - 1), bi));
  stopAt = -1;
  v.pause();
  paused.classList.add('on');
  const b = curBurst();
  if (seek && b) v.currentTime = Math.max(0, b.t_start);
  showMetaGt();
  drawBar();
  highlightGt();
}

function stepBurst(d) {
  let si = segI, bi = burstI + d;
  while (bi < 0) { si--; if (si < 0) { si = 0; bi = 0; break; } bi = segs()[si].bursts.length - 1; }
  while (si < segs().length && bi >= segs()[si].bursts.length) {
    bi -= segs()[si].bursts.length; si++;
  }
  if (si >= segs().length) { si = segs().length - 1; bi = segs()[si].bursts.length - 1; }
  selectBurst(si, bi, true);
}

function playBurst() {
  const b = curBurst();
  if (!b) return;
  if (v.paused && Math.abs(v.currentTime - b.t_start) > 0.05) v.currentTime = b.t_start;
  if (v.paused && v.currentTime >= b.t_end - 0.02) { stepBurst(1); return; }
  stopAt = b.t_end + 1 / clip.fps;
  paused.classList.remove('on');
  v.play();
}

function playWholeSegment() {
  const s = curSeg();
  if (!s) return;
  v.currentTime = s.t_start;
  stopAt = s.t_end + 1 / clip.fps;
  paused.classList.remove('on');
  v.play();
}

function showMetaGt() {
  const s = curSeg(); if (!s) { meta.textContent = '本片无段'; return; }
  const nb = s.bursts.length;
  const warn = nb > 1
    ? `<div class="warn">这一个「段」里其实有 ${nb} 次拣货 —— 现在的评估把它当成 1 个事件</div>`
    : `<div class="ok">这个段就是 1 次拣货</div>`;
  meta.innerHTML = `
    <div class="label">段 ${segI+1} / ${segs().length}　第 ${burstI+1} / ${nb} 次拣货</div>
    ${warn}
    <div>段范围：帧 ${s.frame_start} → ${s.frame_end}（${(s.span/clip.fps).toFixed(1)} 秒），
         其中有标注 ${s.n_labeled} 帧 = ${(s.fill*100).toFixed(0)}%</div>
    <div>货框：${s.tokens.join(', ') || '—'}　人：${s.tracks.join(',') || '—'}　[${s.split}]</div>
    <div class="help">${clip.record_id}</div>`;
}

function highlightGt() {
  list.querySelectorAll('.hd').forEach(el =>
    el.classList.toggle('on', +el.dataset.s === segI));
  list.querySelectorAll('.burst').forEach(el =>
    el.classList.toggle('on', +el.dataset.s === segI && +el.dataset.b === burstI));
  const on = list.querySelector('.burst.on') || list.querySelector('.hd.on');
  if (on) on.scrollIntoView({block:'nearest'});
}

function drawBar() {
  const s = curSeg(); if (!s) return;
  const w = bar.clientWidth || 1200;
  bar.width = w; bar.height = 46;
  const g = bar.getContext('2d');
  g.clearRect(0,0,w,46);
  const a = s.frame_start, b = s.frame_end, span = Math.max(1, b - a);
  const X = f => Math.round((f - a) / span * (w - 1));
  g.fillStyle = '#5b6480'; g.fillRect(0, 16, w, 14);
  s.bursts.forEach((bu, i) => {
    g.fillStyle = (i === burstI) ? '#ffbe4d' : '#f0a020';
    const x0 = X(bu.frame_start), x1 = X(bu.frame_end);
    g.fillRect(x0, 12, Math.max(2, x1 - x0), 22);
    g.fillStyle = '#0f1218'; g.font = '10px sans-serif';
    if (x1 - x0 > 12) g.fillText(String(i+1), x0 + 2, 27);
  });
  g.fillStyle = '#3dd68c';
  s.labeled.forEach(f => g.fillRect(X(f), 18, 1, 10));
  const cf = frameOf(v.currentTime);
  if (cf >= a - span*0.02 && cf <= b + span*0.02) {
    g.fillStyle = '#fff'; g.fillRect(X(Math.max(a, Math.min(b, cf))), 4, 2, 38);
  }
  g.fillStyle = '#8b93a7'; g.font = '10px sans-serif';
  g.fillText('f'+a, 2, 10); 
  const et = 'f'+b; g.fillText(et, w - g.measureText(et).width - 2, 10);
}

function onTimeGt() {
  const s = curSeg(); if (!s) return;
  if (stopAt > 0 && v.currentTime >= stopAt) {
    v.pause(); stopAt = -1; paused.classList.add('on');
  }
  const cf = frameOf(v.currentTime);
  const inSeg = cf >= s.frame_start && cf <= s.frame_end;
  const hit = s.labeled.includes(cf);
  let which = -1;
  s.bursts.forEach((b,i) => { if (cf >= b.frame_start && cf <= b.frame_end) which = i; });
  if (which >= 0 && which !== burstI && !v.paused) { burstI = which; highlightGt(); showMetaGt(); }
  now.innerHTML = `当前帧 ${cf}　`
    + (hit ? '<b class="yes">有人工标注（正在拣货）</b>' : '<b class="no">无标注</b>')
    + (inSeg ? '' : ' <span class="help">（已出段范围）</span>')
    + (which >= 0 ? `　处于第 ${which+1} 次拣货内` : '　处于段内空档');
  drawBar();
}

function loadClip(i) {
  clip = clips[i];
  idx = 0;
  armNext = false;
  pendingIdx = -1;
  v.src = '/video/' + encodeURIComponent(clip.id);
  v.load();
  renderList();
  v.onloadeddata = () => gotoEvent(0, true);
}

function renderList() {
  list.innerHTML = clip.events.map((e,i) =>
    `<div data-i="${i}" class="${i===idx?'on':''}">
      <span class="k ${e.kind}">${e.label}</span>
      <span>#${i+1} f${e.frame_start}-${e.frame_end} ${e.tokens.join(',')}</span>
    </div>`
  ).join('');
  list.querySelectorAll('div').forEach(el => el.onclick = () => gotoEvent(+el.dataset.i, true));
}

function cur() { return clip.events[idx]; }

function showMeta() {
  const e = cur();
  if (!e) { meta.textContent = '无事件'; return; }
  const gap = (e.gap_to_gt == null) ? '' : ` · 距真值 ${e.gap_to_gt} 帧`;
  const prob = (e.prob == null) ? '' : ` · 分数 ${Number(e.prob).toFixed(3)}`;
  meta.innerHTML = `
    <div class="label ${e.kind}">${e.label}</div>
    <div>事件 ${idx+1} / ${clip.n_events}${gap}${prob}</div>
    <div>帧 ${e.frame_start} → ${e.frame_end}（峰值 ${e.peak_frame}） · ${e.t_start.toFixed(2)}s–${e.t_end.toFixed(2)}s</div>
    <div>货框：${e.tokens.join(', ') || '—'}</div>
    <div class="help">${clip.record_id}</div>`;
  ov.src = `/overlay/${encodeURIComponent(clip.id)}/${e.index}?t=${Date.now()}`;
  [...list.children].forEach((el,i) => el.classList.toggle('on', i===idx));
  const on = list.querySelector('.on');
  if (on) on.scrollIntoView({block:'nearest'});
}

function gotoEvent(i, seek) {
  if (!clip || !clip.events.length) return;
  idx = Math.max(0, Math.min(clip.events.length - 1, i));
  armNext = false;
  pendingIdx = -1;
  v.pause();
  paused.classList.add('on');
  const e = cur();
  if (seek) v.currentTime = Math.max(0, e.t_peak);
  showMeta();
}

function continuePlay() {
  if (!clip || !clip.events.length) return;
  if (idx >= clip.events.length - 1) {
    paused.classList.remove('on');
    meta.innerHTML += '<div class="help">已到本片最后一个事件。换片或按 ← 回看。</div>';
    return;
  }
  const end = Math.max(cur().t_end + 1 / clip.fps, cur().t_peak + 0.04);
  pendingIdx = idx + 1;
  const next = clip.events[pendingIdx];
  armNext = true;
  paused.classList.remove('on');
  if (next.t_start <= end + 0.05) {
    gotoEvent(pendingIdx, true);
    return;
  }
  v.currentTime = end;
  v.play();
}

function onTime() {
  if (!armNext || !clip || pendingIdx < 0) return;
  const next = clip.events[pendingIdx];
  if (v.currentTime + 0.02 >= next.t_start) {
    gotoEvent(pendingIdx, true);
  }
}

boot();
</script>
</body>
</html>
"""


class PlayerState:
    def __init__(self, clips: list[dict[str, Any]], cache_dir: Path):
        self.clips = clips
        self.by_id = {c["id"]: c for c in clips}
        self.cache_dir = cache_dir
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    state: PlayerState

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str, *, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, data, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path in ("/", "/index.html"):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/catalog":
            slim = []
            for c in self.state.clips:
                base = {
                    "id": c["id"],
                    "record_id": c["record_id"],
                    "camera": c["camera"],
                    "fps": c["fps"],
                    "video_start_pts_sec": c.get("video_start_pts_sec", 0.0),
                }
                if c.get("mode") == "gt":
                    base.update(
                        {
                            "mode": "gt",
                            "n_segments": c["n_segments"],
                            "n_bursts": c["n_bursts"],
                            "segments": c["segments"],
                        }
                    )
                else:
                    base.update(
                        {
                            "n_events": c["n_events"],
                            "counts": c["counts"],
                            "events": c["events"],
                        }
                    )
                slim.append(base)
            self._json({"clips": slim})
            return

        if path.startswith("/overlay/"):
            parts = path.strip("/").split("/")
            if len(parts) != 3:
                self._send(404, b"not found", "text/plain")
                return
            _, clip_id, ev_s = parts
            clip = self.state.by_id.get(clip_id)
            if not clip:
                self._send(404, b"clip", "text/plain")
                return
            try:
                ev_i = int(ev_s.split("?")[0])
            except ValueError:
                self._send(400, b"bad event", "text/plain")
                return
            if ev_i < 0 or ev_i >= len(clip["events"]):
                self._send(404, b"event", "text/plain")
                return
            with self.state.lock:
                jpg = _render_overlay(clip, clip["events"][ev_i], self.state.cache_dir)
            body = jpg.read_bytes()
            self._send(200, body, "image/jpeg")
            return

        if path.startswith("/video/"):
            clip_id = path[len("/video/") :]
            clip = self.state.by_id.get(clip_id)
            if not clip:
                self._send(404, b"video not found", "text/plain")
                return
            self._serve_file_range(Path(clip["video_path"]))
            return

        self._send(404, b"not found", "text/plain")

    def _serve_file_range(self, file_path: Path) -> None:
        if not file_path.is_file():
            self._send(404, b"missing", "text/plain")
            return
        size = file_path.stat().st_size
        ctype = mimetypes.guess_type(str(file_path))[0] or "video/mp4"
        range_hdr = self.headers.get("Range")
        if not range_hdr:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with file_path.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 256)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            return

        # bytes=start-end
        try:
            units, _, rng = range_hdr.partition("=")
            if units.strip() != "bytes":
                raise ValueError("unit")
            start_s, _, end_s = rng.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
            end = min(end, size - 1)
            if start > end:
                raise ValueError("range")
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return

        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with file_path.open("rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(1024 * 256, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def main() -> int:
    ap = argparse.ArgumentParser(description="网页事件播放器")
    ap.add_argument("--pkg", default="", help="导出目录（检对/误报/漏报模式）")
    ap.add_argument(
        "--manifest",
        default="",
        help="tagged manifest 路径（拣货事件模式，只看标注不看推理结果）",
    )
    ap.add_argument("--split", default="", help="拣货事件模式：只看 train 或 val，留空为全部")
    ap.add_argument(
        "--gap",
        type=int,
        default=25,
        help="拣货事件模式：标注帧间隔超过它就算两次拣货（源帧，25fps 下 25=1 秒）",
    )
    ap.add_argument(
        "--clips",
        default="",
        help="逗号分隔的导出文件 stem；默认加载 --pkg 下全部 clip",
    )
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true", help="不尝试打开浏览器")
    args = ap.parse_args()

    if args.manifest.strip():
        return _serve_gt(args)
    if not args.pkg.strip():
        print("需要 --pkg 或 --manifest", file=sys.stderr)
        return 1

    pkg = Path(args.pkg)
    if not pkg.is_absolute():
        pkg = ROOT / pkg
    if not pkg.is_dir():
        print(f"目录不存在: {pkg}", file=sys.stderr)
        return 1

    skip_names = {"_manifest.json"}
    if args.clips.strip():
        export_files: list[Path] = []
        for stem in [x.strip() for x in args.clips.split(",") if x.strip()]:
            p = pkg / f"{stem}.json"
            if not p.is_file():
                p = pkg / stem
            if not p.is_file():
                print(f"[skip] 找不到导出: {stem}", file=sys.stderr)
                continue
            export_files.append(p)
    else:
        export_files = sorted(
            p
            for p in pkg.glob("*.json")
            if p.name not in skip_names
            and not p.name.startswith("eval_")
            and not p.name.startswith("accuracy_")
        )
    paths = load_paths()

    clips: list[dict[str, Any]] = []
    for p in export_files:
        print(f"收集事件: {p.name} …", flush=True)
        clip = _collect_clip_events(p, paths)
        if clip is None:
            print(f"[skip] 无法收集: {p.name}", file=sys.stderr)
            continue
        c = clip["counts"]
        print(
            f"  {clip['camera']}  事件 {clip['n_events']}  "
            f"检对{c['tp']} 漏报{c['fn']} 误报{c['fp']} 边界{c['fp_boundary']}",
            flush=True,
        )
        clips.append(clip)

    if not clips:
        print("没有可播放的片子", file=sys.stderr)
        return 1

    cache_dir = ROOT / "output" / "viz" / "event_player_cache" / pkg.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return _run_server(
        clips, cache_dir, args, hint="空格=继续播到下一事件并停  ←/→=上一个/下一个事件"
    )


def _serve_gt(args) -> int:
    man = Path(args.manifest)
    if not man.is_absolute():
        man = ROOT / man
    if not man.is_file():
        print(f"manifest 不存在: {man}", file=sys.stderr)
        return 1
    paths = load_paths()
    print(f"读标注段（间隔 >{args.gap} 帧算两次拣货）…", flush=True)
    clips = _collect_gt_clips(man, paths, gap=args.gap, split=args.split.strip())
    if not clips:
        print("没有可播放的切片", file=sys.stderr)
        return 1
    n_seg = sum(c["n_segments"] for c in clips)
    n_bur = sum(c["n_bursts"] for c in clips)
    multi = sum(1 for c in clips for s in c["segments"] if len(s["bursts"]) > 1)
    print(
        f"\n{len(clips)} 条切片：manifest 段 {n_seg} 个 → 按 {args.gap} 帧间隔切出单次拣货 {n_bur} 次",
        flush=True,
    )
    print(f"其中 {multi} 个段含多次拣货（被合并成了一个事件）", flush=True)
    cache_dir = ROOT / "output" / "viz" / "event_player_cache" / "gt"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return _run_server(
        clips,
        cache_dir,
        args,
        hint="空格=播下一次拣货  A=连续播整段（含空档）  ←/→=上一次/下一次",
    )


def _run_server(clips: list[dict[str, Any]], cache_dir: Path, args, *, hint: str) -> int:
    Handler.state = PlayerState(clips, cache_dir)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"\n打开: {url}", flush=True)
    print(f"操作: {hint}", flush=True)
    print("远程 Cursor: 把端口转发到本地后再用浏览器打开上述地址", flush=True)
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
