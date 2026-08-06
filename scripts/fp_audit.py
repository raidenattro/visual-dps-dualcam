#!/usr/bin/env python3
"""误报审核网页工具：逐个回放高分误报，人工判定「漏标 / 不是拣货 / 不确定」。

目的：模型给出高分但标注说没拣货的片段，需要先确认标注是否漏了，
否则压误报等于在拟合错误的标签。

判定结果写 output/audit/fp_audit.json，可中断后继续。
只读 collector。
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_paths
from adapters.record_reader import list_records_from_manifest, load_record, record_dir
from pipeline.box_trigger import BoxTrigger
from scripts.event_player import Handler as BaseHandler
from scripts.event_player import _frame_to_sec, _resolve_video
from scripts.render_fn_fp_frames import _read_frame
from scripts.render_missed_segments import _banner, _draw_boxes, _draw_person

HTML = """<!doctype html>
<html lang="zh"><meta charset="utf-8"/>
<title>误报审核</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 system-ui,"PingFang SC","Microsoft YaHei",sans-serif;
         background:#0f1115; color:#e8eaed; }
  header { padding:10px 14px; background:#171a21; display:flex; gap:12px; align-items:center;
           flex-wrap:wrap; border-bottom:1px solid #262b36; }
  h1 { font-size:16px; margin:0; }
  .stat { font-size:13px; color:#9aa4b2; }
  .stat b { color:#e8eaed; }
  .main { display:grid; grid-template-columns: 1fr 1fr; gap:12px; padding:12px; align-items:start; }
  .pane { min-width:0; }
  video, .overlay { width:100%; border-radius:8px; background:#000; display:block; }
  .info { margin-top:10px; padding:10px 12px; background:#171a21; border-radius:8px; font-size:13px; }
  .info div { margin:3px 0; }
  .k { color:#9aa4b2; display:inline-block; min-width:76px; }
  .btns { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; }
  button { font:14px inherit; padding:10px 14px; border-radius:8px; border:1px solid #313847;
           background:#1e2430; color:#e8eaed; cursor:pointer; }
  button:hover { background:#273044; }
  button.miss { border-color:#3dd68c; }
  button.notpick { border-color:#f0a020; }
  button.unsure { border-color:#5b6480; }
  .help { color:#9aa4b2; font-size:12px; margin-top:8px; }
  .list { margin-top:12px; max-height:46vh; overflow-y:auto; background:#131722; border-radius:8px; }
  .row { padding:6px 10px; border-bottom:1px solid #1d2331; cursor:pointer;
         display:flex; gap:8px; align-items:center; font-size:12.5px; }
  .row:hover { background:#1a2030; }
  .row.cur { background:#243049; }
  .tag { font-size:11px; padding:1px 6px; border-radius:10px; border:1px solid #313847; color:#9aa4b2; }
  .tag.miss { border-color:#3dd68c; color:#3dd68c; }
  .tag.wrongbox { border-color:#e05c8a; color:#e05c8a; }
  .tag.notpick { border-color:#f0a020; color:#f0a020; }
  .tag.unsure { border-color:#5b6480; color:#8b93a5; }
  .tag.old { border-color:#7a6320; color:#c9a227; }
  button.wrongbox { border-color:#e05c8a; }
  .sc { color:#e8eaed; min-width:44px; }
  .warn { color:#c9a227; }
</style>
<header>
  <h1>误报审核</h1>
  <span class="stat">已判 <b id="done">0</b>/<b id="total">0</b>　
    拣这个框 <b id="nmiss">0</b>　拣别的框 <b id="nwrong">0</b>　
    没拣货 <b id="nnot">0</b>　不确定 <b id="nunsure">0</b></span>
  <span class="stat" id="pos"></span>
</header>
<div class="main">
  <div class="pane">
    <video id="v" controls preload="auto" muted></video>
    <div class="info" id="info">加载中…</div>
    <div class="btns">
      <button class="miss" id="b1">1　在拣这个红框（标注漏了）</button>
      <button class="wrongbox" id="b2">2　在拣货，但不是这个框</button>
      <button class="notpick" id="b3">3　没在拣货</button>
      <button class="unsure" id="b4">4　不确定</button>
      <button id="b0">空格　重播</button>
    </div>
    <div class="help">1/2/3/4 判定并自动跳下一个 · ←/→ 上一个/下一个 · 空格重播</div>
  </div>
  <div class="pane">
    <img class="overlay" id="ov" alt="峰值帧"/>
    <div class="help">判断的是<b>黄框内那个人</b>有没有在拣<b>红框</b>——黄色箭头从他的手腕指向该货框。
      其他人画成暗蓝色，不用管。左侧视频循环播放这个片段（前后各留 1 秒）。</div>
    <div class="list" id="list"></div>
  </div>
</div>
<script>
let EV = [], cur = 0, curVideo = '';
const v = document.getElementById('v');

function fmt(t){ const m=Math.floor(t/60), s=(t%60).toFixed(1); return `${m}:${s.padStart(4,'0')}`; }

async function boot(){
  const r = await fetch('/api/events');
  const d = await r.json();
  EV = d.events;
  document.getElementById('total').textContent = EV.length;
  renderList(); stats();
  cur = EV.findIndex(e => !e.verdict);
  if (cur < 0) cur = 0;
  show(cur, true);
  document.getElementById('b1').onclick = () => verdict('missing');
  document.getElementById('b2').onclick = () => verdict('wrong_box');
  document.getElementById('b3').onclick = () => verdict('not_pick');
  document.getElementById('b4').onclick = () => verdict('unsure');
  document.getElementById('b0').onclick = () => replay();
  document.addEventListener('keydown', onKey);
  v.addEventListener('timeupdate', onTime);
}

function show(i, scroll){
  if (i < 0 || i >= EV.length) return;
  cur = i;
  const e = EV[i];
  if (e.video_key !== curVideo){
    curVideo = e.video_key;
    v.src = '/video/' + encodeURIComponent(e.video_key);
    v.addEventListener('loadedmetadata', replay, { once:true });
  } else {
    replay();
  }
  document.getElementById('ov').src = '/overlay/' + i + '?t=' + Date.now();
  document.getElementById('info').innerHTML =
    `<div><span class="k">货框</span>${e.token}</div>` +
    `<div><span class="k">模型分数</span>${e.peak}　（真实拣货段峰值分中位 0.88）</div>` +
    `<div><span class="k">判断对象</span>黄框内的人 #${e.person_track_id || '?'}　×　红框 ${e.token}</div>` +
    (e.kind_desc ? `<div><span class="k">误报类型</span>${e.kind_desc}</div>` : '') +
    `<div><span class="k">持续</span>${e.n} 帧　源帧 ${e.span[0]}–${e.span[1]}　视频 ${fmt(e.t_start)}–${fmt(e.t_end)}</div>` +
    `<div><span class="k">视频</span>${e.clip}</div>` +
    (e.verdict && e.verdict_v !== 2
      ? `<div class="warn">这条是旧口径判的（当时没区分「拣别的框」），建议重判</div>` : '');
  document.getElementById('pos').textContent = `第 ${i+1} 个 / 共 ${EV.length}`;
  highlight(scroll);
}

// 短片段（几帧）一闪而过看不清，前后留一段上下文再循环
const PAD = 1.0;
function replay(){
  const e = EV[cur];
  try { v.currentTime = Math.max(0, e.t_start - PAD); } catch(_) {}
  v.play().catch(()=>{});
}

function onTime(){
  const e = EV[cur];
  if (!e) return;
  if (v.currentTime > e.t_end + PAD) replay();
}

async function verdict(kind){
  const e = EV[cur];
  e.verdict = kind;
  e.verdict_v = 2;
  updateRow(cur); stats();
  fetch('/api/verdict', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ idx: cur, verdict: kind }) });
  const next = EV.findIndex((x,i) => i > cur && !x.verdict);
  // 判定后不滚动列表，否则找不回刚才操作的是哪一条
  show(next >= 0 ? next : Math.min(cur+1, EV.length-1), false);
}

function onKey(ev){
  if (ev.key === '1') { ev.preventDefault(); verdict('missing'); }
  else if (ev.key === '2') { ev.preventDefault(); verdict('wrong_box'); }
  else if (ev.key === '3') { ev.preventDefault(); verdict('not_pick'); }
  else if (ev.key === '4') { ev.preventDefault(); verdict('unsure'); }
  else if (ev.key === ' ') { ev.preventDefault(); replay(); }
  else if (ev.key === 'ArrowLeft') { ev.preventDefault(); show(cur-1, true); }
  else if (ev.key === 'ArrowRight') { ev.preventDefault(); show(cur+1, true); }
}

const LABEL = { missing:'拣这个框', wrong_box:'拣别的框', not_pick:'没拣货', unsure:'不确定' };
const CLS = { missing:'miss', wrong_box:'wrongbox', not_pick:'notpick', unsure:'unsure' };

function rowHTML(e, i){
  const tag = e.verdict
    ? `<span class="tag ${CLS[e.verdict]}">${LABEL[e.verdict]}</span>` +
      (e.verdict_v !== 2 ? `<span class="tag old">旧</span>` : '')
    : '';
  return `<span class="sc">${e.peak}</span>
       <span>${e.token}</span>
       <span style="color:#8b93a5">#${e.person_track_id || '?'}</span>
       <span style="color:#8b93a5">${e.n}帧</span>
       <span style="color:#6b7383;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${e.clip}</span>
       ${tag}`;
}

function renderList(){
  const el = document.getElementById('list');
  el.innerHTML = EV.map((e,i) =>
    `<div class="row${i===cur?' cur':''}" data-i="${i}">${rowHTML(e,i)}</div>`).join('');
  el.querySelectorAll('.row').forEach(r =>
    r.onclick = () => show(parseInt(r.dataset.i, 10), false));
}

// 只重画那一行，保持滚动位置不变
function updateRow(i){
  const r = document.querySelectorAll('.row')[i];
  if (r) r.innerHTML = rowHTML(EV[i], i);
}

function highlight(scroll){
  document.querySelectorAll('.row').forEach((r,i) => r.classList.toggle('cur', i===cur));
  if (!scroll) return;
  const c = document.querySelector('.row.cur');
  if (c) c.scrollIntoView({ block:'nearest' });
}

function stats(){
  const c = { missing:0, wrong_box:0, not_pick:0, unsure:0 };
  EV.forEach(e => { if (e.verdict) c[e.verdict]++; });
  document.getElementById('done').textContent =
    c.missing + c.wrong_box + c.not_pick + c.unsure;
  document.getElementById('nmiss').textContent = c.missing;
  document.getElementById('nwrong').textContent = c.wrong_box;
  document.getElementById('nnot').textContent = c.not_pick;
  document.getElementById('nunsure').textContent = c.unsure;
}

boot();
</script>
</html>
"""


class AuditState:
    def __init__(self, events: list[dict[str, Any]], videos: dict[str, str],
                 records: dict[str, Any], cache_dir: Path, out_path: Path):
        self.events = events
        self.videos = videos
        self.records = records
        self.cache_dir = cache_dir
        self.out_path = out_path
        self.lock = threading.Lock()

    strata: list[dict[str, Any]] = []

    def save(self) -> None:
        payload = {
            "strata": self.strata,
            "n_events": len(self.events),
            "verdicts": {
                str(i): e["verdict"] for i, e in enumerate(self.events) if e.get("verdict")
            },
            "events": self.events,
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )


C_FOCUS = (0, 220, 255)  # 触发误报的人：黄


def _mark_focus(img, person: dict[str, Any], token: str, rec: dict[str, Any]) -> None:
    """给被判断的人套黄框，并从触发的那只手腕画箭头指向货框。

    判定对象是「这个人 × 这个货框」，画面上必须指明是哪一个，否则多人同框时无法判。
    """
    pts = [
        (float(kp[0]), float(kp[1]))
        for kp in (person.get("keypoints") or [])
        if isinstance(kp, (list, tuple)) and len(kp) > 2 and float(kp[2]) >= 0.15
    ]
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, y0 = int(min(xs)) - 10, int(min(ys)) - 10
        x1, y1 = int(max(xs)) + 10, int(max(ys)) + 10
        cv2.rectangle(img, (x0, y0), (x1, y1), C_FOCUS, 2, cv2.LINE_AA)
        cv2.putText(img, "<< THIS PERSON", (x1 + 4, max(18, y0 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, "<< THIS PERSON", (x1 + 4, max(18, y0 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_FOCUS, 1, cv2.LINE_AA)

    box = rec["box_by_token"].get(token)
    if box is None:
        return
    for hit in rec["trigger"].hits_for_person(person):
        if hit["token"] != token:
            continue
        wx, wy = hit["wrist_xy"]
        cv2.arrowedLine(img, (int(wx), int(wy)),
                        (int(box.center[0]), int(box.center[1])),
                        C_FOCUS, 2, cv2.LINE_AA, tipLength=0.12)
        break


def _overlay(state: AuditState, idx: int) -> Path:
    ev = state.events[idx]
    out = state.cache_dir / f"fp_{idx:04d}.jpg"
    if out.is_file():
        return out
    rec = state.records[ev["record_id"]]
    img = _read_frame(Path(state.videos[ev["video_key"]]), int(ev["frame"]))
    if img is None:
        out.write_bytes(b"")
        return out
    iw, ih = rec["infer_width"], rec["infer_height"]
    if iw and ih and (img.shape[1] != iw or img.shape[0] != ih):
        img = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA)
    _draw_boxes(img, rec["boxes"], gt={ev["token"]}, touched=set())

    track = str(ev.get("person_track_id") or "")
    persons = [
        p for p in (rec["frames"].get(int(ev["frame"])) or {}).get("persons") or []
        if isinstance(p, dict)
    ]
    focus = next((p for p in persons if str(p.get("person_track_id")) == track), None)
    for p in persons:
        _draw_person(img, p, is_main=(p is focus or focus is None), wrist_min=0.15)
    if focus is not None:
        _mark_focus(img, focus, ev["token"], rec)

    _banner(img, [
        f"判断：黄框内这个人(#{track or '?'}) 是否在拣红框 {ev['token']}",
        f"模型分数 {ev['peak']}  持续 {ev['n']} 帧  峰值帧 f{ev['frame']}  {ev['clip'][:38]}",
    ])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out


class AuditHandler(BaseHandler):
    astate: AuditState

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/events":
            self._json({"events": self.astate.events})
            return
        if path.startswith("/overlay/"):
            try:
                idx = int(path[len("/overlay/") :].split("?")[0])
            except ValueError:
                self._send(400, b"bad idx", "text/plain")
                return
            if not 0 <= idx < len(self.astate.events):
                self._send(404, b"no event", "text/plain")
                return
            with self.astate.lock:
                jpg = _overlay(self.astate, idx)
            self._send(200, jpg.read_bytes(), "image/jpeg")
            return
        if path.startswith("/video/"):
            key = path[len("/video/") :]
            vp = self.astate.videos.get(key)
            if not vp:
                self._send(404, b"no video", "text/plain")
                return
            self._serve_file_range(Path(vp))
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path != "/api/verdict":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            idx = int(body["idx"])
            verdict = str(body["verdict"])
        except (ValueError, KeyError):
            self._json({"ok": False}, 400)
            return
        if not 0 <= idx < len(self.astate.events):
            self._json({"ok": False}, 400)
            return
        with self.astate.lock:
            self.astate.events[idx]["verdict"] = verdict
            # 版本 2 = 区分了「拣别的框」的口径
            self.astate.events[idx]["verdict_v"] = 2
            self.astate.save()
        self._json({"ok": True})


def _free_port(start: int) -> int:
    for port in range(start, start + 60):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit("找不到空闲端口")


def main() -> int:
    ap = argparse.ArgumentParser(description="误报审核网页工具")
    ap.add_argument("--events", default=str(ROOT / "output/sweep/v5/no_pick_fp.json"))
    ap.add_argument("--manifest", default=str(ROOT / "output/manifests/tagged_aug85_v2.json"))
    ap.add_argument("--min-peak", type=float, default=0.5, help="只审分数不低于它的误报")
    ap.add_argument("--limit", type=int, default=0, help="最多审几个，0=全部")
    ap.add_argument("--out", default=str(ROOT / "output/audit/fp_audit.json"))
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--open", action="store_true", help="启动后打开浏览器")
    args = ap.parse_args()

    paths = load_paths()
    raw = json.loads(Path(args.events).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        strata = raw.get("strata") or []
        raw = raw.get("events") or []
    else:
        strata = []
    evs = [e for e in raw if float(e.get("peak") or 0) >= args.min_peak]
    evs.sort(key=lambda e: -float(e["peak"]))
    if args.limit:
        evs = evs[: args.limit]
    if not evs:
        print("没有满足条件的误报事件")
        return 1

    need = {str(e["record_id"]) for e in evs}
    videos: dict[str, str] = {}
    records: dict[str, Any] = {}
    meta: dict[str, dict[str, Any]] = {}
    for ref in list_records_from_manifest(Path(args.manifest)):
        if ref.record_id not in need:
            continue
        rd = record_dir(ref, paths)
        rec_man = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
        video = _resolve_video(ref.record_id, rec_man, paths)
        if video is None or not video.is_file():
            print(f"[skip] 无视频 {ref.clip_name}")
            continue
        key = ref.clip_name
        videos[key] = str(video)
        rec = load_record(ref, paths)
        records[ref.record_id] = {
            "boxes": rec.boxes,
            "box_by_token": {b.token: b for b in rec.boxes},
            "trigger": BoxTrigger(rec.boxes, wrist_score_min=0.15),
            "frames": {
                int(f.get("source_frame_idx") or f.get("frame_idx") or 0): f for f in rec.frames
            },
            "infer_width": int(rec.meta.get("infer_width") or ref.infer_width or 0),
            "infer_height": int(rec.meta.get("infer_height") or ref.infer_height or 0),
        }
        meta[ref.record_id] = {
            "video_key": key,
            "clip": ref.clip_name,
            "fps": float(rec_man.get("fps") or rec_man.get("video_fps") or 25.0),
            "start_pts": float(rec_man.get("video_start_pts_sec") or 0.0),
        }
        print(f"[ok] {ref.clip_name[:44]}")

    events: list[dict[str, Any]] = []
    for e in evs:
        m = meta.get(str(e["record_id"]))
        if not m:
            continue
        fps, pts = m["fps"], m["start_pts"]
        events.append({
            **e,
            "video_key": m["video_key"],
            "clip": m["clip"],
            "t_start": round(_frame_to_sec(int(e["span"][0]), fps, start_pts=pts), 3),
            "t_end": round(_frame_to_sec(int(e["span"][1]), fps, start_pts=pts), 3),
            "verdict": str(e.get("verdict") or ""),
            "verdict_v": int(e.get("verdict_v") or (1 if e.get("verdict") else 0)),
        })

    out_path = Path(args.out)
    if out_path.is_file():
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
            prev = {
                (v["record_id"], v["token"], tuple(v["span"])): (
                    v.get("verdict"), int(v.get("verdict_v") or 1)
                )
                for v in (old.get("events") or [])
                if v.get("verdict")
            }
            n = n_old = 0
            for ev in events:
                k = (ev["record_id"], ev["token"], tuple(ev["span"]))
                hit = prev.get(k)
                if hit and hit[0]:
                    ev["verdict"], ev["verdict_v"] = hit[0], hit[1]
                    n += 1
                    n_old += hit[1] < 2
            if n:
                print(f"恢复已有判定 {n} 条" + (f"（其中 {n_old} 条是旧口径，建议重判）" if n_old else ""))
        except (ValueError, KeyError):
            pass

    cache = ROOT / "output/audit/cache"
    cache.mkdir(parents=True, exist_ok=True)
    state = AuditState(events, videos, records, cache, out_path)
    state.strata = strata
    AuditHandler.astate = state

    port = _free_port(args.port)
    srv = ThreadingHTTPServer(("0.0.0.0", port), AuditHandler)
    url = f"http://127.0.0.1:{port}/"
    print(
        f"\n待审 {len(events)} 个高分误报（分数 ≥ {args.min_peak}）\n"
        f"打开 {url}\n"
        f"键盘：1=确实在拣货(漏标)  2=不是拣货  3=不确定  空格=重播  ←/→=切换\n"
        f"判定实时写入 {out_path}\n"
    )
    if args.open:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n停止")
    finally:
        with state.lock:
            state.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
