#!/usr/bin/env python3
"""双路拣货面标注服务。视频从 output/dualcam 与 data 里选，标定写 output/calib。"""

from __future__ import annotations

import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dualcam_calib_paths import (
    CALIB_144_24 as CALIB,
    CALIB_ID,
    calib_file,
    migrate_misplaced_144_24,
    skel_file,
    stamp_144_24,
)
from scripts.dualcam_videos import list_videos, resolve_video
from scripts.solve_scene import (
    _wall_reproj_px,
    _wall_x_spread,
    snap_solved_walls_parametric,
    solve_dual,
)

PAGE = ROOT / "scripts" / "dualcam_annot.html"
PLAYER = ROOT / "scripts" / "dualcam_player.html"
GEOM = ROOT / "scripts" / "dualcam_geom.js"
VENDOR = ROOT / "scripts" / "vendor"
VIDEO = ROOT / "output" / "dualcam" / "src.mp4"
SKEL_DIR = ROOT / "output" / "dualcam"
SKEL_PREFERRED = SKEL_DIR / "skel3d_144_24.json"
SKEL_FALLBACKS = (
    SKEL_PREFERRED,
    SKEL_DIR / "skel3d.json",
    SKEL_DIR / "skel3d_stride2.json",
)
# 空壳 skel（三角化 0 人）约几百 KB；完整片通常 >5MB
MIN_SKEL_BYTES = 5_000_000
PORT = 8767


def _default_skel() -> Path | None:
    sized = [p for p in SKEL_FALLBACKS if p.is_file() and p.stat().st_size >= MIN_SKEL_BYTES]
    if sized:
        return sized[0]
    for p in SKEL_FALLBACKS:
        if p.is_file():
            return p
    return None
_VENDOR_TYPES = {".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8"}


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _load() -> dict:
    return _read_calib(CALIB)


def _read_calib(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if path != CALIB:
        data["calib_path"] = str(path.relative_to(ROOT))
        return data
    spread = _wall_x_spread((data.get("solved") or {}).get("walls") or [])
    fixed = stamp_144_24(_fix_solved_in_payload(data))
    if spread >= 0.02 and _wall_x_spread((fixed.get("solved") or {}).get("walls") or []) < 0.02:
        CALIB.write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")
    fixed["calib_path"] = str(CALIB.relative_to(ROOT))
    return fixed


def _video_sources_from_query(query: str) -> dict | None:
    qs = parse_qs(query)
    if not qs:
        return None
    return {
        "mode": (qs.get("mode") or ["stitched"])[0],
        "src": (qs.get("src") or [None])[0],
        "L": (qs.get("L") or [None])[0],
        "R": (qs.get("R") or [None])[0],
    }


def _view_list(body: dict) -> list:
    views = body.get("views") or {}
    if isinstance(views, dict) and "L" in views and "R" in views:
        return [views["L"], views["R"]]
    return list(views) if isinstance(views, list) else []


def _fix_solved_in_payload(body: dict) -> dict:
    sol = body.get("solved")
    if not sol or not sol.get("ok"):
        return body
    views = _view_list(body)
    if len(views) != 2:
        return body
    spread = _wall_x_spread(sol.get("walls") or [])
    if spread < 0.02:
        return body
    out = dict(body)
    out["solved"] = snap_solved_walls_parametric(
        sol, views, float(body.get("aisle") or 2.0)
    )
    return out


def _drop_absent_wall_meshes(payload: dict) -> dict:
    """单面墙或反解里没有的墙，不保留层线网格，避免画面还画出墙2。"""
    meshes = payload.get("slot_meshes") or []
    if not meshes:
        return payload
    keep: set[int] | None = None
    if payload.get("single_wall") and payload.get("active_wall_id") is not None:
        keep = {int(payload["active_wall_id"])}
    else:
        walls = (payload.get("solved") or {}).get("walls") if isinstance(payload.get("solved"), dict) else None
        if walls:
            keep = {int(w["wall_id"]) for w in walls if w.get("wall_id") is not None}
    if keep is None:
        return payload
    payload["slot_meshes"] = [m for m in meshes if int(m.get("wall_id", -1)) in keep]
    return payload


def _save(data: dict) -> Path:
    path = calib_file(data.get("video_sources"))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _drop_absent_wall_meshes(_fix_solved_in_payload(data))
    if path == CALIB:
        payload = stamp_144_24(payload)
    else:
        payload = dict(payload)
        payload["calib_id"] = path.stem
    payload["calib_path"] = str(path.relative_to(ROOT))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _skel_path(query: str) -> Path | None:
    """只允许 output/dualcam/ 下 skel3d*.json。带视频参数时按视频名找，避免回放串到 144/24。"""
    qs = parse_qs(query)
    name = (qs.get("skel") or [None])[0]
    if not name and (qs.get("src") or qs.get("L") or qs.get("R")):
        return skel_file(_video_sources_from_query(query))
    if not name:
        return _default_skel()
    base = Path(name).name
    if not base.startswith("skel3d") or not base.endswith(".json"):
        return None
    cand = (SKEL_DIR / base).resolve()
    try:
        cand.relative_to(SKEL_DIR.resolve())
    except ValueError:
        return None
    return cand


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_HEAD(self) -> None:
        self.do_GET(head_only=True)

    def do_GET(self, head_only: bool = False) -> None:
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            raw = PAGE.read_bytes()
            self._bytes(raw, "text/html; charset=utf-8", head_only)
            return
        if u.path in ("/play", "/player.html"):
            raw = PLAYER.read_bytes()
            self._bytes(raw, "text/html; charset=utf-8", head_only)
            return
        if u.path == "/dualcam_geom.js":
            raw = GEOM.read_bytes()
            self._bytes(raw, "text/javascript; charset=utf-8", head_only)
            return
        if u.path.startswith("/vendor/"):
            rel = Path(u.path[len("/vendor/") :])
            if rel.is_absolute() or ".." in rel.parts:
                self.send_error(404)
                return
            path = (VENDOR / rel).resolve()
            try:
                path.relative_to(VENDOR.resolve())
            except ValueError:
                self.send_error(404)
                return
            if not path.is_file():
                self.send_error(404)
                return
            ctype = _VENDOR_TYPES.get(path.suffix, "application/octet-stream")
            self._stream_file(path, ctype, head_only, cache="public, max-age=86400")
            return
        if u.path == "/api/skel3d":
            path = _skel_path(u.query)
            if path is None or not path.is_file():
                self.send_error(404, "skel3d missing")
                return
            self._stream_file(path, "application/json; charset=utf-8", head_only)
            return
        if u.path == "/api/calib":
            vs = _video_sources_from_query(u.query)
            if vs is None:
                self._json(_load(), head_only)
                return
            path = calib_file(vs)
            data = _read_calib(path)
            data["calib_path"] = str(path.relative_to(ROOT))
            self._json(data, head_only)
            return
        if u.path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if u.path == "/api/videos":
            self._json({"videos": list_videos()}, head_only)
            return
        if u.path == "/api/video":
            src = (parse_qs(u.query).get("src") or [None])[0]
            path = resolve_video(src)
            if path is None:
                self.send_error(404, "video missing")
                return
            self._file_range(path, "video/mp4", head_only)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if u.path == "/api/calib":
            path = _save(body)
            self._json({"ok": True, "path": str(path.relative_to(ROOT))})
            return
        if u.path == "/api/solve":
            views = body.get("views") or {}
            payload = {
                "aisle": body.get("aisle"),
                "prior": body.get("prior"),
                "layout": body.get("layout") or body.get("stereo_layout") or "same_side",
                "views": [views["L"], views["R"]] if isinstance(views, dict) and "L" in views else views,
            }
            res = solve_dual(payload)
            if res.get("ok"):
                views = payload["views"]
                res = snap_solved_walls_parametric(
                    res, views, float(payload.get("aisle") or 2.0)
                )
                res["wall_reproj_px"] = _wall_reproj_px(res["walls"], views, res["cameras"])
                saved = dict(body)
                saved["solved"] = res
                path = _save(saved)
                res = dict(res)
                res["path"] = str(path.relative_to(ROOT))
            self._json(res)
            return
        self.send_error(404)

    def _json(self, obj: dict, head_only: bool = False) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._bytes(raw, "application/json; charset=utf-8", head_only)

    def _bytes(self, raw: bytes, content_type: str, head_only: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        if not head_only:
            self.wfile.write(raw)

    def _stream_file(
        self,
        path: Path,
        content_type: str,
        head_only: bool = False,
        cache: str = "no-store",
    ) -> None:
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(size))
        self.end_headers()
        if head_only:
            return
        try:
            with path.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _file_range(self, path: Path, content_type: str, head_only: bool = False) -> None:
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        rng = None if head_only else self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng.split("=", 1)[1].split(",")[0]
            a, _, b = spec.partition("-")
            if a:
                start = int(a)
            if b:
                end = int(b)
            end = min(end, size - 1)
            start = max(0, start)
            code = 206
        length = max(0, end - start + 1)
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "public, max-age=86400")
        self.send_header("Content-Length", str(length))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return
        try:
            with path.open("rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    chunk = f.read(min(1024 * 1024, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> int:
    vids = list_videos()
    if not vids:
        print(f"找不到 mp4：{VIDEO.parent} 或 data/", file=sys.stderr)
        print("  .venv/bin/python scripts/stitch_dualcam_src.py", file=sys.stderr)
        return 1
    note = migrate_misplaced_144_24()
    if note:
        print(f"migrate        → {note}", file=sys.stderr)
    ip = _lan_ip()
    print(f"dualcam annot  → http://{ip}:{PORT}/   videos={len(vids)}")
    print(f"calib_id       → {CALIB_ID}")
    print(f"dualcam 3D回放 → http://{ip}:{PORT}/play")
    print(f"               → http://127.0.0.1:{PORT}/play")
    print(f"calib          → {CALIB.relative_to(ROOT)}")
    sk = _default_skel()
    if sk:
        print(f"skel3d         → {sk.relative_to(ROOT)}")
    else:
        print("skel3d         → 缺失（运行 scripts/dump_skel3d.py）", file=sys.stderr)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
