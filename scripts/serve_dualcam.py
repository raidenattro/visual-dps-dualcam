#!/usr/bin/env python3
"""双路拼接监控的拣货面标注服务。视频只读，标定写本仓 output/calib。"""

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

from scripts.solve_scene import solve_dual

PAGE = ROOT / "scripts" / "dualcam_annot.html"
PLAYER = ROOT / "scripts" / "dualcam_player.html"
GEOM = ROOT / "scripts" / "dualcam_geom.js"
VENDOR = ROOT / "scripts" / "vendor"
VIDEO = ROOT / "output" / "dualcam" / "src.mp4"
CALIB = ROOT / "output" / "calib" / "dual_1-3.json"
SKEL = ROOT / "output" / "dualcam" / "skel3d.json"
PORT = 8767
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
    if not CALIB.is_file():
        return {}
    try:
        return json.loads(CALIB.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save(data: dict) -> None:
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    CALIB.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _skel_path(query: str) -> Path | None:
    """只允许 output/dualcam/ 下 skel3d*.json；无参数则默认 skel3d.json。"""
    name = (parse_qs(query).get("skel") or [None])[0]
    if not name:
        return SKEL
    base = Path(name).name
    if not base.startswith("skel3d") or not base.endswith(".json"):
        return None
    cand = (SKEL.parent / base).resolve()
    try:
        cand.relative_to(SKEL.parent.resolve())
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
            self._json(_load(), head_only)
            return
        if u.path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if u.path == "/api/video":
            if not VIDEO.is_file():
                self.send_error(404, "video missing")
                return
            self._file_range(VIDEO, "video/mp4", head_only)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if u.path == "/api/calib":
            _save(body)
            self._json({"ok": True, "path": str(CALIB.relative_to(ROOT))})
            return
        if u.path == "/api/solve":
            views = body.get("views") or {}
            payload = {
                "aisle": body.get("aisle"),
                "prior": body.get("prior"),
                "views": [views["L"], views["R"]] if isinstance(views, dict) and "L" in views else views,
            }
            res = solve_dual(payload)
            if res.get("ok"):
                saved = dict(body)
                saved["solved"] = res
                _save(saved)
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
    if not VIDEO.is_file():
        print(f"找不到视频：{VIDEO}", file=sys.stderr)
        return 1
    ip = _lan_ip()
    print(f"dualcam annot  → http://{ip}:{PORT}/")
    print(f"dualcam 3D回放 → http://{ip}:{PORT}/play")
    print(f"               → http://127.0.0.1:{PORT}/play")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
