#!/usr/bin/env python3
"""巷道 3D 查看器：段/机位可选，格子用标注多边形，视频从 collector 只读流出。"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.collector_paths import load_paths
from adapters.record_reader import (
    list_records_from_manifest,
    load_record,
    load_segment_split,
    record_dir,
)
from scripts.render_fn_fp_frames import _resolve_video
from scripts.solve_scene import solve as solve_scene

MANIFEST = ROOT / "output/manifests/tagged_aug85_v4.json"
VIEWER = ROOT / "scripts" / "aisle3d_viewer.html"
CALIB_DIR = ROOT / "output/calib"
DEFAULTS = {
    "camH": 3.0,
    "camDist": 1.5,
    "pitch": 45.0,
    "yaw": 0.0,
    "fovH": 90.0,
    "aisle": 1.7,
    "boxW": 0.42,
    "boxH": 0.36,
    "boxD": 0.48,
    "baseY": 0.12,
    "kptMin": 0.25,
}
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]


def _wall_id(box_id: str) -> int:
    try:
        return int(box_id) // 1000
    except ValueError:
        return 0


@lru_cache(maxsize=1)
def _index() -> dict:
    refs = list_records_from_manifest(MANIFEST)
    segs = load_segment_split(MANIFEST)
    records = []
    for ref in refs:
        items = segs.get(ref.record_id) or []
        records.append(
            {
                "record_id": ref.record_id,
                "camera_slug": ref.camera_slug,
                "clip_name": ref.clip_name,
                "n_segments": len(items),
                "segments": [
                    {
                        "seg_id": str(s.get("seg_id") or f"{s.get('frame_start')}-{s.get('frame_end')}"),
                        "split": s.get("split"),
                        "frame_start": int(s.get("frame_start") or 0),
                        "frame_end": int(s.get("frame_end") or 0),
                        "gt_tokens": list(s.get("gt_tokens") or []),
                        "tracks": [int(x) for x in (s.get("person_track_ids") or []) if str(x).isdigit()],
                    }
                    for s in items
                ],
            }
        )
    return {"records": records}


def _assign_signs(boxes_out: list[dict]) -> None:
    groups: dict[int, list[dict]] = {}
    for b in boxes_out:
        groups.setdefault(int(b["wall"]), []).append(b)
    ranked = sorted(
        groups.items(),
        key=lambda kv: sum(x["center2d"][0] for x in kv[1]) / max(len(kv[1]), 1),
    )
    if len(ranked) == 1:
        for b in ranked[0][1]:
            b["sign"] = -1 if b["center2d"][0] < 400 else 1
        return
    for i, (_, grp) in enumerate(ranked):
        sign = -1 if i == 0 else 1
        for b in grp:
            b["sign"] = sign


@lru_cache(maxsize=8)
def _scene(record_id: str) -> dict:
    refs = {r.record_id: r for r in list_records_from_manifest(MANIFEST)}
    if record_id not in refs:
        raise KeyError(record_id)
    rec = load_record(refs[record_id])
    segs = [s for s in (_index()["records"]) if s["record_id"] == record_id]
    segments = segs[0]["segments"] if segs else []
    boxes_out = []
    for b in rec.boxes:
        pts = b.contour.reshape(-1, 2).astype(float).tolist()
        boxes_out.append(
            {
                "token": b.token,
                "wall": _wall_id(b.box_id),
                "layer": int(b.layer or 0),
                "column": int(b.column or 0),
                "center2d": [round(float(b.center[0]), 2), round(float(b.center[1]), 2)],
                "poly2d": [[round(float(x), 2), round(float(y), 2)] for x, y in pts],
            }
        )
    _assign_signs(boxes_out)
    frames_out = []
    for fr in rec.frames:
        people = []
        for p in fr.get("persons") or []:
            kps = p.get("keypoints") or []
            if len(kps) < 17:
                continue
            people.append(
                {
                    "track": int(p.get("person_track_id") or 0),
                    "kpts": [[round(float(a), 2) for a in kp[:3]] for kp in kps[:17]],
                }
            )
        if people:
            frames_out.append(
                {
                    "frame": int(fr["frame_idx"]),
                    "t": round(float(fr.get("timestamp_sec") or 0.0), 4),
                    "people": people,
                }
            )
    return {
        "record_id": rec.ref.record_id,
        "clip_name": rec.ref.clip_name,
        "camera_slug": rec.ref.camera_slug,
        "image_size": [
            int(rec.meta.get("infer_width") or 852),
            int(rec.meta.get("infer_height") or 480),
        ],
        "fps": float(rec.fps or 25.0),
        "defaults": DEFAULTS,
        "edges": EDGES,
        "boxes": boxes_out,
        "frames": frames_out,
        "segments": segments,
        "has_video": _video_path(record_id) is not None,
    }


def _calib_path(camera_slug: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_.()" else "_" for c in camera_slug)
    return CALIB_DIR / f"{safe}.json"


def _load_calib(camera_slug: str) -> dict:
    p = _calib_path(camera_slug)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_calib(camera_slug: str, data: dict) -> None:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    _calib_path(camera_slug).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _video_path(record_id: str) -> Path | None:
    refs = {r.record_id: r for r in list_records_from_manifest(MANIFEST)}
    ref = refs.get(record_id)
    if ref is None:
        return None
    rec_man = json.loads((record_dir(ref) / "manifest.json").read_text(encoding="utf-8"))
    p = _resolve_video(record_id, rec_man, load_paths())
    return p if p and p.is_file() else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                raw = VIEWER.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if u.path == "/api/index.json":
                self._json(_index())
                return
            if u.path == "/api/scene.json":
                rid = unquote(q.get("record_id") or "")
                self._json(_scene(rid))
                return
            if u.path == "/api/calib":
                self._json(_load_calib(unquote(q.get("camera_slug") or "")))
                return
            if u.path == "/api/video":
                rid = unquote(q.get("record_id") or "")
                path = _video_path(rid)
                if path is None:
                    self.send_error(404, "no video")
                    return
                self._file_range(path, "video/mp4")
                return
            self.send_error(404)
        except KeyError:
            self.send_error(404, "unknown record")
        except Exception as e:
            self.send_error(500, str(e))

    def do_POST(self) -> None:
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        slug = unquote(q.get("camera_slug") or "")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if not slug:
                self.send_error(400, "camera_slug required")
                return
            if u.path == "/api/calib":
                _save_calib(slug, body)
                self._json({"ok": True, "path": str(_calib_path(slug).relative_to(ROOT))})
                return
            if u.path == "/api/solve":
                size = body.get("image_size") or [852, 480]
                res = solve_scene(body, int(size[0]), int(size[1]))
                if res.get("ok"):
                    saved = dict(body)
                    saved["solved"] = res
                    _save_calib(slug, saved)
                self._json(res)
                return
            self.send_error(404)
        except Exception as e:
            self.send_error(500, str(e))

    def _json(self, obj: dict) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _file_range(self, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range")
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
        self.send_header("Content-Length", str(length))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(256 * 1024, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def main() -> int:
    host, port = "0.0.0.0", 8765
    print(f"aisle3d → http://192.168.1.153:{port}/  (index from {MANIFEST})")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
