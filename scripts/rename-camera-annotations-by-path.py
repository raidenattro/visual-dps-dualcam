#!/usr/bin/env python3
"""将 json/cameras/{id}.json 重命名为 json/cameras/{path}.json（与 camera_ips 一致）。"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.config import load_app_config
from services.camera_annotation_paths import camera_annotation_path, legacy_id_annotation_path


def main() -> int:
    app = load_app_config()
    paths = app.get("paths") or {}
    json_dir = str(paths.get("json_dir") or "localdata/json")
    cam_file = str(paths.get("camera_ips_file") or "localdata/camera_ips.json")
    with open(cam_file, "r", encoding="utf-8") as f:
        cameras = json.load(f)
    if not isinstance(cameras, list):
        print("camera_ips 格式无效", file=sys.stderr)
        return 1
    moved = 0
    for cam in cameras:
        if not isinstance(cam, dict):
            continue
        cid = str(cam.get("id") or "").strip()
        if not cid:
            continue
        src = legacy_id_annotation_path(json_dir, cid)
        dst = camera_annotation_path(json_dir, cid, camera=cam, camera_ips_file=cam_file)
        if not src or not dst or src == dst:
            continue
        if not os.path.isfile(src):
            continue
        if os.path.isfile(dst):
            os.remove(src)
            print(f"removed duplicate id file: {os.path.basename(src)} (已有 {os.path.basename(dst)})")
            moved += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)
        print(f"{os.path.basename(src)} -> {os.path.basename(dst)}")
        moved += 1
    print(f"done, actions={moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
