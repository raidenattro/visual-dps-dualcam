"""Per-camera 2D 标注 JSON 路径：文件名使用 camera.path（如 aisle1-L、cam1）。"""

from __future__ import annotations

import json
import os
import re
from typing import Any

_PATH_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

_index_cache: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}


def _load_by_id(camera_ips_file: str) -> dict[str, dict[str, Any]]:
    path = str(camera_ips_file or "").strip()
    if not path or not os.path.isfile(path):
        return {}
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    cached = _index_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        raw = []
    by_id: dict[str, dict[str, Any]] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id") or "").strip()
            if cid:
                by_id[cid] = item
    _index_cache[path] = (mtime, by_id)
    return by_id


def camera_annotation_slug(
    camera_id: str,
    camera: dict | None = None,
    camera_ips_file: str | None = None,
) -> str:
    """标注文件名（无 .json）：优先 channel path，否则 camera_id。"""
    cid = str(camera_id or "").strip()
    if isinstance(camera, dict):
        channel = str(camera.get("path") or "").strip()
        if channel and _PATH_RE.match(channel):
            return channel
    if camera_ips_file:
        rec = _load_by_id(camera_ips_file).get(cid)
        if isinstance(rec, dict):
            channel = str(rec.get("path") or "").strip()
            if channel and _PATH_RE.match(channel):
                return channel
    return cid


def camera_annotation_path(
    json_dir: str,
    camera_id: str,
    camera: dict | None = None,
    camera_ips_file: str | None = None,
) -> str:
    slug = camera_annotation_slug(camera_id, camera, camera_ips_file)
    if not slug:
        return ""
    return os.path.join(json_dir, "cameras", f"{slug}.json")


def legacy_id_annotation_path(json_dir: str, camera_id: str) -> str:
    """旧命名 cameras/{id}.json（仅迁移/读回退）。"""
    cid = str(camera_id or "").strip()
    if not cid:
        return ""
    return os.path.join(json_dir, "cameras", f"{cid}.json")


def existing_camera_annotation_path(
    json_dir: str,
    camera_id: str,
    camera: dict | None = None,
    camera_ips_file: str | None = None,
) -> str:
    """解析磁盘上实际存在的标注路径（新 slug 优先，否则旧 id 文件）。"""
    canonical = camera_annotation_path(json_dir, camera_id, camera, camera_ips_file)
    if canonical and os.path.isfile(canonical):
        return canonical
    legacy = legacy_id_annotation_path(json_dir, camera_id)
    if legacy and os.path.isfile(legacy):
        return legacy
    return canonical or legacy
