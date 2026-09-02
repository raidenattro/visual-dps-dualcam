"""摄像头配置持久化与 CRUD。"""

import json
import os
import re
from typing import List
from urllib.parse import urlparse

from services.runtime_config_service import normalize_camera_settings
from services.mediamtx_service import (
    SOURCE_EXTERNAL,
    SOURCE_PUBLISHER,
    SOURCE_RTSP_PULL,
    build_playback_url,
    path_from_url,
    sync_mediamtx_config,
)

_PATH_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_stream_url(url: str) -> str | None:
    parsed = urlparse(str(url or "").strip())
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("rtsp", "rtsps", "http", "https"):
        return "视频流地址需以 rtsp://、rtsps://、http:// 或 https:// 开头"
    host = (parsed.hostname or "").strip()
    if not host:
        return "视频流地址无效：缺少主机名"
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
        try:
            octets = [int(x) for x in host.split(".")]
        except ValueError:
            return "IP 地址格式无效"
        if len(octets) != 4 or any(o < 0 or o > 255 for o in octets):
            return "IP 地址无效（每段应为 0–255）"
    return None


def _normalize_record(raw: dict) -> dict | None:
    if not isinstance(raw, dict):
        return None

    source_type = str(raw.get("source_type") or SOURCE_RTSP_PULL).strip()
    if source_type == "v4l2":
        source_type = SOURCE_PUBLISHER
    path = str(raw.get("path") or "").strip()
    cid = str(raw.get("id") or "").strip()
    name = str(raw.get("name") or "").strip()
    url = str(raw.get("url") or "").strip()

    if not path:
        path = cid
    if not path and url:
        path = path_from_url(url)
    if not path and source_type != SOURCE_EXTERNAL:
        return None
    if not path:
        return None

    if not _PATH_RE.match(path):
        return None

    if not name:
        name = path

    if source_type == SOURCE_EXTERNAL:
        if not url:
            return None
    else:
        if not url:
            url = build_playback_url(path)

    # 旧数据没有独立 id 时，id 仍等于 path；新数据 id 为自增数字
    if not cid:
        cid = path

    record = {
        "id": cid,
        "name": name,
        "path": path,
        "url": url,
        "source_type": source_type,
        "enabled": bool(raw.get("enabled", True)),
        "pull_url": str(raw.get("pull_url") or "").strip(),
    }
    if "settings" in raw:
        settings = normalize_camera_settings(raw.get("settings"))
        if settings:
            record["settings"] = settings
    return record


def _legacy_to_record(item: dict) -> dict | None:
    url = str(item.get("url", "")).strip()
    if not url:
        return None
    name = str(item.get("name", "")).strip() or url
    path = path_from_url(url) or "cam"
    return _normalize_record(
        {
            "name": name,
            "path": path,
            "url": url,
            "source_type": SOURCE_PUBLISHER if path and "127.0.0.1" in url else SOURCE_EXTERNAL,
        }
    )


def load_cameras(camera_file: str) -> List[dict]:
    if not os.path.exists(camera_file):
        return []

    try:
        data = json.loads(open(camera_file, "r", encoding="utf-8").read())
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    items = []
    seen = set()
    for raw in data:
        rec = (
            _normalize_record(raw)
            if (raw.get("source_type") or raw.get("path") or raw.get("id"))
            else _legacy_to_record(raw)
        )
        if not rec or rec["id"] in seen:
            continue
        seen.add(rec["id"])
        items.append(rec)
    return items


def save_cameras(camera_file: str, items: List[dict]):
    os.makedirs(os.path.dirname(camera_file) or ".", exist_ok=True)
    with open(camera_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _next_camera_id(items: List[dict]) -> str:
    """自增数字 id。旧记录若仍用 path 当 id，不影响计数。"""
    max_n = 0
    for cam in items:
        cid = str(cam.get("id") or "").strip()
        if cid.isdigit():
            max_n = max(max_n, int(cid))
    return str(max_n + 1)


def _path_taken(items: List[dict], path: str, *, except_id: str | None = None) -> bool:
    p = str(path or "").strip()
    if not p:
        return False
    for cam in items:
        if str(cam.get("path") or "").strip() != p:
            continue
        if except_id is not None and str(cam.get("id") or "") == str(except_id):
            continue
        return True
    return False


def _is_auto_playback_url(url: str, path: str) -> bool:
    cur = str(url or "").strip()
    if not cur:
        return True
    return cur == build_playback_url(path)


def validate_camera_payload(data: dict, existing_id: str | None = None) -> tuple[dict | None, str | None]:
    path = str(data.get("path") or "").strip()
    if not path:
        path = str(data.get("id") or existing_id or "").strip()

    source_type = str(data.get("source_type") or SOURCE_RTSP_PULL).strip()
    if source_type == "v4l2":
        return None, "不再支持本地摄像头，请改为「外部推流」或「拉取外部流」"
    name = str(data.get("name") or "").strip()
    url = str(data.get("url") or "").strip()
    pull_url = str(data.get("pull_url") or "").strip()

    if not path:
        return None, "通道编号不能为空"
    if not _PATH_RE.match(path):
        return None, "通道编号仅支持字母、数字、下划线、中划线（1–64 个字符）；若以数字开头（如 71），写入 mediamtx.yml 时会自动加双引号"
    if not name:
        return None, "名称不能为空"

    if source_type == SOURCE_EXTERNAL:
        if not url:
            return None, "请填写完整的视频流地址"
        url_err = _validate_stream_url(url)
        if url_err:
            return None, url_err
    elif source_type == SOURCE_RTSP_PULL:
        if not pull_url:
            return None, "请填写上游视频流地址"
        if not url:
            url = build_playback_url(path)
    elif source_type == SOURCE_PUBLISHER:
        if not url:
            url = build_playback_url(path)
    else:
        return None, f"不支持的 source_type: {source_type}"

    rec_id = str(existing_id or data.get("id") or path).strip()
    raw_rec = {
        "id": rec_id,
        "path": path,
        "name": name,
        "url": url,
        "source_type": source_type,
        "enabled": data.get("enabled", True),
        "pull_url": pull_url,
    }
    if "settings" in data:
        try:
            raw_rec["settings"] = normalize_camera_settings(data.get("settings"), strict=True)
        except ValueError as exc:
            return None, str(exc)
    rec = _normalize_record(raw_rec)
    if not rec:
        return None, "配置无效"
    return rec, None


def create_camera(camera_file: str, mediamtx_config_path: str, data: dict) -> dict:
    rec, err = validate_camera_payload(data)
    if err:
        return {"error": err}

    items = load_cameras(camera_file)
    if _path_taken(items, rec["path"]):
        return {"error": f"通道号已被使用: {rec['path']}"}

    rec["id"] = _next_camera_id(items)
    items.append(rec)
    save_cameras(camera_file, items)
    mtx = sync_mediamtx_config(mediamtx_config_path, items)
    return {"status": "success", "camera": rec, "items": items, "mediamtx": mtx}


def update_camera(camera_file: str, mediamtx_config_path: str, camera_id: str, data: dict) -> dict:
    items = load_cameras(camera_file)
    idx = next((i for i, c in enumerate(items) if c["id"] == camera_id), -1)
    if idx < 0:
        return {"error": "未找到该摄像头"}

    old = items[idx]
    old_path = str(old.get("path") or camera_id).strip()
    new_path = str((data or {}).get("path") or old_path).strip()
    if _path_taken(items, new_path, except_id=camera_id):
        return {"error": f"通道号已被使用: {new_path}"}

    merged = {**old, **(data or {}), "id": camera_id, "path": new_path}
    if _is_auto_playback_url(str(old.get("url") or ""), old_path):
        incoming_url = str((data or {}).get("url") or "").strip()
        if not incoming_url or incoming_url == build_playback_url(old_path):
            merged["url"] = build_playback_url(new_path)

    rec, err = validate_camera_payload(merged, existing_id=camera_id)
    if err:
        return {"error": err}

    items[idx] = rec
    save_cameras(camera_file, items)
    mtx = sync_mediamtx_config(mediamtx_config_path, items)
    return {"status": "success", "camera": rec, "items": items, "mediamtx": mtx}


def delete_camera(
    camera_file: str,
    mediamtx_config_path: str,
    camera_id: str,
    json_dir: str | None = None,
) -> dict:
    from services.inference_container_service import stop_inference_container

    items = load_cameras(camera_file)
    new_items = [c for c in items if c["id"] != camera_id]
    if len(new_items) == len(items):
        return {"error": "未找到该摄像头"}

    stop_inference_container(camera_id)
    save_cameras(camera_file, new_items)
    mtx = sync_mediamtx_config(mediamtx_config_path, new_items)
    if json_dir:
        from services.annotation_service import delete_camera_annotation

        delete_camera_annotation(camera_id, json_dir)
    return {"status": "success", "items": new_items, "mediamtx": mtx}


def get_camera(camera_file: str, camera_id: str) -> dict:
    items = load_cameras(camera_file)
    for c in items:
        if c["id"] == camera_id:
            return {"status": "success", "camera": c}
    return {"error": "未找到该摄像头"}


def apply_mediamtx(camera_file: str, mediamtx_config_path: str) -> dict:
    items = load_cameras(camera_file)
    mtx = sync_mediamtx_config(mediamtx_config_path, items)
    return {"status": "success", "mediamtx": mtx, "items": items}


# 兼容旧 API：仅 name + url
def load_camera_ips(camera_ips_file: str) -> List[dict]:
    return [{"name": c["name"], "url": c["url"]} for c in load_cameras(camera_ips_file)]


def save_camera_ips(camera_ips_file: str, items: List[dict]):
    existing = {c["url"]: c for c in load_cameras(camera_ips_file)}
    merged = []
    for item in items:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        name = str(item.get("name", "")).strip() or url
        if url in existing:
            rec = {**existing[url], "name": name}
        else:
            rec = _legacy_to_record({"name": name, "url": url})
        if rec:
            merged.append(rec)
    save_cameras(camera_ips_file, merged)
