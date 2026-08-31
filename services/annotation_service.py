"""标注 JSON 读写服务。"""

import json
import os
from datetime import datetime

from core.state import STATE


def camera_annotation_path(json_dir: str, camera_id: str) -> str:
    cid = str(camera_id or "").strip()
    if not cid:
        return ""
    return os.path.join(json_dir, "cameras", f"{cid}.json")


def _read_annotation_file(json_path: str):
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _source_info_matches_camera(source_info: dict, camera: dict, camera_id: str) -> bool:
    if not isinstance(source_info, dict) or not isinstance(camera, dict):
        return False
    cam_url = str(camera.get("url") or "").strip()
    cam_name = str(camera.get("name") or "").strip()
    cid = str(camera_id or "").strip()
    src_url = str(source_info.get("camera_url") or "").strip()
    src_name = str(source_info.get("camera_name") or "").strip()
    if src_url and cam_url and src_url == cam_url:
        return True
    if src_name and cam_name and src_name == cam_name:
        return True
    if src_name and cid and src_name == cid:
        return True
    return False


def _find_latest_annotation_for_camera(json_dir: str, camera: dict, camera_id: str) -> str:
    if not os.path.isdir(json_dir):
        return ""
    best_path = ""
    best_mtime = 0.0
    for name in os.listdir(json_dir):
        if not name.endswith(".json") or name == "STATE.json":
            continue
        path = os.path.join(json_dir, name)
        if not os.path.isfile(path):
            continue
        data = _read_annotation_file(path)
        if not isinstance(data, dict):
            continue
        src = data.get("source_info")
        if not _source_info_matches_camera(src if isinstance(src, dict) else {}, camera, camera_id):
            continue
        mtime = os.path.getmtime(path)
        if mtime >= best_mtime:
            best_mtime = mtime
            best_path = path
    return best_path


def _empty_annotation_template(camera: dict, camera_id: str) -> dict:
    cam = camera if isinstance(camera, dict) else {}
    return {
        "annotation_size": {"width": 0, "height": 0},
        "source_info": {
            "capture_source": "camera",
            "camera_url": str(cam.get("url") or "").strip(),
            "camera_name": str(cam.get("name") or camera_id or "").strip(),
        },
        "shelves": [],
    }


def materialize_camera_annotation(
    camera_id: str,
    json_dir: str,
    camera: dict | None = None,
) -> str:
    """为摄像头创建空的 per-camera 标注文件（若尚不存在）。"""
    cid = str(camera_id or "").strip()
    if not cid:
        return ""
    stable_path = camera_annotation_path(json_dir, cid)
    if os.path.isfile(stable_path):
        return stable_path
    cam = camera if isinstance(camera, dict) else {}
    save_camera_annotation(_empty_annotation_template(cam, cid), cid, json_dir)
    return stable_path


def ensure_camera_annotation_file(
    camera_id: str,
    json_dir: str,
    default_json_file: str,
    camera: dict | None = None,
) -> str:
    """返回推理应使用的标注相对路径；必要时从 legacy 文件物化到 cameras/{id}.json。"""
    cid = str(camera_id or "").strip()
    if not cid:
        return default_json_file

    stable_path = camera_annotation_path(json_dir, cid)
    if os.path.isfile(stable_path):
        return stable_path

    cam = camera if isinstance(camera, dict) else {}
    legacy_path = _find_latest_annotation_for_camera(json_dir, cam, cid)
    if legacy_path:
        data = _read_annotation_file(legacy_path)
        if isinstance(data, dict):
            save_camera_annotation(data, cid, json_dir)
            return stable_path

    return materialize_camera_annotation(cid, json_dir, cam)


def save_camera_annotation(data: dict, camera_id: str, json_dir: str) -> dict:
    cid = str(camera_id or "").strip()
    if not cid:
        return {"status": "error", "error": "camera_id is required"}
    json_path = camera_annotation_path(json_dir, cid)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    STATE.json_path = json_path
    return {"status": "success", "json_path": json_path, "camera_id": cid}


def load_camera_annotation(camera_id: str, json_dir: str, default_json_file: str, camera: dict | None = None):
    cid = str(camera_id or "").strip()
    if not cid:
        return {"error": "camera_id is required"}

    stable_path = camera_annotation_path(json_dir, cid)
    if os.path.isfile(stable_path):
        data = _read_annotation_file(stable_path)
        if isinstance(data, dict):
            return {"status": "success", "data": data, "json_path": stable_path, "camera_id": cid}

    cam = camera if isinstance(camera, dict) else {}
    legacy_path = _find_latest_annotation_for_camera(json_dir, cam, cid)
    if legacy_path:
        data = _read_annotation_file(legacy_path)
        if isinstance(data, dict):
            return {"status": "success", "data": data, "json_path": legacy_path, "camera_id": cid}

    return {
        "error": "annotation not found",
        "json_path": stable_path,
        "camera_id": cid,
    }


def annotation_payload_for_api(load_result: dict) -> dict:
    if load_result.get("status") != "success":
        return {
            "status": "error",
            "message": load_result.get("error", "annotation not found"),
            "json_path": load_result.get("json_path"),
        }
    config_data = load_result["data"]
    source_info = config_data.get("source_info", {}) if isinstance(config_data, dict) else {}
    if not isinstance(source_info, dict):
        source_info = {}
    return {
        "status": "success",
        "json_path": load_result.get("json_path"),
        "data": config_data,
        "boxes": flatten_annotation_boxes(config_data),
        "shelves": config_data.get("shelves", []) if isinstance(config_data, dict) else [],
        "grid_shape": config_data.get("grid_shape", []) if isinstance(config_data, dict) else [],
        "shelf_corners": config_data.get("shelf_corners", []) if isinstance(config_data, dict) else [],
        "annotation_size": config_data.get("annotation_size") if isinstance(config_data, dict) else None,
        "source_info": source_info,
    }


def resolve_annotation_path(json_dir: str, default_json_file: str) -> str:
    if STATE.json_path:
        return STATE.json_path
    return default_json_file


def save_annotation(data: dict, default_json_file: str, json_dir: str):
    os.makedirs(json_dir, exist_ok=True)

    if STATE.json_path:
        json_path = STATE.json_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(json_dir, f"annotation_{timestamp}.json")
        STATE.json_path = json_path

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {"status": "success", "json_path": json_path}


def load_annotation(default_json_file: str, json_dir: str):
    json_path = resolve_annotation_path(json_dir, default_json_file)
    if not os.path.exists(json_path):
        return {"error": "annotation not found", "json_path": json_path}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {"error": f"invalid annotation json: {e}", "json_path": json_path}

    return {"status": "success", "data": data, "json_path": json_path}


def flatten_annotation_boxes(config_data: dict) -> list:
    if not isinstance(config_data, dict):
        return []

    raw_boxes = []
    shelves = config_data.get("shelves")
    if isinstance(shelves, list):
        for shelf in shelves:
            if not isinstance(shelf, dict):
                continue
            shelf_code = str(shelf.get("shelf_code", "") or "").strip()
            boxes = shelf.get("boxes", [])
            if not isinstance(boxes, list):
                continue
            for box in boxes:
                if not isinstance(box, dict):
                    continue
                item = dict(box)
                if shelf_code and not item.get("shelf_code"):
                    item["shelf_code"] = shelf_code
                raw_boxes.append(item)

    if raw_boxes:
        return raw_boxes

    boxes = config_data.get("boxes", [])
    return boxes if isinstance(boxes, list) else []
