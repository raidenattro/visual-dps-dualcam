"""巷道组：两路相机绑定 + 标定 JSON。未成组禁止开推理。"""

from __future__ import annotations

import json
import os
import re
from typing import Any

_AISLE_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
ROLES = ("L", "R")


def _json_dir(json_dir: str | None = None) -> str:
    if json_dir:
        return json_dir
    return os.environ.get("JSON_DIR", "").strip() or "localdata/json"


def aisles_dir(json_dir: str | None = None) -> str:
    return os.path.join(_json_dir(json_dir), "aisles")


def aisle_path(aisle_id: str, json_dir: str | None = None) -> str:
    return os.path.join(aisles_dir(json_dir), f"{aisle_id}.json")


def _read(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _write(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def empty_aisle(aisle_id: str) -> dict[str, Any]:
    from services.dualcam_config import aabb_from_section, default_prior, get_dualcam_section

    sec = get_dualcam_section()
    prior = default_prior(sec)
    aabb = aabb_from_section(sec)
    size = [int(sec["calib_width"]), int(sec["calib_height"])]
    return {
        "aisle_id": aisle_id,
        "aisle": 2.0,
        "contact_m": float(sec["contact_m"]),
        "prior": dict(prior),
        "aabb": {k: [v[0], v[1]] for k, v in aabb.items()},
        "cameras": {"L": {"camera_id": "", "role": "L"}, "R": {"camera_id": "", "role": "R"}},
        "views": {
            "L": {
                "name": "L",
                "image_size": list(size),
                "prior": dict(prior),
                "walls": _empty_walls(),
            },
            "R": {
                "name": "R",
                "image_size": list(size),
                "prior": dict(prior),
                "walls": _empty_walls(),
            },
        },
        "slot_meshes": [],
        "required_wall_ids": [1],
        "solved": {"ok": False},
    }


def _empty_walls() -> list[dict]:
    return [
        {"wall_id": 1, "width": 2.2, "height": 2.0, "base": 0.0, "quad": [], "shelf_code": "", "n_layers": 4, "n_cols": 4},
        {"wall_id": 2, "width": 2.2, "height": 2.0, "base": 0.0, "quad": [], "shelf_code": "", "n_layers": 4, "n_cols": 4},
    ]


def parse_wall_ids(raw: Any) -> list[int]:
    out: list[int] = []
    if not isinstance(raw, (list, tuple)):
        return out
    for item in raw:
        try:
            wid = int(item)
        except (TypeError, ValueError):
            continue
        if wid >= 1 and wid not in out:
            out.append(wid)
    return out


def required_wall_ids(aisle: dict | None) -> list[int]:
    """本巷道参与拣货的墙。现场只有一面货架时只填那一面。"""
    data = aisle if isinstance(aisle, dict) else {}
    parsed = parse_wall_ids(data.get("required_wall_ids"))
    if parsed:
        return parsed
    from_mesh: list[int] = []
    for mesh in data.get("slot_meshes") or []:
        if not isinstance(mesh, dict):
            continue
        try:
            wid = int(mesh.get("wall_id"))
        except (TypeError, ValueError):
            continue
        if wid >= 1 and wid not in from_mesh:
            from_mesh.append(wid)
    return from_mesh or [1]


def wall_shelf_code(aisle: dict | None, wall_id: int) -> str:
    data = aisle if isinstance(aisle, dict) else {}
    views = data.get("views") or {}
    for role in ROLES:
        walls = (views.get(role) or {}).get("walls") if isinstance(views, dict) else None
        if not isinstance(walls, list):
            continue
        for wall in walls:
            if not isinstance(wall, dict):
                continue
            try:
                wid = int(wall.get("wall_id") or 0)
            except (TypeError, ValueError):
                continue
            if wid != int(wall_id):
                continue
            code = str(wall.get("shelf_code") or "").strip()
            if code:
                return code
    for mesh in data.get("slot_meshes") or []:
        if not isinstance(mesh, dict):
            continue
        try:
            wid = int(mesh.get("wall_id") or 0)
        except (TypeError, ValueError):
            continue
        if wid == int(wall_id):
            return str(mesh.get("shelf_code") or "").strip()
    return ""


def list_aisles(json_dir: str | None = None) -> list[dict]:
    d = aisles_dir(json_dir)
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        data = _read(os.path.join(d, name))
        if not data:
            continue
        cams = data.get("cameras") or {}
        meshes = data.get("slot_meshes") or []
        out.append({
            "aisle_id": data.get("aisle_id") or name[:-5],
            "camera_l": (cams.get("L") or {}).get("camera_id") or "",
            "camera_r": (cams.get("R") or {}).get("camera_id") or "",
            "solved": bool((data.get("solved") or {}).get("ok")),
            "mesh_walls": len(meshes) if isinstance(meshes, list) else 0,
        })
    return out


def load_aisle(aisle_id: str, json_dir: str | None = None) -> dict | None:
    aid = str(aisle_id or "").strip()
    if not aid:
        return None
    return _read(aisle_path(aid, json_dir))


def save_aisle(data: dict, json_dir: str | None = None) -> dict:
    aid = str(data.get("aisle_id") or "").strip()
    if not _AISLE_RE.match(aid):
        raise ValueError("巷道编号仅支持字母、数字、下划线、中划线（1–64）")
    data = dict(data)
    data["aisle_id"] = aid
    data["required_wall_ids"] = required_wall_ids(data)
    meshes = data.get("slot_meshes")
    if isinstance(meshes, list):
        synced = []
        for mesh in meshes:
            if not isinstance(mesh, dict):
                continue
            m = dict(mesh)
            try:
                wid = int(m.get("wall_id") or 0)
            except (TypeError, ValueError):
                synced.append(m)
                continue
            sc = str(m.get("shelf_code") or "").strip() or wall_shelf_code(data, wid)
            if sc:
                m["shelf_code"] = sc
            synced.append(m)
        data["slot_meshes"] = synced
    _write(aisle_path(aid, json_dir), data)
    return data


def camera_group(camera_id: str, json_dir: str | None = None) -> dict | None:
    """查找相机所属巷道：{aisle_id, role}。"""
    cid = str(camera_id or "").strip()
    if not cid:
        return None
    for item in list_aisles(json_dir):
        aid = item["aisle_id"]
        data = load_aisle(aid, json_dir)
        if not data:
            continue
        cams = data.get("cameras") or {}
        for role in ROLES:
            if str((cams.get(role) or {}).get("camera_id") or "").strip() == cid:
                return {"aisle_id": aid, "role": role}
    return None


def grouped_cameras(json_dir: str | None = None) -> dict[str, dict]:
    """camera_id → {aisle_id, role}。"""
    out: dict[str, dict] = {}
    for item in list_aisles(json_dir):
        data = load_aisle(item["aisle_id"], json_dir)
        if not data:
            continue
        cams = data.get("cameras") or {}
        for role in ROLES:
            cid = str((cams.get(role) or {}).get("camera_id") or "").strip()
            if cid:
                out[cid] = {"aisle_id": item["aisle_id"], "role": role}
    return out


def require_grouped(camera_id: str, json_dir: str | None = None) -> tuple[dict | None, str | None]:
    """开推理前检查：必须属于完整的 L+R 组。"""
    g = camera_group(camera_id, json_dir)
    if not g:
        return None, "未勾选同一组：禁止开推理。请先在巷道标注页把左右路绑成一组。"
    data = load_aisle(g["aisle_id"], json_dir)
    if not data:
        return None, "巷道标定文件缺失"
    cams = data.get("cameras") or {}
    left = str((cams.get("L") or {}).get("camera_id") or "").strip()
    right = str((cams.get("R") or {}).get("camera_id") or "").strip()
    if not left or not right:
        return None, "同一组必须指定左路和右路两台摄像头"
    if left == right:
        return None, "左右路不能是同一台摄像头"
    return {"aisle_id": g["aisle_id"], "role": g["role"], "L": left, "R": right}, None


def require_inference_ready(camera_id: str, json_dir: str | None = None) -> tuple[dict | None, str | None]:
    """开推理前：成组 + 已反解 + 已勾选拣货墙都有层线和货架号。缺哪一步就说哪一步。"""
    grouped, err = require_grouped(camera_id, json_dir)
    if err:
        return None, err
    aid = grouped["aisle_id"]
    data = load_aisle(aid, json_dir) or {}
    solved = data.get("solved") or {}
    if not solved.get("ok"):
        return None, (
            f"巷道 {aid} 尚未反解：请到「巷道标注」页点「1. 反解并对齐」。"
            "没有相机外参无法把 2D 姿态抬到 3D，开了检测也不会产生碰撞事件。"
        )
    need = required_wall_ids(data)
    meshes = [m for m in (data.get("slot_meshes") or []) if isinstance(m, dict)]
    by_wall: dict[int, dict] = {}
    for mesh in meshes:
        try:
            wid = int(mesh.get("wall_id"))
        except (TypeError, ValueError):
            continue
        by_wall[wid] = mesh
    missing = [w for w in need if w not in by_wall]
    if missing:
        walls = "、".join(f"墙{w}" for w in missing)
        return None, (
            f"巷道 {aid} 已配置拣货墙 {[f'墙{w}' for w in need]}，但{walls}尚未生成货格层线。"
            "请到「巷道标注」反解后拖层线，再点「保存墙标定」。"
            "没有 slot_meshes 就无法做 3D 贴墙碰撞。"
        )
    no_shelf = [w for w in need if not str((by_wall[w] or {}).get("shelf_code") or "").strip()
                and not wall_shelf_code(data, w)]
    if no_shelf:
        walls = "、".join(f"墙{w}" for w in no_shelf)
        return None, (
            f"巷道 {aid} 的{walls}未填写货架号。"
            "请到「巷道标注」页填写该墙的货架号（shelf_code）并保存。"
            "碰撞事件按「货架号:货位编号」上报。"
        )
    grouped["solved"] = True
    grouped["mesh_walls"] = len(need)
    grouped["required_wall_ids"] = need
    return grouped, None


def bind_group(
    aisle_id: str,
    camera_l: str,
    camera_r: str,
    json_dir: str | None = None,
) -> dict:
    aid = str(aisle_id or "").strip()
    if not _AISLE_RE.match(aid):
        raise ValueError("巷道编号仅支持字母、数字、下划线、中划线（1–64）")
    left = str(camera_l or "").strip()
    right = str(camera_r or "").strip()
    if not left or not right:
        raise ValueError("同一组必须指定左路和右路")
    if left == right:
        raise ValueError("左右路不能是同一台摄像头")

    for other in list_aisles(json_dir):
        oid = other["aisle_id"]
        if oid == aid:
            continue
        data = load_aisle(oid, json_dir) or {}
        cams = data.get("cameras") or {}
        for role in ROLES:
            cid = str((cams.get(role) or {}).get("camera_id") or "").strip()
            if cid in (left, right):
                raise ValueError(f"摄像头 {cid} 已属于巷道 {oid}")

    data = load_aisle(aid, json_dir) or empty_aisle(aid)
    data["aisle_id"] = aid
    data.setdefault("cameras", {})
    data["cameras"]["L"] = {"camera_id": left, "role": "L"}
    data["cameras"]["R"] = {"camera_id": right, "role": "R"}
    save_aisle(data, json_dir)
    return data


def unbind_group(aisle_id: str, json_dir: str | None = None) -> dict | None:
    data = load_aisle(aisle_id, json_dir)
    if not data:
        return None
    data["cameras"] = {"L": {"camera_id": "", "role": "L"}, "R": {"camera_id": "", "role": "R"}}
    save_aisle(data, json_dir)
    return data


def shard_key_for_camera(camera_id: str, json_dir: str | None = None) -> str:
    """分片键：已成组用 aisle_id，否则 camera_id（未成组不应发 pose）。"""
    g = camera_group(camera_id, json_dir)
    if g:
        return g["aisle_id"]
    return str(camera_id or "").strip()
