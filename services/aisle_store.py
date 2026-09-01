"""巷道组：两路相机绑定 + 标定 JSON。未成组禁止开推理。"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
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


_REPO_ROOT = Path(__file__).resolve().parents[1]
PICKSTATE_CALIB_CANDIDATES = (
    Path("/home/hqit/workspace/visual-dps-pick-state/output/calib/dual_1-3.json"),
    _REPO_ROOT / "fixtures" / "dual_1-3.json",
)


def resolve_pickstate_calib(path: str | None = None) -> Path:
    raw = str(path or os.environ.get("PICKSTATE_CALIB") or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_file():
            raise FileNotFoundError(f"标定文件不存在：{p}")
        return p
    for cand in PICKSTATE_CALIB_CANDIDATES:
        if cand.is_file():
            return cand
    raise FileNotFoundError("找不到实验仓 dual_1-3.json（visual-dps-pick-state 或本仓 fixtures）")


def _shelf_map(aisle: dict | None) -> dict[int, str]:
    out: dict[int, str] = {}
    data = aisle if isinstance(aisle, dict) else {}
    views = data.get("views") or {}
    for role in ROLES:
        for wall in ((views.get(role) or {}).get("walls") or []):
            if not isinstance(wall, dict):
                continue
            try:
                wid = int(wall.get("wall_id") or 0)
            except (TypeError, ValueError):
                continue
            code = str(wall.get("shelf_code") or "").strip()
            if wid >= 1 and code and wid not in out:
                out[wid] = code
    for mesh in data.get("slot_meshes") or []:
        if not isinstance(mesh, dict):
            continue
        try:
            wid = int(mesh.get("wall_id") or 0)
        except (TypeError, ValueError):
            continue
        code = str(mesh.get("shelf_code") or "").strip()
        if wid >= 1 and code and wid not in out:
            out[wid] = code
    return out


def merge_pickstate_calib(aisle: dict, calib: dict) -> dict:
    """把实验仓 dualcam 标定并进本仓巷道：保留 aisle_id / 相机绑定 / aabb / 货架号。"""
    if not isinstance(calib, dict) or not (calib.get("views") or calib.get("solved")):
        raise ValueError("标定文件缺少 views 或 solved")
    out = dict(aisle)
    keep_id = str(aisle.get("aisle_id") or "").strip()
    keep_cams = copy.deepcopy(aisle.get("cameras") or {})
    keep_aabb = copy.deepcopy(aisle.get("aabb")) if aisle.get("aabb") else None
    shelves = _shelf_map(aisle)
    for key in ("aisle", "contact_m", "prior"):
        if key in calib and calib[key] is not None:
            out[key] = copy.deepcopy(calib[key])
    solved = calib.get("solved")
    if isinstance(solved, dict):
        out["solved"] = copy.deepcopy(solved)
    meshes_in = calib.get("slot_meshes")
    mesh_by_wall: dict[int, dict] = {}
    if isinstance(meshes_in, list):
        meshes = []
        for mesh in meshes_in:
            if not isinstance(mesh, dict):
                continue
            m = copy.deepcopy(mesh)
            try:
                wid = int(m.get("wall_id") or 0)
            except (TypeError, ValueError):
                meshes.append(m)
                continue
            code = str(m.get("shelf_code") or "").strip() or shelves.get(wid) or f"wall{wid}"
            m["shelf_code"] = code
            if wid >= 1:
                mesh_by_wall[wid] = m
            meshes.append(m)
        out["slot_meshes"] = meshes
    views_in = calib.get("views") or {}
    views_out: dict[str, Any] = {}
    for role in ROLES:
        src = views_in.get(role) if isinstance(views_in, dict) else None
        if not isinstance(src, dict):
            views_out[role] = copy.deepcopy((aisle.get("views") or {}).get(role) or {"name": role})
            continue
        view = copy.deepcopy(src)
        view["name"] = role
        walls = []
        for wall in view.get("walls") or []:
            if not isinstance(wall, dict):
                continue
            w = dict(wall)
            try:
                wid = int(w.get("wall_id") or 0)
            except (TypeError, ValueError):
                walls.append(w)
                continue
            mesh = mesh_by_wall.get(wid) or {}
            w["shelf_code"] = str(w.get("shelf_code") or "").strip() or shelves.get(wid) or f"wall{wid}"
            layers = int(w.get("n_layers") or mesh.get("n_layers") or mesh.get("rows") or 4)
            cols = int(w.get("n_cols") or mesh.get("cols") or 4)
            w["n_layers"] = layers
            w["n_cols"] = cols
            walls.append(w)
        view["walls"] = walls
        views_out[role] = view
    out["views"] = views_out
    out["aisle_id"] = keep_id
    out["cameras"] = keep_cams
    if keep_aabb:
        out["aabb"] = keep_aabb
    if not parse_wall_ids(out.get("required_wall_ids")):
        out["required_wall_ids"] = sorted(mesh_by_wall.keys()) or [1]
    return out


def import_pickstate_calib(
    aisle_id: str,
    calib_path: str | None = None,
    json_dir: str | None = None,
) -> tuple[dict, str]:
    """从实验仓 JSON 导入标定，覆盖四角 / 层线 / 反解 / 货格，不改摄像头绑定。"""
    aid = str(aisle_id or "").strip()
    data = load_aisle(aid, json_dir)
    if not data:
        raise FileNotFoundError(f"巷道 {aid} 不存在")
    src = resolve_pickstate_calib(calib_path)
    calib = _read(str(src))
    if not calib:
        raise ValueError(f"无法读取标定：{src}")
    merged = merge_pickstate_calib(data, calib)
    saved = save_aisle(merged, json_dir)
    return saved, str(src)


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


def apply_capture_sizes(
    aisle_id: str,
    sizes: dict[str, Any],
    json_dir: str | None = None,
    views_overlay: dict | None = None,
) -> tuple[dict | None, bool]:
    """把抽帧得到的真实宽高写入 L/R image_size。尺寸变了则作废反解。

    sizes 形如 {"L": {"width": 960, "height": 720}, "R": {...}}。
    views_overlay 可带上尚未保存的四角，避免只改了磁盘上的旧点。
    返回 (aisle, size_changed)。
    """
    from services.dualcam_config import remap_view_image_size

    data = load_aisle(aisle_id, json_dir)
    if not data:
        return None, False
    views = dict(data.get("views") or {})
    if isinstance(views_overlay, dict):
        for role in ROLES:
            overlay = views_overlay.get(role)
            if isinstance(overlay, dict):
                views[role] = dict(overlay)
    changed = False
    for role in ROLES:
        spec = sizes.get(role) if isinstance(sizes, dict) else None
        if not isinstance(spec, dict):
            continue
        try:
            width = int(spec.get("width") or 0)
            height = int(spec.get("height") or 0)
        except (TypeError, ValueError):
            continue
        view = dict(views.get(role) or {"name": role})
        if remap_view_image_size(view, width, height):
            changed = True
        views[role] = view
    data["views"] = views
    if changed:
        data["solved"] = {"ok": False}
    return save_aisle(data, json_dir), changed


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


def create_aisle_with_cameras(
    aisle_id: str,
    camera_l: dict,
    camera_r: dict,
    *,
    camera_file: str,
    mediamtx_config_path: str,
    json_dir: str | None = None,
) -> dict:
    """一次创建巷道并写入左右路摄像头（上游地址即 IP/RTSP 入口）。"""
    from services.camera_store import create_camera, delete_camera

    aid = str(aisle_id or "").strip()
    if not _AISLE_RE.match(aid):
        return {"error": "巷道编号仅支持字母、数字、下划线、中划线（1–64）"}
    existing = load_aisle(aid, json_dir)
    if existing:
        cams = existing.get("cameras") or {}
        has_l = str((cams.get("L") or {}).get("camera_id") or "").strip()
        has_r = str((cams.get("R") or {}).get("camera_id") or "").strip()
        if has_l or has_r:
            return {"error": f"巷道 {aid} 已存在"}

    left = dict(camera_l or {})
    right = dict(camera_r or {})
    left.setdefault("path", f"{aid}-L")
    right.setdefault("path", f"{aid}-R")
    if not str(left.get("name") or "").strip():
        left["name"] = f"{aid} 左路"
    if not str(right.get("name") or "").strip():
        right["name"] = f"{aid} 右路"

    created: list[str] = []
    r1 = create_camera(camera_file, mediamtx_config_path, left)
    if r1.get("error"):
        return {"error": f"左路：{r1['error']}"}
    created.append(str((r1.get("camera") or {}).get("id") or ""))
    r2 = create_camera(camera_file, mediamtx_config_path, right)
    if r2.get("error"):
        for cid in created:
            if cid:
                delete_camera(camera_file, mediamtx_config_path, cid)
        return {"error": f"右路：{r2['error']}"}
    created.append(str((r2.get("camera") or {}).get("id") or ""))
    try:
        aisle = bind_group(aid, created[0], created[1], json_dir)
    except ValueError as exc:
        for cid in created:
            if cid:
                delete_camera(camera_file, mediamtx_config_path, cid)
        return {"error": str(exc)}
    return {
        "status": "success",
        "aisle": aisle,
        "camera_l": r1.get("camera"),
        "camera_r": r2.get("camera"),
        "items": r2.get("items"),
        "mediamtx": r2.get("mediamtx"),
    }


def delete_aisle_with_cameras(
    aisle_id: str,
    *,
    camera_file: str,
    mediamtx_config_path: str,
    json_dir: str | None = None,
) -> dict:
    """删除巷道绑定的左右路摄像头，并解开成组（标定 JSON 保留）。"""
    from services.camera_store import delete_camera, load_cameras

    aid = str(aisle_id or "").strip()
    data = load_aisle(aid, json_dir)
    if not data:
        return {"error": "巷道不存在"}
    cams = data.get("cameras") or {}
    ids = []
    for role in ROLES:
        cid = str((cams.get(role) or {}).get("camera_id") or "").strip()
        if cid:
            ids.append(cid)
    last = {"items": load_cameras(camera_file), "mediamtx": None}
    for cid in ids:
        r = delete_camera(camera_file, mediamtx_config_path, cid)
        if r.get("error") and r["error"] != "未找到该摄像头":
            return {"error": f"{cid}：{r['error']}"}
        if r.get("status") == "success":
            last = r
    unbind_group(aid, json_dir)
    return {
        "status": "success",
        "aisle_id": aid,
        "items": last.get("items"),
        "mediamtx": last.get("mediamtx"),
    }
