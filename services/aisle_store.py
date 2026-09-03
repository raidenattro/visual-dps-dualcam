"""巷道组：两路相机绑定 + 标定 JSON。未成组禁止开推理。"""

from __future__ import annotations

import json
import os
import re
from typing import Any

_AISLE_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
ROLES = ("L", "R")
# 巷道只有墙1/墙2；新巷道默认两面都参与拣货。
ALLOWED_WALL_IDS = (1, 2)
DEFAULT_REQUIRED_WALL_IDS = [1, 2]
# json_dir → (stamp, camera_id→{aisle_id, role})
_GROUPED_CACHE: dict[str, tuple[tuple, dict[str, dict]]] = {}


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
        "required_wall_ids": list(DEFAULT_REQUIRED_WALL_IDS),
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
        if wid in ALLOWED_WALL_IDS and wid not in out:
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
        if wid in ALLOWED_WALL_IDS and wid not in from_mesh:
            from_mesh.append(wid)
    return from_mesh or list(DEFAULT_REQUIRED_WALL_IDS)


def views_for_solve(aisle: dict | None) -> list[dict]:
    """反解只吃勾选拣货墙；未勾选的墙即使有四角也不送进求解器。"""
    data = aisle if isinstance(aisle, dict) else {}
    need = set(required_wall_ids(data))
    views = data.get("views") if isinstance(data.get("views"), dict) else {}
    out: list[dict] = []
    for role in ROLES:
        raw = views.get(role)
        view = dict(raw) if isinstance(raw, dict) else {"name": role}
        view["name"] = role
        walls: list[dict] = []
        for wall in view.get("walls") or []:
            if not isinstance(wall, dict):
                continue
            try:
                wid = int(wall.get("wall_id") or 0)
            except (TypeError, ValueError):
                continue
            if wid in need:
                walls.append(wall)
        view["walls"] = walls
        out.append(view)
    return out


def missing_required_quads(aisle: dict | None) -> list[str]:
    """勾选墙在左右路是否都标满四角。返回如「左路墙1」。"""
    data = aisle if isinstance(aisle, dict) else {}
    need = required_wall_ids(data)
    views = data.get("views") if isinstance(data.get("views"), dict) else {}
    role_label = {"L": "左路", "R": "右路"}
    missing: list[str] = []
    for role in ROLES:
        raw = views.get(role)
        walls = raw.get("walls") if isinstance(raw, dict) else []
        by_id: dict[int, dict] = {}
        for wall in walls or []:
            if not isinstance(wall, dict):
                continue
            try:
                wid = int(wall.get("wall_id") or 0)
            except (TypeError, ValueError):
                continue
            by_id[wid] = wall
        for wid in need:
            quad = (by_id.get(wid) or {}).get("quad") or []
            if not isinstance(quad, list) or len(quad) < 4:
                missing.append(f"{role_label[role]}墙{wid}")
    return missing


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


def list_aisles(json_dir: str | None = None, *, bound_only: bool = True) -> list[dict]:
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
        camera_l = (cams.get("L") or {}).get("camera_id") or ""
        camera_r = (cams.get("R") or {}).get("camera_id") or ""
        if bound_only and (not str(camera_l).strip() or not str(camera_r).strip()):
            continue
        meshes = data.get("slot_meshes") or []
        out.append({
            "id": data.get("id"),
            "aisle_id": data.get("aisle_id") or name[:-5],
            "camera_l": camera_l,
            "camera_r": camera_r,
            "solved": bool((data.get("solved") or {}).get("ok")),
            "mesh_walls": len(meshes) if isinstance(meshes, list) else 0,
        })
    return out


def _next_aisle_pk(json_dir: str | None = None) -> int:
    """巷道内部自增主键；aisle_id 只是显示用的巷道号。"""
    max_n = 0
    for item in list_aisles(json_dir, bound_only=False):
        raw = item.get("id")
        if str(raw).isdigit():
            max_n = max(max_n, int(raw))
    return max_n + 1


def _ensure_aisle_pk(data: dict, json_dir: str | None = None) -> dict:
    data = dict(data)
    if str(data.get("id") or "").isdigit():
        data["id"] = int(data["id"])
        return data
    data["id"] = _next_aisle_pk(json_dir)
    return data


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
    from services.dualcam_config import calib_size_from_view, remap_view_image_size, uniform_pixel_scale, scale_solved_for_pixel_resize

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
    scales: list[float] = []
    aspect_mismatch = False
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
        old_w, old_h = calib_size_from_view(view)
        if remap_view_image_size(view, width, height):
            changed = True
            sc = uniform_pixel_scale(old_w, old_h, width, height)
            if sc is None:
                aspect_mismatch = True
            else:
                scales.append(sc)
        views[role] = view
        if aspect_mismatch:
            break
    data["views"] = views
    if changed:
        solved = data.get("solved")
        same_scale = (
            not aspect_mismatch
            and scales
            and len({round(s, 6) for s in scales}) == 1
        )
        if isinstance(solved, dict) and solved.get("ok") and same_scale:
            data["solved"] = scale_solved_for_pixel_resize(solved, scales[0])
        else:
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


def rename_aisle(old_id: str, new_id: str, json_dir: str | None = None) -> dict:
    """改巷道号（显示名）。内部 id 不变；标定文件改名。检测会先停，避免 AISLE_ID 指向旧文件。"""
    old = str(old_id or "").strip()
    new = str(new_id or "").strip()
    if old == new:
        data = load_aisle(old, json_dir)
        return {"status": "success", "aisle": data} if data else {"error": "巷道不存在"}
    if not _AISLE_RE.match(new):
        return {"error": "巷道编号仅支持字母、数字、下划线、中划线（1–64）"}
    if load_aisle(new, json_dir):
        return {"error": f"巷道 {new} 已存在"}
    data = load_aisle(old, json_dir)
    if not data:
        return {"error": "巷道不存在"}

    from services.inference_container_service import stop_inference_container

    cams = data.get("cameras") or {}
    for role in ROLES:
        cid = str((cams.get(role) or {}).get("camera_id") or "").strip()
        if cid:
            stop_inference_container(cid)

    data = _ensure_aisle_pk(data, json_dir)
    data["aisle_id"] = new
    saved = save_aisle(data, json_dir)
    old_path = aisle_path(old, json_dir)
    if os.path.isfile(old_path):
        try:
            os.remove(old_path)
        except OSError:
            pass
    return {"status": "success", "aisle": saved, "renamed_from": old}


def camera_group(camera_id: str, json_dir: str | None = None) -> dict | None:
    """查找相机所属巷道：{aisle_id, role}。"""
    cid = str(camera_id or "").strip()
    if not cid:
        return None
    for item in list_aisles(json_dir, bound_only=False):
        aid = item["aisle_id"]
        data = load_aisle(aid, json_dir)
        if not data:
            continue
        cams = data.get("cameras") or {}
        for role in ROLES:
            if str((cams.get(role) or {}).get("camera_id") or "").strip() == cid:
                return {"aisle_id": aid, "role": role}
    return None


def _aisles_stamp(json_dir: str | None = None) -> tuple:
    """巷道目录内容指纹：文件名 + mtime + size。用来让 grouped_cameras 热路径免读盘。"""
    d = aisles_dir(json_dir)
    try:
        names = tuple(sorted(n for n in os.listdir(d) if n.endswith(".json")))
    except OSError:
        return (d, ())
    parts: list[tuple[str, int, int]] = []
    for name in names:
        path = os.path.join(d, name)
        try:
            st = os.stat(path)
            parts.append((name, int(st.st_mtime_ns), int(st.st_size)))
        except OSError:
            parts.append((name, 0, 0))
    return (d, tuple(parts))


def grouped_cameras(json_dir: str | None = None) -> dict[str, dict]:
    """camera_id → {aisle_id, role}。按巷道 JSON mtime 缓存，worker 每条 pose 都要查。"""
    root = _json_dir(json_dir)
    stamp = _aisles_stamp(json_dir)
    hit = _GROUPED_CACHE.get(root)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    out: dict[str, dict] = {}
    for item in list_aisles(json_dir, bound_only=False):
        data = load_aisle(item["aisle_id"], json_dir)
        if not data:
            continue
        cams = data.get("cameras") or {}
        for role in ROLES:
            cid = str((cams.get(role) or {}).get("camera_id") or "").strip()
            if cid:
                out[cid] = {"aisle_id": item["aisle_id"], "role": role}
    _GROUPED_CACHE[root] = (stamp, out)
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

    for other in list_aisles(json_dir, bound_only=False):
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
                delete_camera(camera_file, mediamtx_config_path, cid, json_dir=json_dir)
        return {"error": f"右路：{r2['error']}"}
    created.append(str((r2.get("camera") or {}).get("id") or ""))
    try:
        aisle = bind_group(aid, created[0], created[1], json_dir)
    except ValueError as exc:
        for cid in created:
            if cid:
                delete_camera(camera_file, mediamtx_config_path, cid, json_dir=json_dir)
        return {"error": str(exc)}
    aisle = save_aisle(_ensure_aisle_pk(aisle, json_dir), json_dir)
    return {
        "status": "success",
        "aisle": aisle,
        "camera_l": r1.get("camera"),
        "camera_r": r2.get("camera"),
        "items": r2.get("items"),
        "mediamtx": r2.get("mediamtx"),
    }


def update_aisle_cameras(
    aisle_id: str,
    camera_l: dict,
    camera_r: dict,
    *,
    camera_file: str,
    mediamtx_config_path: str,
    json_dir: str | None = None,
    new_aisle_id: str | None = None,
) -> dict:
    """更新已成组巷道的左右路视频流；巷道号（aisle_id）可改，内部 id 不变。"""
    from services.camera_store import update_camera

    aid = str(aisle_id or "").strip()
    data = load_aisle(aid, json_dir)
    if not data:
        return {"error": "巷道不存在"}
    renamed_from = None
    nxt = str(new_aisle_id or "").strip()
    if nxt and nxt != aid:
        renamed = rename_aisle(aid, nxt, json_dir)
        if renamed.get("error"):
            return renamed
        aid = nxt
        renamed_from = renamed.get("renamed_from") or aisle_id
        data = renamed.get("aisle") or load_aisle(aid, json_dir)
    cams = data.get("cameras") or {}
    id_l = str((cams.get("L") or {}).get("camera_id") or "").strip()
    id_r = str((cams.get("R") or {}).get("camera_id") or "").strip()
    if not id_l or not id_r:
        return {"error": "巷道未绑定左右路"}

    r1 = update_camera(camera_file, mediamtx_config_path, id_l, dict(camera_l or {}))
    if r1.get("error"):
        return {"error": f"左路：{r1['error']}"}
    r2 = update_camera(camera_file, mediamtx_config_path, id_r, dict(camera_r or {}))
    if r2.get("error"):
        return {"error": f"右路：{r2['error']}"}
    aisle = load_aisle(aid, json_dir) or data
    out = {
        "status": "success",
        "aisle": aisle,
        "camera_l": r1.get("camera"),
        "camera_r": r2.get("camera"),
        "items": r2.get("items"),
        "mediamtx": r2.get("mediamtx"),
    }
    if renamed_from:
        out["renamed_from"] = renamed_from
    return out


def delete_aisle_with_cameras(
    aisle_id: str,
    *,
    camera_file: str,
    mediamtx_config_path: str,
    json_dir: str | None = None,
) -> dict:
    """删除巷道：左右路摄像头、标注 JSON、巷道标定文件一并删掉。"""
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
        r = delete_camera(camera_file, mediamtx_config_path, cid, json_dir=json_dir)
        if r.get("error") and r["error"] != "未找到该摄像头":
            return {"error": f"{cid}：{r['error']}"}
        if r.get("status") == "success":
            last = r
    aisle_json = aisle_path(aid, json_dir)
    if os.path.isfile(aisle_json):
        try:
            os.remove(aisle_json)
        except OSError:
            return {"error": f"无法删除巷道文件 {aid}"}
    return {
        "status": "success",
        "aisle_id": aid,
        "items": last.get("items"),
        "mediamtx": last.get("mediamtx"),
    }


def purge_unbound_aisles_and_cameras(
    *,
    camera_file: str,
    mediamtx_config_path: str,
    json_dir: str | None = None,
) -> dict:
    """清掉未成组巷道 JSON、未编入巷道的摄像头，以及无主标注 JSON。"""
    from services.annotation_service import delete_camera_annotation
    from services.camera_store import delete_camera, load_cameras

    removed_aisles: list[str] = []
    bound_cam_ids: set[str] = set()
    d = aisles_dir(json_dir)
    if os.path.isdir(d):
        for name in list(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(d, name)
            data = _read(path)
            if not data:
                continue
            cams = data.get("cameras") or {}
            left = str((cams.get("L") or {}).get("camera_id") or "").strip()
            right = str((cams.get("R") or {}).get("camera_id") or "").strip()
            aid = str(data.get("aisle_id") or name[:-5]).strip()
            if left and right:
                bound_cam_ids.add(left)
                bound_cam_ids.add(right)
                continue
            try:
                os.remove(path)
                removed_aisles.append(aid)
            except OSError:
                pass

    removed_cameras: list[str] = []
    last_items = load_cameras(camera_file)
    last_mtx = None
    for cam in list(last_items):
        cid = str(cam.get("id") or "").strip()
        if not cid or cid in bound_cam_ids:
            continue
        r = delete_camera(camera_file, mediamtx_config_path, cid, json_dir=json_dir)
        if r.get("status") == "success":
            removed_cameras.append(cid)
            last_items = r.get("items") or last_items
            last_mtx = r.get("mediamtx")

    keep = {str(c.get("id") or "") for c in last_items}
    removed_annotations: list[str] = []
    cam_dir = os.path.join(_json_dir(json_dir), "cameras")
    if os.path.isdir(cam_dir):
        for name in list(os.listdir(cam_dir)):
            if not name.endswith(".json"):
                continue
            cid = name[:-5]
            if cid in keep:
                continue
            if delete_camera_annotation(cid, _json_dir(json_dir)):
                removed_annotations.append(cid)

    return {
        "status": "success",
        "removed_aisles": removed_aisles,
        "removed_cameras": removed_cameras,
        "removed_annotations": removed_annotations,
        "items": last_items,
        "mediamtx": last_mtx,
    }
