"""巷道组与标定 API。"""

from __future__ import annotations

from fastapi import APIRouter, Request

from dualcam.geom import make_layer_mesh, wall_by_id
from dualcam.solve import solve_dual
from services.aisle_store import (
    aisle_layout,
    apply_capture_sizes,
    bind_group,
    camera_group,
    create_aisle_with_cameras,
    delete_aisle_with_cameras,
    list_aisles,
    load_aisle,
    missing_required_quads,
    save_aisle,
    sync_slot_meshes_to_solved,
    unbind_group,
    update_aisle_cameras,
    views_for_solve,
    wall_shelf_code,
)
from services.dualcam_overlay import overlay_for_role
from services.event_engine.sharding import logical_shard_id


def _layout_change_blocked(aisle: dict) -> str | None:
    """检测开着时不许改对侧/同侧，避免外参和正在跑的姿态对不上。"""
    try:
        from services.inference_container_service import get_inference_status
    except Exception:
        return None
    cams = aisle.get("cameras") or {}
    for role, label in (("L", "左路"), ("R", "右路")):
        cid = str((cams.get(role) or {}).get("camera_id") or "").strip()
        if not cid:
            continue
        try:
            status = str((get_inference_status(cid) or {}).get("status") or "")
        except Exception:
            continue
        if status in ("running", "starting"):
            return f"{label}检测还在跑，请先停止后再改巷道布局（对侧/同侧）。"
    return None


def register_aisle_routes(
    router: APIRouter,
    *,
    json_dir: str = "localdata/json",
    camera_ips_file: str = "localdata/camera_ips.json",
    mediamtx_config_path: str = "localdata/mediamtx.yml",
):
    @router.get("/aisles")
    async def api_list_aisles():
        items = list_aisles(json_dir)
        for it in items:
            it["logical_shard"] = logical_shard_id(it["aisle_id"])
        return {"status": "success", "items": items}

    @router.post("/aisles")
    async def api_create_aisle(payload: dict, request: Request):
        from services.annotation_service import materialize_camera_annotation
        from services.audit_service import audit_from_result

        aid = str((payload or {}).get("aisle_id") or "").strip()
        result = create_aisle_with_cameras(
            aid,
            (payload or {}).get("camera_l") or (payload or {}).get("L") or {},
            (payload or {}).get("camera_r") or (payload or {}).get("R") or {},
            camera_file=camera_ips_file,
            mediamtx_config_path=mediamtx_config_path,
            json_dir=json_dir,
            layout=(payload or {}).get("layout"),
        )
        if result.get("status") == "success":
            for cam in (result.get("camera_l"), result.get("camera_r")):
                if cam:
                    materialize_camera_annotation(
                        cam.get("id") or cam.get("path"), json_dir, camera=cam
                    )
            aisle = dict(result["aisle"])
            aisle["logical_shard"] = logical_shard_id(aid)
            result["aisle"] = aisle
        audit_from_result(request, "aisle.create", "aisle", aid, result)
        return result

    @router.delete("/aisles/{aisle_id}")
    async def api_delete_aisle(aisle_id: str, request: Request):
        from services.audit_service import audit_from_result

        result = delete_aisle_with_cameras(
            aisle_id,
            camera_file=camera_ips_file,
            mediamtx_config_path=mediamtx_config_path,
            json_dir=json_dir,
        )
        audit_from_result(request, "aisle.delete", "aisle", aisle_id, result)
        return result

    @router.put("/aisles/{aisle_id}/cameras")
    async def api_update_aisle_cameras(aisle_id: str, payload: dict, request: Request):
        from services.audit_service import audit_from_result

        result = update_aisle_cameras(
            aisle_id,
            (payload or {}).get("camera_l") or (payload or {}).get("L") or {},
            (payload or {}).get("camera_r") or (payload or {}).get("R") or {},
            camera_file=camera_ips_file,
            mediamtx_config_path=mediamtx_config_path,
            json_dir=json_dir,
            new_aisle_id=str((payload or {}).get("aisle_id") or "").strip() or None,
        )
        if result.get("status") == "success":
            aisle = dict(result["aisle"])
            aid = str(aisle.get("aisle_id") or aisle_id)
            aisle["logical_shard"] = logical_shard_id(aid)
            result["aisle"] = aisle
        audit_from_result(request, "aisle.update_cameras", "aisle", aisle_id, result)
        return result

    @router.get("/aisles/by-camera/{camera_id}")
    async def api_aisle_by_camera(camera_id: str):
        g = camera_group(camera_id, json_dir)
        if not g:
            return {"status": "error", "error": "该摄像头尚未编入巷道同一组"}
        data = load_aisle(g["aisle_id"], json_dir)
        if not data:
            return {"status": "error", "error": "巷道标定文件缺失"}
        aisle = dict(data)
        aisle["logical_shard"] = logical_shard_id(g["aisle_id"])
        return {
            "status": "success",
            "aisle": aisle,
            "role": g["role"],
            "overlay": overlay_for_role(aisle, g["role"]),
        }

    @router.get("/aisles/{aisle_id}")
    async def api_get_aisle(aisle_id: str):
        data = load_aisle(aisle_id, json_dir)
        if not data:
            return {"status": "error", "error": "巷道不存在"}
        data = dict(data)
        data["logical_shard"] = logical_shard_id(aisle_id)
        return {"status": "success", "aisle": data}

    @router.put("/aisles/{aisle_id}/group")
    async def api_bind_group(aisle_id: str, payload: dict):
        try:
            data = bind_group(
                aisle_id,
                str(payload.get("camera_l") or payload.get("L") or ""),
                str(payload.get("camera_r") or payload.get("R") or ""),
                json_dir,
            )
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        data = dict(data)
        data["logical_shard"] = logical_shard_id(aisle_id)
        return {"status": "success", "aisle": data}

    @router.delete("/aisles/{aisle_id}/group")
    async def api_unbind_group(aisle_id: str):
        data = unbind_group(aisle_id, json_dir)
        if not data:
            return {"status": "error", "error": "巷道不存在"}
        return {"status": "success", "aisle": data}

    @router.put("/aisles/{aisle_id}")
    async def api_save_aisle(aisle_id: str, payload: dict):
        payload = dict(payload or {})
        payload["aisle_id"] = aisle_id
        old = load_aisle(aisle_id, json_dir)
        if old and aisle_layout(old) != aisle_layout(payload):
            blocked = _layout_change_blocked(old)
            if blocked:
                return {"status": "error", "error": blocked}
        try:
            data = save_aisle(payload, json_dir)
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        return {"status": "success", "aisle": data}

    @router.post("/aisles/{aisle_id}/capture-sizes")
    async def api_apply_capture_sizes(aisle_id: str, payload: dict):
        """抽帧后把各路真实像素写入 image_size，并按 contain 逆映射已标四角。"""
        sizes = payload.get("sizes") if isinstance(payload, dict) else None
        if not isinstance(sizes, dict):
            sizes = payload if isinstance(payload, dict) else {}
        views_overlay = payload.get("views") if isinstance(payload, dict) else None
        data, changed = apply_capture_sizes(
            aisle_id, sizes, json_dir, views_overlay=views_overlay if isinstance(views_overlay, dict) else None
        )
        if not data:
            return {"status": "error", "error": "巷道不存在"}
        data = dict(data)
        data["logical_shard"] = logical_shard_id(aisle_id)
        return {"status": "success", "aisle": data, "size_changed": changed}

    @router.post("/aisles/{aisle_id}/solve")
    async def api_solve_aisle(aisle_id: str, payload: dict | None = None):
        data = dict(payload or load_aisle(aisle_id, json_dir) or {})
        data["aisle_id"] = aisle_id
        missing = missing_required_quads(data)
        if missing:
            return {
                "status": "error",
                "error": f"请先标齐{'、'.join(missing)}。未勾选「参与拣货」的墙不用标。",
                "aisle": data,
            }
        dual_payload = {
            "aisle": data.get("aisle", 2.0),
            "prior": data.get("prior") or {},
            "layout": aisle_layout(data),
            "ground_rect": data.get("ground_rect") or {},
            "views": views_for_solve(data),
        }
        solved = solve_dual(dual_payload)
        data["solved"] = solved
        if solved.get("ok"):
            data = sync_slot_meshes_to_solved(data)
        save_aisle(data, json_dir)
        if not solved.get("ok"):
            return {"status": "error", "error": solved.get("error") or "反解失败", "aisle": data}
        return {"status": "success", "aisle": data}

    @router.post("/aisles/{aisle_id}/mesh")
    async def api_make_mesh(aisle_id: str, payload: dict):
        data = load_aisle(aisle_id, json_dir)
        if not data:
            return {"status": "error", "error": "巷道不存在"}
        solved = data.get("solved") or {}
        if not solved.get("ok"):
            return {
                "status": "error",
                "error": "尚未反解：请先点「1. 反解并对齐」。没有墙面世界坐标就无法生成货格层线。",
            }
        wall_id = int(payload.get("wall_id") or 1)
        wall = wall_by_id(solved, wall_id)
        if not wall:
            return {"status": "error", "error": f"没有墙 {wall_id}"}
        n_layers = int(payload.get("n_layers") or payload.get("rows") or 4)
        n_cols = int(payload.get("n_cols") or payload.get("cols") or 4)
        old = next(
            (
                m for m in (data.get("slot_meshes") or [])
                if isinstance(m, dict) and int(m.get("wall_id") or 0) == wall_id
            ),
            None,
        )
        mesh = make_layer_mesh(wall_id, wall["corners"], n_layers=n_layers, cols=n_cols)
        shelf = str(payload.get("shelf_code") or "").strip() or wall_shelf_code(data, wall_id)
        if old and isinstance(old, dict):
            if not shelf:
                shelf = str(old.get("shelf_code") or "").strip()
            if old.get("cell_ids"):
                mesh["cell_ids"] = old["cell_ids"]
            if old.get("deleted"):
                mesh["deleted"] = old["deleted"]
        if shelf:
            mesh["shelf_code"] = shelf
        meshes = [m for m in (data.get("slot_meshes") or []) if int(m.get("wall_id") or 0) != wall_id]
        meshes.append(mesh)
        data["slot_meshes"] = meshes
        if "contact_m" in payload:
            data["contact_m"] = float(payload["contact_m"])
        save_aisle(data, json_dir)
        return {"status": "success", "aisle": data}
