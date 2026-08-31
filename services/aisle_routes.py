"""巷道组与标定 API。"""

from __future__ import annotations

from fastapi import APIRouter

from dualcam.geom import make_layer_mesh, wall_by_id
from dualcam.solve import solve_dual
from services.aisle_store import (
    bind_group,
    camera_group,
    list_aisles,
    load_aisle,
    save_aisle,
    unbind_group,
    wall_shelf_code,
)
from services.dualcam_overlay import overlay_for_role
from services.event_engine.sharding import logical_shard_id


def register_aisle_routes(router: APIRouter, *, json_dir: str = "localdata/json"):
    @router.get("/aisles")
    async def api_list_aisles():
        items = list_aisles(json_dir)
        for it in items:
            it["logical_shard"] = logical_shard_id(it["aisle_id"])
        return {"status": "success", "items": items}

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
        try:
            data = save_aisle(payload, json_dir)
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        return {"status": "success", "aisle": data}

    @router.post("/aisles/{aisle_id}/solve")
    async def api_solve_aisle(aisle_id: str, payload: dict | None = None):
        data = dict(payload or load_aisle(aisle_id, json_dir) or {})
        data["aisle_id"] = aisle_id
        views = data.get("views") or {}
        dual_payload = {
            "aisle": data.get("aisle", 2.0),
            "prior": data.get("prior") or {},
            "views": [views.get("L") or {}, views.get("R") or {}],
        }
        if dual_payload["views"][0]:
            dual_payload["views"][0]["name"] = "L"
        if dual_payload["views"][1]:
            dual_payload["views"][1]["name"] = "R"
        solved = solve_dual(dual_payload)
        data["solved"] = solved
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
