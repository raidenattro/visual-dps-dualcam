"""把巷道 3D mesh 投影成监控/矩阵可用的 2D token + 多边形。"""

from __future__ import annotations

from typing import Any

from dualcam.geom import mesh_cells, project_pix
from services.aisle_store import wall_shelf_code
from services.box_identity import box_collision_token
from services.dualcam_config import calib_size_from_view, get_dualcam_section


def collision_token(shelf_code: str, box_id: Any) -> str:
    """与 visual-dps 相同：``{shelf_code}:{货位编号}``。"""
    return box_collision_token({
        "shelf_code": str(shelf_code or "").strip(),
        "box_id": str(box_id or "").strip(),
    })


def _view_cam(aisle: dict, role: str) -> dict | None:
    solved = aisle.get("solved") or {}
    cams = solved.get("cameras") or {}
    cam = cams.get(role)
    return cam if isinstance(cam, dict) else None


def overlay_for_role(aisle: dict, role: str) -> dict[str, Any]:
    """投影本路看到的货格：video_polygon 在标定像素系，token 用货架号+货位编号。"""
    aid = str(aisle.get("aisle_id") or "").strip()
    role = str(role or "").strip().upper() or "L"
    views = aisle.get("views") or {}
    view = views.get(role) if isinstance(views, dict) else {}
    cw, ch = calib_size_from_view(view if isinstance(view, dict) else None, get_dualcam_section())
    cam = _view_cam(aisle, role)
    shelves: list[dict] = []
    boxes: list[dict] = []
    empty = {
        "aisle_id": aid,
        "role": role,
        "annotation_width": cw,
        "annotation_height": ch,
        "shelves": [],
        "boxes": [],
    }
    if not cam:
        return empty

    for mesh in aisle.get("slot_meshes") or []:
        if not isinstance(mesh, dict):
            continue
        wall_id = mesh.get("wall_id")
        try:
            wid = int(wall_id)
        except (TypeError, ValueError):
            continue
        shelf_code = str(mesh.get("shelf_code") or "").strip() or wall_shelf_code(aisle, wid)
        if not shelf_code:
            continue
        cells = mesh_cells(mesh)
        shelf_boxes: list[dict] = []
        rows = int(mesh.get("rows") or 0)
        cols = int(mesh.get("cols") or 0)
        for cell in cells:
            poly = []
            ok = True
            for corner in cell.get("corners") or []:
                uv = project_pix(corner, cam)
                if uv is None:
                    ok = False
                    break
                poly.append([float(uv[0]), float(uv[1])])
            if not ok or len(poly) < 3:
                continue
            box_id = str(cell.get("box_id") or "").strip()
            if not box_id:
                continue
            box = {
                "box_id": box_id,
                "shelf_code": shelf_code,
                "layer": int(cell.get("row") or 0) + 1,
                "column": int(cell.get("col") or 0) + 1,
                "wall_id": wall_id,
                "video_polygon": poly,
            }
            shelf_boxes.append(box)
            boxes.append(box)
        if not shelf_boxes:
            continue
        shelves.append({
            "shelf_code": shelf_code,
            "shelf_name": str(mesh.get("shelf_name") or shelf_code),
            "grid_shape": [rows, cols] if rows and cols else [],
            "wall_id": wall_id,
            "boxes": shelf_boxes,
            "shelf_corners": [],
        })

    return {
        "aisle_id": aid,
        "role": role,
        "annotation_width": cw,
        "annotation_height": ch,
        "shelves": shelves,
        "boxes": boxes,
    }
