"""3D 货格：像素射线 ∩ 墙面再投影，两路应对上同一物理点。"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dualcam.geom import (
    contact_slots,
    drag_vertex,
    equal_row_ys,
    make_grid_vertices,
    make_layer_mesh,
    mesh_from_row_ys,
    mesh_cells,
    move_layer_row,
    offset_corners,
    project_pix,
    ray_plane,
    signed_wall_dist,
    triangulate_pixels,
    vert_index,
    wall_by_id,
    wall_plane,
    wall_y_span,
)

CALIB = ROOT / "fixtures" / "dual_1-3.json"


def _load():
    if not CALIB.is_file():
        pytest.skip("缺少 dual_1-3.json")
    data = json.loads(CALIB.read_text(encoding="utf-8"))
    solved = data.get("solved") or {}
    if not solved.get("ok"):
        pytest.skip("dual_1-3.json 没有有效 solved")
    return data, solved


def test_project_unproject_roundtrip_on_wall():
    _, solved = _load()
    wall = wall_by_id(solved, 1)
    p0, n = wall_plane(wall["corners"])
    cam = solved["cameras"]["L"]
    p = np.array(wall["corners"][1], float)  # ② 顶近
    uv = project_pix(p, cam)
    assert uv is not None
    back = ray_plane(uv[0], uv[1], cam, p0, n)
    assert back is not None
    assert np.linalg.norm(np.array(back) - p) < 1e-5


def test_same_3d_point_projects_near_both_view_corners():
    """左路① 与右路② 是同一物理角；投影应落在各自 2D 四角附近（残差量级）。"""
    data, solved = _load()
    p = np.array(wall_by_id(solved, 1)["corners"][0], float)
    uv_l = project_pix(p, solved["cameras"]["L"])
    uv_r = project_pix(p, solved["cameras"]["R"])
    q_l = np.array(data["views"]["L"]["walls"][0]["quad"][0], float)
    q_r = np.array(data["views"]["R"]["walls"][0]["quad"][1], float)  # 对向：左① = 右②
    assert np.linalg.norm(np.array(uv_l) - q_l) < 30
    assert np.linalg.norm(np.array(uv_r) - q_r) < 30


def test_drag_on_left_moves_right_projection():
    """在左路拖一个格点 = 射线∩面；右路投影必须跟着变，且仍在墙上。"""
    _, solved = _load()
    wall = wall_by_id(solved, 1)
    p0, n = wall_plane(wall["corners"])
    cam_l, cam_r = solved["cameras"]["L"], solved["cameras"]["R"]
    verts = make_grid_vertices(wall["corners"], 4, 4)
    i = vert_index(4, 4, 1, 3)  # 近处一列
    uv0 = project_pix(verts[i], cam_l)
    moved = ray_plane(uv0[0] + 8, uv0[1] - 5, cam_l, p0, n)
    assert moved is not None
    assert abs(moved[0] - wall["corners"][0][0]) < 1e-6  # 仍在 x=const 墙面
    uv_r0 = project_pix(verts[i], cam_r)
    uv_r1 = project_pix(moved, cam_r)
    assert uv_r0 is not None and uv_r1 is not None
    assert np.hypot(uv_r1[0] - uv_r0[0], uv_r1[1] - uv_r0[1]) > 1.0


def test_grid_cell_count_and_corners():
    corners = [[-1, 2, 2], [-1, 2, 0], [-1, 0, 0], [-1, 0, 2]]
    mesh = {"wall_id": 1, "rows": 4, "cols": 4, "vertices": make_grid_vertices(corners, 4, 4)}
    cells = mesh_cells(mesh)
    assert len(cells) == 16
    assert len(mesh["vertices"]) == 25
    c0 = cells[0]["corners"][0]
    assert c0 == pytest.approx([-1, 2, 2])
    assert cells[0]["box_id"] == "r0c0"


def test_offset_corners_moves_toward_aisle():
    corners = [[-1, 2, 2], [-1, 2, 0], [-1, 0, 0], [-1, 0, 2]]
    out = offset_corners(corners, sign=-1, inset=0.3)
    assert out[0][0] == pytest.approx(-0.7)
    assert out[0][1:] == pytest.approx([2, 2])


def test_triangulate_recovers_off_plane_point():
    """开口不在墙上时，两路像素三角化应回到真实 3D 点。"""
    _, solved = _load()
    cam_l, cam_r = solved["cameras"]["L"], solved["cameras"]["R"]
    wall = wall_by_id(solved, 1)
    p = np.array(wall["corners"][1], float) + np.array([0.30, 0, 0])  # 朝巷道 30cm
    uv_l, uv_r = project_pix(p, cam_l), project_pix(p, cam_r)
    back = triangulate_pixels(uv_l[0], uv_l[1], cam_l, uv_r[0], uv_r[1], cam_r)
    assert back is not None
    assert np.linalg.norm(np.array(back) - p) < 0.01


def test_stereo_drag_keeps_other_view_then_aligns_both():
    """Shift 三角化：先在左路跟鼠标，再在右路对准，应回到离面真点。"""
    _, solved = _load()
    cam_l, cam_r = solved["cameras"]["L"], solved["cameras"]["R"]
    wall = wall_by_id(solved, 1)
    p0, n = wall_plane(wall["corners"])
    start = np.array(wall["corners"][1], float)
    true = start + np.array([0.30, 0, 0])
    uv_l_true = project_pix(true, cam_l)
    uv_r_true = project_pix(true, cam_r)
    after_l = drag_vertex(
        uv_l_true[0], uv_l_true[1], cam_l, cam_r, start, p0, n, stereo=True,
    )
    uv_l1 = project_pix(after_l, cam_l)
    assert np.hypot(uv_l1[0] - uv_l_true[0], uv_l1[1] - uv_l_true[1]) < 2
    after_r = drag_vertex(
        uv_r_true[0], uv_r_true[1], cam_r, cam_l, after_l, p0, n, stereo=True,
    )
    assert np.linalg.norm(np.array(after_r) - true) < 0.03
    uv_l2, uv_r2 = project_pix(after_r, cam_l), project_pix(after_r, cam_r)
    assert np.hypot(uv_l2[0] - uv_l_true[0], uv_l2[1] - uv_l_true[1]) < 3
    assert np.hypot(uv_r2[0] - uv_r_true[0], uv_r2[1] - uv_r_true[1]) < 3


def test_contact_slot_into_wall_not_on_or_aisle():
    """只在有向距离 < 0（伸进墙）时报警；贴墙和通道里都不报。"""
    _, solved = _load()
    wall = wall_by_id(solved, 1)
    mesh = make_layer_mesh(1, wall["corners"], pitch=0.45, n_layers=4, cols=4)
    cells = mesh_cells(mesh)
    c = np.mean(np.array(cells[0]["corners"]), axis=0)
    on_wall = c.copy()
    on_wall[0] = wall["corners"][0][0]
    assert signed_wall_dist(on_wall, wall) == pytest.approx(0, abs=1e-6)
    assert contact_slots(on_wall, [mesh], solved) == []

    inward = 1.0 if int(wall.get("sign", -1)) < 0 else -1.0
    into = on_wall.copy()
    into[0] = on_wall[0] - 0.02 * inward
    assert signed_wall_dist(into, wall) == pytest.approx(-0.02, abs=1e-6)
    hits = contact_slots(into, [mesh], solved)
    assert len(hits) == 1
    assert hits[0]["box_id"] == "r0c0"

    in_aisle = on_wall.copy()
    in_aisle[0] = 0.0
    assert signed_wall_dist(in_aisle, wall) > 0
    assert contact_slots(in_aisle, [mesh], solved) == []


def test_layer_initial_equal_split():
    corners = [[-1, 2, 2], [-1, 2, 0], [-1, 0, 0], [-1, 0, 2]]
    ys = equal_row_ys(0, 2, 4)
    assert ys == pytest.approx([2.0, 1.5, 1.0, 0.5, 0.0])
    mesh = make_layer_mesh(1, corners, n_layers=4, cols=4)
    assert mesh["rows"] == 4
    assert mesh["row_ys"] == pytest.approx([2.0, 1.5, 1.0, 0.5, 0.0])
    z0 = mesh["vertices"][vert_index(4, 4, 1, 0)][2]
    z1 = mesh["vertices"][vert_index(4, 4, 1, 1)][2]
    z2 = mesh["vertices"][vert_index(4, 4, 1, 2)][2]
    assert (z1 - z0) == pytest.approx(z2 - z1)


def test_move_one_row_leaves_others():
    corners = [[-1, 2, 2], [-1, 2, 0], [-1, 0, 0], [-1, 0, 2]]
    mesh = make_layer_mesh(1, corners, n_layers=4, cols=4)
    moved = move_layer_row(mesh, corners, 1, 1.2)
    assert moved["row_ys"][1] == pytest.approx(1.2)
    assert moved["row_ys"][2] == pytest.approx(1.0)
    assert moved["row_ys"][3] == pytest.approx(0.5)
    again = move_layer_row(moved, corners, 2, 0.7)
    assert again["row_ys"][1] == pytest.approx(1.2)
    assert again["row_ys"][2] == pytest.approx(0.7)


def test_mesh_from_custom_row_ys():
    corners = [[-1, 2, 2], [-1, 2, 0], [-1, 0, 0], [-1, 0, 2]]
    mesh = mesh_from_row_ys(1, corners, [2.0, 1.6, 1.1, 0.4, 0.0], cols=3)
    assert mesh["rows"] == 4
    assert mesh["cols"] == 3
    assert mesh["vertices"][vert_index(4, 3, 1, 0)][1] == pytest.approx(1.6)
    assert wall_y_span(corners) == (0.0, 2.0)
