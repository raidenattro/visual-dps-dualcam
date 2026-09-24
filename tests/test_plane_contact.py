"""单应腕点碰撞：合成相机投影墙四角 → H 复原墙面坐标；面上点重合，离面越远两路落点越分。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dualcam_geom import project_pix  # noqa: E402
from scripts.plane_contact import PlaneContact, apply_h, homography_to_wall  # noqa: E402

W, H, X = 2.2, 2.0, -1.0
CORNERS = [[X, H, W], [X, H, 0.0], [X, 0.0, 0.0], [X, 0.0, W]]  # 顶远 顶近 底近 底远


def _cam(C, look_at, f=700.0):
    C = np.asarray(C, float)
    fwd = np.asarray(look_at, float) - C
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return {"f": f, "cx": 640.0, "cy": 360.0, "C": C.tolist(), "fwd": fwd.tolist(),
            "right": right.tolist(), "down": down.tolist()}


def _calib():
    # 同侧双机：都在 x≈0 附近、z 为负（巷道起点后退），朝墙中心看
    cams = {"L": _cam([0.0, 3.0, -1.3], [X, 1.0, 1.1]), "R": _cam([0.6, 3.1, -2.1], [X, 1.0, 1.1])}
    views = {}
    for v, cam in cams.items():
        quad = [project_pix(c, cam) for c in CORNERS]
        views[v] = {"walls": [{"wall_id": 1, "width": W, "height": H, "base": 0.0, "quad": quad}]}
    return {
        "aisle": 2.0,
        "views": views,
        "solved": {"ok": True, "cameras": cams,
                   "walls": [{"wall_id": 1, "sign": -1, "z_near": 0.0, "corners": CORNERS}]},
        "slot_meshes": [],
    }, cams


def test_homography_recovers_wall_coords():
    calib, cams = _calib()
    quad = calib["views"]["L"]["walls"][0]["quad"]
    Hm = homography_to_wall(quad, W, H)
    # 面上任意点：世界 → 像素 → H → 墙面 (z, y)，误差 < 1mm
    for z, y in [(0.3, 0.5), (1.1, 1.0), (2.0, 1.8)]:
        uv = project_pix([X, y, z], cams["L"])
        zy = apply_h(Hm, uv)
        assert np.allclose(zy, [z, y], atol=1e-3)


def test_on_plane_point_coincides_off_plane_separates():
    calib, cams = _calib()
    pc = PlaneContact.from_calib(calib, tau=0.15)
    assert pc.wall_ids == [1]
    dists = []
    for off in (0.0, 0.1, 0.3, 0.6):
        p = [X + off, 1.0, 1.1]  # 沿巷道方向离墙 off 米
        uvl, uvr = project_pix(p, cams["L"]), project_pix(p, cams["R"])
        r = pc.wrist(uvl, 0.9, uvr, 0.9)
        assert r is not None
        dists.append(r["dist"])
        if off == 0.0:
            assert r["dist"] < 1e-3
            assert np.allclose(r["zy"], [1.1, 1.0], atol=1e-3)
            assert r["in"]
        else:
            assert r["sgn"] is not None and r["sgn"] > 0  # 悬在巷道里为正
    assert all(b > a for a, b in zip(dists, dists[1:]))  # 分离距离随离墙距离单调增


def test_inside_wall_is_negative_and_low_score_skipped():
    calib, cams = _calib()
    pc = PlaneContact.from_calib(calib)
    p = [X - 0.2, 1.0, 1.1]  # 伸进货架 20cm
    uvl, uvr = project_pix(p, cams["L"]), project_pix(p, cams["R"])
    r = pc.wrist(uvl, 0.9, uvr, 0.9)
    assert r is not None and r["sgn"] < 0
    assert pc.wrist(uvl, 0.1, uvr, 0.9) is None
