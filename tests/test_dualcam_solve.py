"""单墙反解必须把相机留在巷道内；双墙 camX 边界不收紧。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dualcam.solve import (
    _bounds,
    _camera_in_aisle,
    _camx_half_limit,
    _project,
    _wall_corners,
    solve,
)

IMG_W, IMG_H = 1280, 720
_WALL = {"width": 3.0, "height": 2.0, "base": 0.2}


def _pose_z(*, cam_x=0.0, cam_h=2.84, cam_z=-1.56, pitch=45.0, yaw=25.0, roll=0.0, fov=90.0):
    f = (IMG_W / 2) / math.tan(math.radians(fov / 2))
    return np.array(
        [
            f,
            cam_x,
            cam_h,
            cam_z,
            math.radians(pitch),
            math.radians(yaw),
            math.radians(roll),
        ],
        float,
    )


def _quad_from_pose(sign: int, aisle: float = 2.0, znear: float = 0.0, **pose):
    pts = _wall_corners(_WALL, sign, aisle, znear)
    uv = _project(pts, _pose_z(**pose), IMG_W / 2, IMG_H / 2)
    assert np.isfinite(uv).all(), uv
    return uv.tolist()


def test_camx_bounds_only_tighten_for_single_wall():
    lo, hi = _bounds(9, IMG_W)
    assert lo[1] == pytest.approx(-2.0)
    assert hi[1] == pytest.approx(2.0)
    lo, hi = _bounds(8, IMG_W, aisle=2.0, single_wall=True)
    limit = _camx_half_limit(2.0)
    assert lo[1] == pytest.approx(-limit)
    assert hi[1] == pytest.approx(limit)
    assert limit < 1.0


def test_camera_in_aisle_rejects_outside_wall():
    assert _camera_in_aisle(0.0, 2.0) is True
    assert _camera_in_aisle(0.8, 2.0) is True
    assert _camera_in_aisle(1.65, 2.0) is False
    assert _camera_in_aisle(-1.65, 2.0) is False


def test_single_wall_solve_keeps_camera_inside_aisle():
    """巷道内相机看墙2：反解 camX 必须仍在 ±aisle/2 内。"""
    quad = _quad_from_pose(sign=1, yaw=25.0, cam_x=0.0)
    res = solve(
        {
            "aisle": 2.0,
            "prior": {"camH": 2.84, "camDist": 1.56, "pitch": 45.0, "yaw": 25.0},
            "walls": [{"wall_id": 2, "quad": quad, **_WALL}],
        },
        IMG_W,
        IMG_H,
    )
    assert res.get("ok"), res
    cam_x = float(res["camera"]["camX"])
    assert abs(cam_x) < 1.0
    assert _camera_in_aisle(cam_x, 2.0)


def test_single_wall_outside_pose_does_not_keep_camera_behind_wall():
    """四角若来自墙外相机，约束后要么失败，要么把相机拉回巷道内。"""
    quad = _quad_from_pose(sign=1, yaw=8.0, cam_x=1.65)
    res = solve(
        {
            "aisle": 2.0,
            "prior": {"camH": 2.84, "camDist": 1.56, "pitch": 45.0, "yaw": 8.0},
            "walls": [{"wall_id": 2, "quad": quad, **_WALL}],
        },
        IMG_W,
        IMG_H,
    )
    if not res.get("ok"):
        assert "巷道内" in str(res.get("error") or "")
        return
    assert abs(float(res["camera"]["camX"])) < 1.0
