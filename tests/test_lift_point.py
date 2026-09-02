"""高置信度路建 3D：单路不丢腕，缝大时钉在高分射线上。"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dualcam.lift import (
    ARM_CONF_JOINTS,
    ARM_CONF_POWER,
    CONF_POWER,
    KPT_MIN,
    LELB,
    RELB,
    conf_weights,
    lift_point,
    point_on_ray,
    ray,
    triangulate_ends,
)

CALIB = ROOT / "fixtures/dual_1-3.json"


def load_cams():
    import json
    from dualcam.lift import wall_plane_from_solved
    data = json.loads(CALIB.read_text(encoding="utf-8"))
    sol = data["solved"]
    return sol["cameras"], wall_plane_from_solved(sol, 1), sol


def _cams():
    if not CALIB.is_file():
        pytest.skip("缺少 dual_1-3.json")
    return load_cams()


def test_point_on_ray_stays_on_ray():
    cams, _, _ = _cams()
    uv = np.array([640.0, 360.0])
    ref = np.array([0.0, 1.2, 1.0])
    p = point_on_ray(uv, cams["L"], ref)
    C, d = ray(uv, cams["L"])
    v = p - C
    assert float(np.linalg.norm(np.cross(v, d))) < 1e-6


def test_only_high_conf_keeps_wrist_with_prev_depth():
    cams, plane, _ = _cams()
    uv_l = np.array([700.0, 360.0])
    uv_r = np.array([600.0, 240.0])
    prev = np.array([-0.2, 1.15, 1.0])
    p, _g, src = lift_point(uv_l, 0.80, uv_r, 0.10, cams, plane, prev)
    assert p is not None
    assert src == "Lhold"
    C, d = ray(uv_l, cams["L"])
    assert float(np.linalg.norm(np.cross(p - C, d))) < 1e-6


def test_only_high_conf_without_prev_is_mono_not_contact():
    cams, plane, _ = _cams()
    uv_l = np.array([700.0, 360.0])
    uv_r = np.array([600.0, 240.0])
    p, _g, src = lift_point(uv_l, 0.80, uv_r, 0.10, cams, plane, None)
    assert src == "Lmono"
    assert p is not None


def test_both_below_min_is_missing():
    cams, plane, _ = _cams()
    p, _g, src = lift_point(
        np.array([700.0, 360.0]), KPT_MIN - 0.05,
        np.array([600.0, 240.0]), KPT_MIN - 0.05,
        cams, plane, np.array([-0.2, 1.15, 1.0]),
    )
    assert p is None and src is None


def test_close_scores_weight_blend_not_midpoint():
    """缝小且两路都过门槛：3D 按分数加权，不再 50/50 或赢者通吃。"""
    cams, plane, _ = _cams()
    uv_l = np.array([746.3, 236.5])
    uv_r = np.array([580.0, 360.0])  # 故意偏一点，两路交点分开
    p1, p2, g = triangulate_ends(uv_l, uv_r, cams)
    if g > 0.20 or float(np.linalg.norm(p1 - p2)) < 0.02:
        pytest.skip("该像素对无法拉开交点")
    mid = 0.5 * (p1 + p2)
    p_eq, _, src_eq = lift_point(uv_l, 0.70, uv_r, 0.70, cams, plane, None)
    assert src_eq == "stereo"
    assert float(np.linalg.norm(p_eq - mid)) < 1e-6
    p_l, _, src_l = lift_point(uv_l, 0.90, uv_r, 0.40, cams, plane, None)
    assert src_l == "L"
    assert float(np.linalg.norm(p_l - p1)) < float(np.linalg.norm(p_l - p2))
    assert float(np.linalg.norm(p_l - p1)) < float(np.linalg.norm(mid - p1)) - 1e-9
    p_r, _, src_r = lift_point(uv_l, 0.40, uv_r, 0.90, cams, plane, None)
    assert src_r == "R"
    assert float(np.linalg.norm(p_r - p2)) < float(np.linalg.norm(p_r - p1))
    linear = (0.90 * p1 + 0.40 * p2) / 1.30
    assert float(np.linalg.norm(p_l - p1)) < float(np.linalg.norm(linear - p1)) - 1e-9


def test_conf_weights_square_weakens_low_score():
    """0.40 对 0.90：线性约 31%，平方约 16%。"""
    wl, wr = conf_weights(0.90, 0.40, CONF_POWER)
    assert wr / (wl + wr) < 0.40 / 1.30 - 0.05
    w_eq_l, w_eq_r = conf_weights(0.70, 0.70, CONF_POWER)
    assert abs(w_eq_l - w_eq_r) < 1e-12
    assert ARM_CONF_POWER == 2.5
    assert {LELB, RELB} <= ARM_CONF_JOINTS
