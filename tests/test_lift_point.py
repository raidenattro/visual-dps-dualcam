"""高置信度路建 3D：单路不丢腕，缝大时钉在高分射线上。"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dualcam_lift import KPT_MIN, lift_point, load_cams, point_on_ray, ray

CALIB = ROOT / "output/calib/dual_1-3.json"


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
