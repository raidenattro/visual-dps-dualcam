"""层高真值应进入反解残差，而不是解完再往墙上贴。"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dualcam_geom import default_row_heights, row_ys_from_heights
from scripts.solve_scene import _project, _wall_corners, _wall_line, solve


def test_row_ys_from_heights_snaps_to_base():
    ys = row_ys_from_heights(0.0, 2.0, [0.65, 0.45, 0.45, 0.45])
    assert ys[0] == pytest.approx(2.0)
    assert ys[-1] == pytest.approx(0.0)
    assert ys[1:] == pytest.approx([1.35, 0.90, 0.45, 0.0])
    assert default_row_heights(4, 2.0, 0.45) == pytest.approx([0.65, 0.45, 0.45, 0.45])


def _synth_z(img_w=1280):
    z = np.zeros(8)
    z[0] = (img_w / 2) / math.tan(math.radians(90 / 2))
    z[1] = 0.0
    z[2] = 2.84
    z[3] = -1.56
    z[4] = math.radians(45)
    z[5] = 0.0
    z[6] = 0.0
    z[7] = 0.3
    return z


def test_solve_layer_line_matches_physical_y():
    img_w, img_h = 1280, 720
    z = _synth_z(img_w)
    wall = {"wall_id": 1, "width": 2.2, "height": 2.0, "base": 0.0}
    aisle, sign = 2.0, 1
    cx, cy = img_w / 2.0, img_h / 2.0
    quad = _project(_wall_corners(wall, sign, aisle, z[7]), z, cx, cy).tolist()
    y = 0.45
    uv = _project(_wall_line(wall, sign, aisle, z[7], y), z, cx, cy).tolist()
    cal = {
        "aisle": aisle,
        "prior": {"camH": 2.84, "camDist": 1.56, "pitch": 45, "yaw": 0},
        "walls": [{**wall, "quad": quad}],
        "layer_lines": [{"wall_id": 1, "y": y, "uv": uv}],
    }
    res = solve(cal, img_w, img_h)
    assert res["ok"]
    assert res["resid_px"] < 2.0
    assert res["layer_resid_px"]
    assert max(res["layer_resid_px"]) < 2.0
    # 解出的墙矩形底沿应在 y=0，0.45m 线落在世界系同一高度
    corners = res["walls"][0]["corners"]
    ys = [p[1] for p in corners]
    assert min(ys) == pytest.approx(0.0, abs=0.02)
    assert max(ys) == pytest.approx(2.0, abs=0.02)
