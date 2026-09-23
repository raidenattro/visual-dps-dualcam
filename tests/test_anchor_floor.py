"""3D 骨架贴地：仅平移 Y，XZ 与相对骨长不变。"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.dualcam_lift import WORLD_FLOOR_Y, anchor_person_to_floor, anchor_xyz_list_to_floor


def test_anchor_person_to_floor_shifts_y_only():
    raw3 = [None] * 17
    xyz = [None] * 17
    raw3[15] = np.array([0.5, 1.2, 1.0])
    raw3[16] = np.array([0.6, 1.0, 1.0])
    xyz[15] = [0.5, 1.2, 1.0]
    xyz[16] = [0.6, 1.0, 1.0]
    anchor_person_to_floor(raw3, xyz, floor_y=WORLD_FLOOR_Y)
    assert float(raw3[16][1]) == WORLD_FLOOR_Y
    assert float(raw3[15][1]) == pytest.approx(0.2)
    assert float(raw3[15][0]) == 0.5


def test_anchor_xyz_list_after_smooth():
    xyz = [None] * 17
    xyz[16] = [0.6, 0.5, 1.0]
    anchor_xyz_list_to_floor(xyz)
    assert xyz[16][1] == WORLD_FLOOR_Y
