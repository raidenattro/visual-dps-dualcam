"""拼接片 2D 坐标归一化：混有邻路半幅点时不能整帧减 1280。"""

from __future__ import annotations

import numpy as np

from scripts.dualcam_lift import (
    HALF_W,
    normalize_pose_pack_for_calib,
    pose_in_calib_frame,
    shift_stitch_uv,
)
import numpy as np


def test_shift_stitch_uv_per_keypoint_only():
    k = np.array([[1203.0, 852.0], [1439.0, 800.0]], float)
    out = shift_stitch_uv(k)
    np.testing.assert_allclose(out[0], [1203.0, 852.0])
    np.testing.assert_allclose(out[1], [159.0, 800.0])


def test_normalize_mixed_left_person():
    pack = [
        {
            "i": 1000,
            "t": 0.0,
            "L": {
                "k": [
                    np.array([[1203.0, 852.0], [1439.0, 800.0]], float),
                ],
                "s": [np.ones(2)],
            },
            "R": {"k": [np.array([[400.0, 400.0]], float)], "s": [np.ones(1)]},
        }
    ]
    normalize_pose_pack_for_calib(pack)
    kl = pack[0]["L"]["k"][0]
    assert float(kl[0, 0]) > 1000
    assert float(kl[1, 0]) < HALF_W


def test_pose_in_calib_frame_rejects_stitch_spill():
    k = np.zeros((17, 2), float)
    s = np.zeros(17, float)
    k[5] = [600, 400]
    k[6] = [1300, 400]  # 邻路半幅，归一化前
    s[5] = s[6] = 0.9
    assert not pose_in_calib_frame(k, s)
