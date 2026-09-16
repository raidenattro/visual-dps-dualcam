"""方案 A：lift_person17 躯干锚点 + 四肢共享深度。"""

from __future__ import annotations

import numpy as np

from dualcam.lift import LELB, LSHO, LWRIST, lift_person17


def _sym_cams():
    return {
        "L": {
            "f": 900.0,
            "cx": 640.0,
            "cy": 360.0,
            "C": np.array([-0.78, 1.4, 0.0]),
            "right": np.array([0.0, 0.0, 1.0]),
            "down": np.array([0.0, -1.0, 0.0]),
            "fwd": np.array([1.0, 0.0, 0.0]),
        },
        "R": {
            "f": 900.0,
            "cx": 640.0,
            "cy": 360.0,
            "C": np.array([0.78, 1.4, 0.0]),
            "right": np.array([0.0, 0.0, -1.0]),
            "down": np.array([0.0, -1.0, 0.0]),
            "fwd": np.array([-1.0, 0.0, 0.0]),
        },
    }


def _person_at(x: float, y: float, z: float, cams: dict) -> tuple[np.ndarray, np.ndarray]:
    """在 world 点生成左右一致 2D（肩在 x,y,z）。"""
    kl = np.zeros((17, 2), float)
    kr = np.zeros((17, 2), float)
    sl = np.zeros(17, float)
    sr = np.zeros(17, float)
    from dualcam.geom import project_pix

    p = np.array([x, y, z], float)
    ul = project_pix(p, cams["L"])
    ur = project_pix(p, cams["R"])
    assert ul and ur
    for ji in (LSHO, LELB, LWRIST):
        kl[ji] = ul
        kr[ji] = ur
        sl[ji] = 0.9
        sr[ji] = 0.9
    return kl, sl, kr, sr


def test_lift_person17_limb_shares_shoulder_depth():
    cams = _sym_cams()
    kl, sl, kr, sr = _person_at(0.0, 1.35, 0.8, cams)
    xyz, srcs = lift_person17(
        kl, sl, kr, sr, cams, None, lambda _j: None, has_r=True,
    )
    assert xyz[LSHO] is not None
    assert xyz[LWRIST] is not None
    assert srcs[LWRIST] in ("stereo", "L", "R")
    sh = np.asarray(xyz[LSHO], float)
    wr = np.asarray(xyz[LWRIST], float)
    assert float(np.linalg.norm(wr - sh)) < 1.0


def test_lift_person17_no_lmono_on_wrist_when_torso_ready():
    """有躯干锚点时，腕不应再被 ray_plane 打到墙平面（src 不含 mono）。"""
    cams = _sym_cams()
    kl, sl, kr, sr = _person_at(-0.2, 1.35, 0.9, cams)
    # 故意让右路腕分数为 0，走单路 hold 深度
    sr[LWRIST] = 0.0
    kl[LWRIST] = kl[LSHO] + np.array([40.0, 30.0])
    sl[LWRIST] = 0.85
    xyz, srcs = lift_person17(
        kl, sl, kr, sr, cams, {"p0": np.zeros(3), "n": np.array([1.0, 0.0, 0.0])},
        lambda _j: None,
        has_r=True,
    )
    assert xyz[LSHO] is not None
    assert srcs[LWRIST] in ("Lhold", "L", None)
    assert srcs[LWRIST] not in ("Lmono", "Rmono")


def test_lift_person17_skips_face_joints_in_skip_set():
    cams = _sym_cams()
    kl, sl, kr, sr = _person_at(0.1, 1.35, 0.75, cams)
    skip = frozenset({0, 1, 2, 3, 4})
    xyz, srcs = lift_person17(
        kl, sl, kr, sr, cams, None, lambda _j: None, has_r=True, skip_joint=skip,
    )
    for ji in range(5):
        assert xyz[ji] is None
        assert srcs[ji] is None
    assert xyz[5] is not None


def test_lift_person17_torso_before_limb_order():
    cams = _sym_cams()
    kl, sl, kr, sr = _person_at(0.1, 1.35, 0.75, cams)
    xyz, srcs = lift_person17(
        kl, sl, kr, sr, cams, None, lambda _j: None, has_r=True,
    )
    assert xyz[LSHO] is not None
    assert xyz[LELB] is not None
    assert srcs[LSHO] in ("stereo", "L", "R", "Lhold", "Rhold", "Lmono", "Rmono")
