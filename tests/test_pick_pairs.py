"""左右路多人匹配：NMS、续帧、巷道内、3D 去重。"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dualcam_lift import (
    _torso_xy,
    in_aisle,
    load_cams,
    nms_indices,
    pick_pair,
    pick_pairs,
)

NPZ = ROOT / "output/dualcam/poses_5fps.npz"
CALIB = ROOT / "output/calib/dual_1-3.json"


def _need():
    if not NPZ.is_file() or not CALIB.is_file():
        pytest.skip("缺少 poses 或 calib")
    return load_cams()[0], np.load(NPZ, allow_pickle=True)["frames"]


def _at(pack, t0: float):
    ts = np.array([float(f["t"]) for f in pack])
    return pack[int(np.argmin(np.abs(ts - t0)))]


def test_pick_pairs_unique_and_at_least_best():
    cams, pack = _need()
    found = False
    for fr in pack:
        n_l = len(fr["L"]["k"])
        n_r = len(fr["R"]["k"])
        if n_l < 2 or n_r < 2:
            continue
        pairs = pick_pairs(fr["L"], fr["R"], cams)
        ls = [p[0] for p in pairs]
        rs = [p[1] for p in pairs]
        assert len(ls) == len(set(ls))
        assert len(rs) == len(set(rs))
        assert len(pairs) <= min(n_l, n_r)
        one = pick_pair(fr["L"], fr["R"], cams)
        if one and pairs:
            assert one in pairs
        found = True
        break
    if not found:
        pytest.skip("npz 里没有双人以上的帧")


def test_nms_drops_stacked_double_detect():
    """1:29 左右都是同一人叠了两个框，NMS 后只留一个。"""
    _, pack = _need()
    fr = _at(pack, 89.20)
    assert len(fr["L"]["k"]) >= 2 and len(fr["R"]["k"]) >= 2
    assert len(nms_indices(fr["L"]["k"], fr["L"]["s"])) == 1
    assert len(nms_indices(fr["R"]["k"], fr["R"]["s"])) == 1


def test_duplicate_3d_not_emitted_at_129():
    cams, pack = _need()
    fr = _at(pack, 89.20)
    pairs = pick_pairs(fr["L"], fr["R"], cams)
    assert len(pairs) <= 1


def test_prefer_keeps_same_person_when_right_has_one():
    """1:24 左路两人右路一人：无续帧会跳到另一个左框，有续帧应钉住上一帧。"""
    cams, pack = _need()
    a, b = _at(pack, 84.16), _at(pack, 84.20)
    first = pick_pairs(a["L"], a["R"], cams)
    assert first
    i, j, _ = first[0]
    prefer = [
        (
            _torso_xy(a["L"]["k"][i], a["L"]["s"][i]),
            _torso_xy(a["R"]["k"][j], a["R"]["s"][j]),
        )
    ]
    unlocked = pick_pairs(b["L"], b["R"], cams)
    locked = pick_pairs(b["L"], b["R"], cams, prefer=prefer)
    assert locked
    assert locked[0][0] == 0
    if unlocked:
        assert unlocked[0][0] == 1


def test_in_aisle_rejects_camera_ghost_and_ceiling():
    assert in_aisle(np.array([-0.2, 1.15, 1.0]))
    assert not in_aisle(np.array([-0.5, 1.0, -0.3]))  # 镜头前
    assert not in_aisle(np.array([-0.1, 2.02, 2.0]))  # 棚顶误检
    assert not in_aisle(np.array([-0.08, 1.70, 2.14]))
