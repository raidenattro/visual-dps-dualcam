"""左右路多人贪心匹配：每人只用一次。"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dualcam_lift import load_cams, pick_pair, pick_pairs

NPZ = ROOT / "output/dualcam/poses_5fps.npz"
CALIB = ROOT / "output/calib/dual_1-3.json"


def test_pick_pairs_unique_and_at_least_best():
    if not NPZ.is_file() or not CALIB.is_file():
        pytest.skip("缺少 poses 或 calib")
    cams, _, _ = load_cams()
    pack = np.load(NPZ, allow_pickle=True)["frames"]
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
        if one:
            assert pairs[0] == one
        found = True
        break
    if not found:
        pytest.skip("npz 里没有双人以上的帧")
