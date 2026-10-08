"""单路时间对比：Δ 换算帧数，停住才算，缺对照帧不算。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dwell_single import build_dwell, dwell_on, lag_frames, wrist_shift  # noqa: E402


def _person(uv, score=0.9):
    k = np.zeros((1, 17, 2), np.float32)
    s = np.zeros((1, 17), np.float32)
    s[0, :] = score
    k[0, 5] = uv
    k[0, 6] = (uv[0] + 40, uv[1])
    k[0, 11] = (uv[0], uv[1] + 80)
    k[0, 12] = (uv[0] + 40, uv[1] + 80)
    k[0, 9] = uv
    k[0, 10] = (uv[0] + 30, uv[1])
    return k, s


def test_lag_matches_daily_25fps_and_15fps():
    assert lag_frames(25, 0.24) == 6
    assert lag_frames(15, 0.24) == 4


def test_still_wrist_is_on_and_missing_past_is_not():
    assert wrist_shift([10, 10], [12, 11]) == np.hypot(2, 1)
    assert wrist_shift([10, 10], None) is None
    assert dwell_on(5.0, 20.0)
    assert not dwell_on(21.0, 20.0)
    assert not dwell_on(None, 20.0)

    frames = []
    # 15fps，Δ=0.24s → 提前 4 帧。前 4 帧没有对照。
    for i in range(8):
        uv = (100.0, 200.0) if i < 6 else (100.0 + 50.0, 200.0)
        k, s = _person(uv)
        frames.append({"i": i, "t": i / 15.0, "k": k, "s": s})
    built, summary = build_dwell(frames, fps=15.0, delta_s=0.24, tau_px=20.0)
    # i=4 对照 i=0，都停在 (100,200)，左腕位移 0
    w = built[4]["people"][0]["w"]["9"]
    assert w["d"] == 0.0 and w["on"]
    assert "u0" in w
    # i=3 还没有 4 帧前的点
    assert built[3]["people"][0]["w"]["9"]["d"] is None
    assert not built[3]["people"][0]["w"]["9"]["on"]
    # i=6 对照 i=2：从 (100,200) 到 (150,200)，超过 20px
    moved = built[6]["people"][0]["w"]["9"]
    assert moved["d"] == 50.0 and not moved["on"]
    assert summary["lag_frames"] == 4
    assert summary["unit"] == "px"
