"""门控配置：开关关闭时不影响；打开时按阈值过滤。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experts.action_gate import ActionGate
from pipeline.runner import PickStatePipeline


def test_action_gate_disabled_always_allows():
    g = ActionGate({"enabled": False})
    ok, p = g.allow([0.0] * 10)
    assert ok and p == 1.0


def test_box_gate_reads_config():
    pipe = PickStatePipeline(
        {
            "pair_state": {
                "enabled": True,
                "linear_expert": {
                    "model": {
                        "feature_keys": ["depth_ratio"],
                        "scaler_mean": [0.0],
                        "scaler_scale": [1.0],
                        "coef": [0.0],
                        "intercept": 0.0,
                    }
                },
                "box_gate": {
                    "enabled": True,
                    "depth_ratio_min": 0.25,
                    "center_dist_max": 1.5,
                },
            },
            "box_trigger": {"wrist_score_min": 0.15},
            "alarm": {"min_consecutive_frames": 1},
        }
    )
    assert pipe.box_gate_enabled
    assert pipe.box_depth_min == 0.25
    assert pipe.box_center_max == 1.5
    assert not pipe.action_gate.enabled
