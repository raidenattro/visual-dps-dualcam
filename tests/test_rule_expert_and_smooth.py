from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experts.rule_expert import RulePickExpert
from pipeline.runner import PickStatePipeline, load_pipeline_config
from pipeline.smooth import ScalarSmoother, SmoothConfig
from pipeline.types import FrameContext


def test_rule_expert_blocks_fast_standing_without_triple90():
    ex = RulePickExpert({})
    score, detail = ex.score(
        {
            "ankle_max_speed_norm": 0.2,
            "arm_torso_angle_max": 10.0,
            "elbow_angle_mean": 10.0,
            "wrist_elevation_angle_max": 10.0,
            "shoulder_hip_knee_angle_min": 160.0,
        }
    )
    assert score == 0.0
    assert detail["block"] is True


def test_rule_expert_allows_triple90_exemption():
    ex = RulePickExpert({})
    score, detail = ex.score(
        {
            "ankle_max_speed_norm": 0.2,
            "arm_torso_angle_max": 100.0,
            "elbow_angle_mean": 160.0,
            "wrist_elevation_angle_max": 70.0,
            "shoulder_hip_knee_angle_min": 160.0,
        }
    )
    assert score == 1.0
    assert detail["triple90"] is True
    assert detail["block"] is False


def test_ema_smooth_reduces_spike():
    s = ScalarSmoother(SmoothConfig(enabled=True, window_frames=5, method="ema", ema_alpha=0.4))
    assert s.update(0.0) == 0.0
    v = s.update(1.0)
    assert v is not None and 0.0 < v < 1.0


def test_pipeline_gates_boxes_when_not_picking():
    cfg = load_pipeline_config(ROOT / "configs" / "pipeline.baseline_rule_expert.json")
    pipe = PickStatePipeline(cfg)
    ctx = FrameContext(record_id="x", frame_idx=1)
    # 高速站立无 triple90 → 非拣货 → 清空 box
    out = pipe.process_frame(
        ctx,
        feature_rows=[
            {
                "person_track_id": "1",
                "ankle_max_speed_norm": 0.2,
                "arm_torso_angle_max": 10.0,
                "elbow_angle_mean": 10.0,
                "wrist_elevation_angle_max": 10.0,
                "shoulder_hip_knee_angle_min": 160.0,
            }
        ],
        provisional_box_hits=["Box_1"],
    )
    assert out.box_hits == []
    assert out.pick_decisions[0].is_picking is False
