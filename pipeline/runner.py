"""按 configs/*.json 装配并跑通一帧（骨架；特征/碰撞细节后续补齐）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experts.rule_expert import RulePickExpert
from pipeline.smooth import ScalarSmoother, SmoothConfig
from pipeline.types import FrameContext, PickDecision, PipelineResult


def load_pipeline_config(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class PickStatePipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        pick_cfg = config.get("pick_state") or {}
        self.threshold = float(pick_cfg.get("threshold", 0.5))
        self.require_pick = bool((config.get("box_trigger") or {}).get("require_pick_state", True))

        score_smooth = SmoothConfig(**(pick_cfg.get("score_smooth") or {}))
        self._score_smoothers: dict[str, ScalarSmoother] = {}
        self._score_smooth_cfg = score_smooth

        scorer_name = str(pick_cfg.get("scorer") or "rule_expert")
        if scorer_name == "rule_expert":
            self.scorer = RulePickExpert(pick_cfg.get("rule_expert") or {})
        else:
            raise ValueError(f"未知 scorer: {scorer_name}（后续可挂 linear / hgb）")

    def reset_session(self) -> None:
        self._score_smoothers.clear()
        self.scorer.reset()

    def _smoother(self, track_id: str) -> ScalarSmoother:
        if track_id not in self._score_smoothers:
            self._score_smoothers[track_id] = ScalarSmoother(self._score_smooth_cfg)
        return self._score_smoothers[track_id]

    def process_frame(
        self,
        ctx: FrameContext,
        *,
        feature_rows: list[dict[str, Any]] | None = None,
        provisional_box_hits: list[str] | None = None,
    ) -> PipelineResult:
        """feature_rows: 每行含 person_track_id + 特征字段；provisional_box_hits 为腕点碰撞候选。"""
        decisions: list[PickDecision] = []
        any_picking = False
        for row in feature_rows or []:
            track_id = str(row.get("person_track_id") or "0")
            raw_score, detail = self.scorer.score(row)
            smooth = self._smoother(track_id).update(raw_score)
            smooth_v = float(smooth if smooth is not None else raw_score)
            is_picking = smooth_v >= self.threshold
            any_picking = any_picking or is_picking
            decisions.append(
                PickDecision(
                    person_track_id=track_id,
                    score_raw=float(raw_score),
                    score_smooth=smooth_v,
                    is_picking=is_picking,
                    expert=self.scorer.name,
                    detail=detail,
                )
            )

        box_hits = list(provisional_box_hits or [])
        if self.require_pick and not any_picking:
            box_hits = []

        return PipelineResult(
            frame_idx=ctx.frame_idx,
            pick_decisions=decisions,
            box_hits=box_hits,
            alarm_hits=[],  # 时序告警后续接
            debug={"require_pick_state": self.require_pick, "any_picking": any_picking},
        )
