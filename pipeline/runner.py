"""按 configs/*.json 装配并跑完整 record：FeatureBank → PickState → BoxTrigger → Alarm。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experts.linear_expert import LinearPickExpert
from experts.rule_expert import RulePickExpert
from features.bank import FeatureBank
from pipeline.alarm import AlarmTracker
from pipeline.box_trigger import BoxTrigger
from pipeline.smooth import ScalarSmoother, SmoothConfig
from pipeline.types import FrameContext, PickDecision, PipelineResult


def load_pipeline_config(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_scorer(name: str, cfg: dict[str, Any]):
    if name == "rule_expert":
        return RulePickExpert(cfg.get("rule_expert") or {})
    if name == "linear_expert":
        return LinearPickExpert(cfg.get("linear_expert") or {})
    raise ValueError(f"未知 scorer: {name}")


class PickStatePipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        pick_cfg = config.get("pick_state") or {}
        box_cfg = config.get("box_trigger") or {}
        alarm_cfg = config.get("alarm") or {}

        self.threshold = float(pick_cfg.get("threshold", 0.5))
        self.require_pick = bool(box_cfg.get("require_pick_state", True))
        self.wrist_score_min = float(box_cfg.get("wrist_score_min", 0.3))
        self.pose_frame_interval = int(config.get("pose_frame_interval") or 2)

        self._score_smooth_cfg = SmoothConfig(**(pick_cfg.get("score_smooth") or {}))
        self._score_smoothers: dict[str, ScalarSmoother] = {}
        self.scorer = _build_scorer(str(pick_cfg.get("scorer") or "rule_expert"), pick_cfg)
        self.alarm = AlarmTracker(
            min_consecutive_frames=int(alarm_cfg.get("min_consecutive_frames", 3)),
            cooldown_frames=int(alarm_cfg.get("cooldown_frames", 0)),
        )

    def reset_session(self) -> None:
        self._score_smoothers.clear()
        self.scorer.reset()
        self.alarm.reset()

    def _smoother(self, track_id: str) -> ScalarSmoother:
        if track_id not in self._score_smoothers:
            self._score_smoothers[track_id] = ScalarSmoother(self._score_smooth_cfg)
        return self._score_smoothers[track_id]

    def _decide(self, row: dict[str, Any]) -> PickDecision:
        track_id = str(row.get("person_track_id") or "0")
        raw_score, detail = self.scorer.score(row)
        smooth = self._smoother(track_id).update(raw_score)
        smooth_v = float(smooth if smooth is not None else raw_score)
        return PickDecision(
            person_track_id=track_id,
            score_raw=float(raw_score),
            score_smooth=smooth_v,
            is_picking=smooth_v >= self.threshold,
            expert=self.scorer.name,
            detail=detail,
        )

    def process_frame(
        self,
        ctx: FrameContext,
        *,
        feature_rows: list[dict[str, Any]] | None = None,
        box_trigger: BoxTrigger | None = None,
        provisional_box_hits: list[str] | None = None,
    ) -> PipelineResult:
        """按人判定拣货态；非拣货态的人不贡献碰撞（对齐 DPS blocked → continue）。"""
        decisions: list[PickDecision] = []
        tokens: set[str] = set()
        hit_detail: list[dict[str, Any]] = []

        for row in feature_rows or []:
            decision = self._decide(row)
            decisions.append(decision)
            if self.require_pick and not decision.is_picking:
                continue
            person = row.get("_person")
            if box_trigger is not None and isinstance(person, dict):
                for hit in box_trigger.hits_for_person(person):
                    tokens.add(hit["token"])
                    hit_detail.append(hit)

        if box_trigger is None:
            keep = provisional_box_hits or []
            if self.require_pick and not any(d.is_picking for d in decisions):
                keep = []
            tokens.update(keep)

        collisions = sorted(tokens)
        alarms = self.alarm.step(collisions, ctx.frame_idx)

        return PipelineResult(
            frame_idx=ctx.frame_idx,
            pick_decisions=decisions,
            box_hits=collisions,
            alarm_hits=alarms,
            debug={"hits": hit_detail},
        )

    def run_record(
        self,
        record,
        *,
        frame_indices: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """跑完一条 record，返回与 collector 评估器兼容的 upload 行。"""
        self.reset_session()
        bank = FeatureBank(
            infer_width=int(record.meta.get("infer_width") or record.ref.infer_width or 1),
            infer_height=int(record.meta.get("infer_height") or record.ref.infer_height or 1),
            video_fps=float(record.fps or 15.0),
        )
        trigger = BoxTrigger(record.boxes, wrist_score_min=self.wrist_score_min)

        by_key: dict[int, dict[str, Any]] = {}
        for frame in record.frames:
            key = int(frame.get("source_frame_idx") or frame.get("frame_idx") or 0)
            by_key[key] = frame

        if frame_indices is not None:
            keys = sorted(frame_indices)
        else:
            keys = sorted(by_key)

        out: list[dict[str, Any]] = []
        for export_key in keys:
            # 无检测的帧仍需产出空行，否则告警连续帧计数与 baseline 不可比
            frame = by_key.get(export_key) or {
                "frame_idx": export_key,
                "source_frame_idx": export_key,
                "timestamp_sec": 0.0,
                "persons": [],
            }
            rows = bank.rows_for_frame(frame)
            ctx = FrameContext(
                record_id=record.ref.record_id,
                frame_idx=export_key,
                camera_slug=record.ref.camera_slug,
            )
            result = self.process_frame(ctx, feature_rows=rows, box_trigger=trigger)

            probs = [d.score_smooth for d in result.pick_decisions]
            out.append(
                {
                    "record_id": record.ref.record_id,
                    "frame_idx": export_key,
                    "is_picking": bool(result.alarm_hits),
                    "picking_prob": round(max(probs), 4) if probs else None,
                    "predicted_box_tokens": [],
                    "rule_collisions": result.box_hits,
                    "rule_alarm_collisions": result.alarm_hits,
                }
            )

        return out
