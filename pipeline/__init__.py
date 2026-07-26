"""可装配流水线：FeatureBank → PickStateScorer → BoxTrigger → Alarm。"""

from pipeline.types import FeatureVector, FrameContext, PickDecision, PipelineResult

__all__ = ["FeatureVector", "FrameContext", "PickDecision", "PipelineResult"]
