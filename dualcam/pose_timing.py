"""直播 hold / SSE 合并用的时间常量（原 skel3d_smooth 中仍在用的部分）。"""

from __future__ import annotations

# dump 设计周期 40ms；无 ts 时用 frame_idx * DESIGN_DT 估时
DESIGN_DT = 0.04
# 3D hold 最长（原 dump 8 帧 @25fps）
HOLD_SEC = 8 * DESIGN_DT
# event 骨架相对 pose 允许滞后（live_bus 合并）
EVENT_SKELETON_STALE_S = 0.40


def pose_time(pose: dict | None, frame_idx: int = 0) -> float:
    if isinstance(pose, dict):
        t = float(pose.get("ts") or pose.get("captured_at") or 0.0)
        if t > 0:
            return t
    return float(frame_idx) * DESIGN_DT
