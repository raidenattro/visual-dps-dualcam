"""事件识别引擎（碰撞等），与姿态推理进程解耦。"""

from services.event_engine.collision import CollisionProcessor

__all__ = ["CollisionProcessor"]
