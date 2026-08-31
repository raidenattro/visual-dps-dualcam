"""消费 pose stream：按巷道对齐 L/R，再走 DualcamProcessor。"""

from __future__ import annotations

import logging
import time

from services.aisle_store import grouped_cameras, load_aisle
from services.event_engine.dualcam_processor import DualcamProcessor
from services.event_engine.sharding import owns_aisle
from services.event_engine.worker import EventRedisWorker

logger = logging.getLogger(__name__)

# 15fps 约 67ms/帧；窗过宽会配错人
PAIR_WINDOW_SEC = 0.12


class DualcamRedisWorker(EventRedisWorker):
    """同一 aisle_id 的 L/R 必须落在本 worker 的 shard 上。"""

    def __init__(self, app_config: dict, callback_reporter=None):
        super().__init__(app_config, callback_reporter=callback_reporter)
        self._aisle_proc: dict[str, DualcamProcessor] = {}
        self._aisle_mtime: dict[str, float] = {}
        self._pending: dict[str, dict[str, tuple[float, dict]]] = {}

    def _json_root(self) -> str:
        return self._json_dir

    def _get_processor(self, camera_id: str, infer_w: int, infer_h: int):
        """单路不再建 CollisionProcessor；成组后走 aisle processor。"""
        return None

    def _aisle_processor(self, aisle_id: str) -> DualcamProcessor | None:
        import os

        path = os.path.join(self._json_root(), "aisles", f"{aisle_id}.json")
        mtime = os.path.getmtime(path) if os.path.isfile(path) else 0.0
        proc = self._aisle_proc.get(aisle_id)
        if proc is not None and self._aisle_mtime.get(aisle_id) == mtime:
            return proc
        data = load_aisle(aisle_id, self._json_root())
        if not data:
            return None
        proc = DualcamProcessor(data)
        self._aisle_proc[aisle_id] = proc
        self._aisle_mtime[aisle_id] = mtime
        return proc

    def _owns_aisle(self, aisle_id: str) -> bool:
        return owns_aisle(aisle_id)

    async def _handle_pose_payload(self, payload: str) -> None:
        import asyncio
        import json

        from services.box_identity import parse_collision_token
        from services.event_bus import publish_event_frame
        from services.pipeline_log import log_pipeline_stage

        try:
            pose = json.loads(payload)
        except json.JSONDecodeError:
            return
        if not isinstance(pose, dict) or pose.get("kind") != "pose":
            return

        camera_id = str(pose.get("camera_id") or "").strip()
        if not camera_id:
            return

        groups = grouped_cameras(self._json_root())
        g = groups.get(camera_id)
        if not g:
            logger.warning("dualcam worker: 未成组相机丢弃 pose camera=%s", camera_id)
            return
        aisle_id = g["aisle_id"]
        role = g["role"]
        if not self._owns_aisle(aisle_id):
            return

        now = float(pose.get("ts") or time.time())
        bucket = self._pending.setdefault(aisle_id, {})
        bucket[role] = (now, pose)

        # 丢掉过期半帧
        stale = [k for k, (t, _) in bucket.items() if now - t > PAIR_WINDOW_SEC * 3]
        for k in stale:
            bucket.pop(k, None)

        if "L" not in bucket or "R" not in bucket:
            return
        t_l, pose_l = bucket["L"]
        t_r, pose_r = bucket["R"]
        if abs(t_l - t_r) > PAIR_WINDOW_SEC:
            return

        bucket.pop("L", None)
        bucket.pop("R", None)

        proc = self._aisle_processor(aisle_id)
        if proc is None or not proc.ready():
            return

        started = time.monotonic()
        result = await asyncio.to_thread(proc.process_pair, pose_l, pose_r)
        worker_ms = round((time.monotonic() - started) * 1000.0, 1)
        frame_idx = int(result.get("frame_idx") or 0)
        collisions = result.get("collisions") or []
        alarm_collisions = result.get("alarm_collisions") or []

        log_pipeline_stage(
            "worker_done",
            camera_id=aisle_id,
            frame_idx=frame_idx,
            worker_ms=worker_ms,
            hits=len(collisions),
            alarms=len(alarm_collisions),
        )

        for cid in (proc.cam_l, proc.cam_r):
            if not cid:
                continue
            await asyncio.to_thread(
                publish_event_frame,
                cid,
                frame_idx=frame_idx,
                collisions=collisions,
                alarm_collisions=alarm_collisions,
                skeletons=result.get("skeletons"),
            )
            if self.callback_reporter and alarm_collisions and cid == proc.cam_l:
                upload_tag = f"infer_{cid}"
                video_time_sec = frame_idx / max(self._video_fps, 1.0)
                for collision in alarm_collisions:
                    shelf_code, box_id = parse_collision_token(collision)
                    if not box_id:
                        continue
                    self.callback_reporter.enqueue_pick_finished(
                        box_id=box_id,
                        frame_idx=frame_idx,
                        video_time_sec=video_time_sec,
                        upload_tag=upload_tag,
                        shelf_code=shelf_code or None,
                    )
