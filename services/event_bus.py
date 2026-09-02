"""Redis 事件 overlay 总线（事件 Worker → UI）。"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import redis as sync_redis

logger = logging.getLogger(__name__)

EVENT_CHANNEL_PREFIX = "event:live:"
EVENT_SNAPSHOT_PREFIX = "event:snapshot:"
EVENT_SCHEMA_VERSION = 1
SNAPSHOT_TTL_SEC = max(3, int(os.environ.get("LIVE_SNAPSHOT_TTL_SEC", "10")))
_POOL: sync_redis.ConnectionPool | None = None


def redis_url() -> str:
    from services.live_bus import redis_url as _url

    return _url()


def channel_for(camera_id: str) -> str:
    return f"{EVENT_CHANNEL_PREFIX}{camera_id}"


def snapshot_key_for(camera_id: str) -> str:
    return f"{EVENT_SNAPSHOT_PREFIX}{camera_id}"


def _sync_redis() -> sync_redis.Redis:
    """复用连接池。worker 每帧 L/R 各 publish 一次，禁止每次 from_url+close。"""
    global _POOL
    if _POOL is None:
        _POOL = sync_redis.ConnectionPool.from_url(redis_url(), decode_responses=True)
    return sync_redis.Redis(connection_pool=_POOL)


def _camera_id_from_channel(channel: str) -> str:
    if channel.startswith(EVENT_CHANNEL_PREFIX):
        return channel[len(EVENT_CHANNEL_PREFIX) :]
    return channel


def build_event_frame(
    *,
    camera_id: str,
    frame_idx: int,
    collisions: list,
    alarm_collisions: list,
    skeletons: list | None = None,
    persons_3d: list | None = None,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "schema": EVENT_SCHEMA_VERSION,
        "kind": "event",
        "ts": time.time(),
        "camera_id": str(camera_id),
        "frame_idx": int(frame_idx),
        "collisions": list(collisions),
        "alarm_collisions": list(alarm_collisions),
    }
    if skeletons is not None:
        frame["skeletons"] = list(skeletons)
    if persons_3d is not None:
        frame["persons_3d"] = list(persons_3d)
    return frame


def publish_event_frame(
    camera_id: str,
    *,
    frame_idx: int,
    collisions: list,
    alarm_collisions: list,
    skeletons: list | None = None,
    persons_3d: list | None = None,
) -> bool:
    cid = str(camera_id or "").strip()
    if not cid:
        return False
    frame = build_event_frame(
        camera_id=cid,
        frame_idx=frame_idx,
        collisions=collisions,
        alarm_collisions=alarm_collisions,
        skeletons=skeletons,
        persons_3d=persons_3d,
    )
    payload = json.dumps(frame, ensure_ascii=False, separators=(",", ":"))
    try:
        client = _sync_redis()
        pipe = client.pipeline(transaction=False)
        pipe.set(snapshot_key_for(cid), payload, ex=SNAPSHOT_TTL_SEC)
        pipe.publish(channel_for(cid), payload)
        pipe.execute()
        return True
    except Exception as exc:
        logger.warning("Redis publish_event_frame failed camera=%s: %s", cid, exc)
        global _POOL
        _POOL = None
        return False


def get_event_snapshot(camera_id: str) -> dict[str, Any] | None:
    cid = str(camera_id or "").strip()
    if not cid:
        return None
    try:
        client = _sync_redis()
        raw = client.get(snapshot_key_for(cid))
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("Redis get_event_snapshot failed camera=%s: %s", cid, exc)
        return None


def camera_id_from_channel(channel: str) -> str:
    return _camera_id_from_channel(channel)
