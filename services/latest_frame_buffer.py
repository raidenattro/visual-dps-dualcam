"""线程安全「仅保留最新帧」缓冲，供 RTSP 后台读帧与推理快照消费。"""

from __future__ import annotations

import os
import threading
import time

import numpy as np

_DEFAULT_TTL_SEC = 1.0
_ENV_TTL = "RTSP_FRAME_BUFFER_TTL_SEC"


def frame_buffer_ttl_sec() -> float:
    """最新帧有效时长（秒）；超时后 snapshot 视为无帧。默认 1s，环境变量可配。"""
    raw = os.environ.get(_ENV_TTL, str(_DEFAULT_TTL_SEC)).strip()
    try:
        value = float(raw)
    except ValueError:
        value = _DEFAULT_TTL_SEC
    return max(0.001, value)


class LatestFrameBuffer:
    """后台读帧线程写入；推理侧通过 snapshot() 取副本，互不阻塞解码。"""

    def __init__(self, ttl_sec: float | None = None) -> None:
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._captured_at: float = 0.0
        self._seq: int = 0
        self._ttl_sec = frame_buffer_ttl_sec() if ttl_sec is None else max(0.001, float(ttl_sec))

    def _clear_locked(self) -> None:
        self._latest = None
        self._captured_at = 0.0

    def _is_expired_locked(self, now: float) -> bool:
        if self._latest is None:
            return False
        return (now - self._captured_at) > self._ttl_sec

    def update(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest = frame
            self._captured_at = time.time()
            self._seq += 1

    def snapshot(self) -> tuple[bool, np.ndarray | None, float, int]:
        with self._lock:
            if self._latest is None:
                return False, None, 0.0, self._seq
            now = time.time()
            if self._is_expired_locked(now):
                self._clear_locked()
                return False, None, 0.0, self._seq
            return True, self._latest.copy(), self._captured_at, self._seq

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq
