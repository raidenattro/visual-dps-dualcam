"""摄像头 MJPEG 预览流：单路 RTSP 长连接，多客户端共享。"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, Iterator, Optional

import cv2

from services.camera_service import normalize_rtsp_url
from services.video_service import resize_frame_to_height

STREAM_FPS = max(1, int(os.environ.get("STREAM_FPS", "15")))
STREAM_JPEG_QUALITY = max(30, min(95, int(os.environ.get("STREAM_JPEG_QUALITY", "68"))))
BOUNDARY = b"frame"

_lock = threading.Lock()
_sessions: Dict[str, "MjpegStreamSession"] = {}


class MjpegStreamSession:
    def __init__(self, camera_id: str, url: str, capture_height: int):
        self.camera_id = camera_id
        self.url = url
        self.capture_height = capture_height
        self._clients = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._last_frame_at = 0.0

    def acquire(self) -> None:
        with _lock:
            self._clients += 1
            if self._clients == 1:
                self._start_reader()

    def release(self) -> None:
        with _lock:
            self._clients = max(0, self._clients - 1)
            if self._clients == 0:
                self._running = False

    def get_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg

    def _start_reader(self) -> None:
        if self._thread and self._thread.is_alive():
            self._running = True
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, name=f"mjpeg-{self.camera_id}", daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        interval = 1.0 / STREAM_FPS
        cap = None
        while True:
            with _lock:
                if not self._running or self._clients <= 0:
                    break
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(normalize_rtsp_url(self.url))
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if not cap.isOpened():
                    time.sleep(0.5)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                time.sleep(0.2)
                continue

            frame = resize_frame_to_height(frame, self.capture_height)
            ok_enc, buf = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY],
            )
            if ok_enc:
                with self._frame_lock:
                    self._latest_jpeg = buf.tobytes()
                    self._last_frame_at = time.time()

            time.sleep(interval)

        if cap is not None:
            cap.release()


def _session_key(camera_id: str, capture_height: int) -> str:
    return f"{camera_id}:{capture_height}"


def _get_session(camera_id: str, url: str, capture_height: int) -> MjpegStreamSession:
    key = _session_key(camera_id, capture_height)
    with _lock:
        session = _sessions.get(key)
        if session is None or session.url != url or session.capture_height != capture_height:
            if session is not None:
                session._running = False
            session = MjpegStreamSession(camera_id, url, capture_height)
            _sessions[key] = session
        return session


def iter_mjpeg(camera_id: str, url: str, capture_height: int) -> Iterator[bytes]:
    session = _get_session(camera_id, url, capture_height)
    session.acquire()
    interval = 1.0 / STREAM_FPS
    try:
        while session._running or session._clients > 0:
            with _lock:
                if session._clients <= 0:
                    break
            jpeg = session.get_jpeg()
            if jpeg:
                yield (
                    b"--" + BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            time.sleep(interval)
    finally:
        session.release()


def mjpeg_media_type() -> str:
    return f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}"


def stream_recently_active(camera_id: str, max_age: float = 30.0) -> bool:
    """MJPEG 会话近期是否成功出帧（用于补充 RTSP 探测结果）。"""
    now = time.time()
    prefix = f"{camera_id}:"
    with _lock:
        for key, session in _sessions.items():
            if not str(key).startswith(prefix):
                continue
            last_at = float(session._last_frame_at or 0)
            if last_at and (now - last_at) <= max_age:
                return True
    return False
