"""推理尺寸必须跟真实帧走，不能信启动时的 CAP_PROP。"""

from __future__ import annotations

import numpy as np

from services.inference_service import _compute_infer_resolution, _sync_infer_size_from_frame
from services.rtsp_capture import vf_with_fixed_scale


def test_compute_does_not_upscale_360p_when_target_is_720():
    w, h, resize = _compute_infer_resolution(640, 360, 720)
    assert (w, h) == (640, 360)
    assert resize is False


def test_compute_downscales_1080p_to_720():
    w, h, resize = _compute_infer_resolution(1920, 1080, 720)
    assert h == 720
    assert w == 1280
    assert resize is True


def test_sync_from_frame_fixes_stale_720p_probe():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    sw, sh, iw, ih, resize, changed = _sync_infer_size_from_frame(
        frame, 1280, 720, 720, 1280, 720, False,
    )
    assert changed is True
    assert (sw, sh) == (640, 360)
    assert (iw, ih) == (640, 360)
    assert resize is False


def test_sync_from_frame_noop_when_matches():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    sw, sh, iw, ih, resize, changed = _sync_infer_size_from_frame(
        frame, 640, 360, 720, 640, 360, False,
    )
    assert changed is False
    assert (sw, sh, iw, ih) == (640, 360, 640, 360)
    assert resize is False


def test_ffmpeg_vf_appends_scale_after_hwdownload():
    vf = vf_with_fixed_scale("hwdownload,format=bgr24", 1280, 720)
    assert vf == "hwdownload,format=bgr24,scale=1280:720"


def test_ffmpeg_vf_keeps_existing_scale():
    vf = vf_with_fixed_scale("scale=640:360,format=bgr24", 1280, 720)
    assert "scale=640:360" in vf
    assert "scale=1280:720" not in vf
