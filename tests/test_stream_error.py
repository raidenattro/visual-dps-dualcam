"""摄像头离线原因：未推流 / 未注册 / 编码。"""

from services.mediamtx_service import SOURCE_PUBLISHER, SOURCE_RTSP_PULL, describe_stream_error


def _snap(*, configured, ready, path="cam1"):
    return {
        "configured": set(configured),
        "runtime": {path: {"ready": ready}} if path in configured else {},
        "api_ok": True,
    }


def test_publisher_not_publishing():
    cam = {"id": "cam1", "path": "cam1", "source_type": SOURCE_PUBLISHER}
    msg = describe_stream_error(cam, _snap(configured=["cam1"], ready=False), online=False)
    assert "no one is publishing" in msg
    assert "cam1" in msg


def test_path_not_configured():
    cam = {"id": "cam1", "path": "cam1", "source_type": SOURCE_PUBLISHER}
    msg = describe_stream_error(cam, _snap(configured=[], ready=False), online=False)
    assert "path is not configured" in msg


def test_codec_only_after_probe():
    cam = {"id": "cam1", "path": "cam1", "source_type": SOURCE_PUBLISHER}
    snap = _snap(configured=["cam1"], ready=True)
    assert describe_stream_error(cam, snap, online=False, probed=False) == ""
    assert "H.264" in describe_stream_error(cam, snap, online=False, probed=True)


def test_online_has_no_error():
    cam = {"id": "cam1", "path": "cam1", "source_type": SOURCE_PUBLISHER}
    assert describe_stream_error(cam, _snap(configured=["cam1"], ready=True), online=True) == ""


def test_rtsp_pull_not_ready():
    cam = {
        "id": "cam1",
        "path": "cam1",
        "source_type": SOURCE_RTSP_PULL,
        "pull_url": "rtsp://192.168.1.10/live",
    }
    msg = describe_stream_error(cam, _snap(configured=["cam1"], ready=False), online=False)
    assert "上游拉流未就绪" in msg
    assert "192.168.1.10" in msg
