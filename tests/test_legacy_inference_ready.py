"""Legacy 单路推理就绪校验。"""

import json

import pytest

from services.aisle_store import bind_group, require_legacy_inference_ready


@pytest.fixture
def json_dir(tmp_path):
    d = tmp_path / "json"
    (d / "cameras").mkdir(parents=True)
    (d / "aisles").mkdir(parents=True)
    return str(d)


def _write_cam_json(json_dir: str, camera_id: str, boxes: list | None = None):
    path = f"{json_dir}/cameras/{camera_id}.json"
    payload = {
        "annotation_size": {"width": 640, "height": 360},
        "source_info": {"camera_name": camera_id},
        "boxes": boxes or [],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_legacy_ready_with_boxes(json_dir: str):
    _write_cam_json(json_dir, "lonely", [{"box_id": "1", "video_polygon": [[0, 0], [1, 0], [1, 1]]}])
    ok, err = require_legacy_inference_ready("lonely", json_dir)
    assert err is None
    assert ok and ok.get("camera_id") == "lonely"


def test_legacy_rejects_grouped(json_dir: str, monkeypatch):
    bind_group("a1", "cam-l", "cam-r", json_dir)
    _write_cam_json(json_dir, "cam-l", [{"box_id": "1", "video_polygon": [[0, 0], [1, 0], [1, 1]]}])
    ok, err = require_legacy_inference_ready("cam-l", json_dir)
    assert ok is None
    assert err and "已编入巷道" in err


def test_legacy_rejects_empty_boxes(json_dir: str):
    _write_cam_json(json_dir, "empty", [])
    ok, err = require_legacy_inference_ready("empty", json_dir)
    assert ok is None
    assert err and "货框" in err
