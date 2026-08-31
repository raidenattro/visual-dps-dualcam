"""UV letterbox 缩放、配对窗、开推理校验、货架号:货位编号 token。"""

from __future__ import annotations

from pathlib import Path

import pytest

from services.aisle_store import bind_group, require_grouped, require_inference_ready, save_aisle
from services.dualcam_config import pair_window_sec, scale_keypoints_to_calib
from services.dualcam_overlay import collision_token
from services.runtime_config_service import CAMERA_OVERRIDE_KEYS


def test_scale_identity_when_sizes_match():
    persons = [{"keypoints": [[100.0, 50.0, 0.9]]}]
    out, meta = scale_keypoints_to_calib(persons, 1280, 720, 1280, 720)
    assert meta["mode"] == "identity"
    assert out[0]["keypoints"][0][0] == 100.0


def test_scale_letterbox_same_aspect():
    persons = [{"keypoints": [[100.0, 50.0, 0.8]]}]
    out, meta = scale_keypoints_to_calib(persons, 640, 360, 1280, 720)
    assert meta["mode"] == "letterbox"
    assert meta["sx"] == pytest.approx(meta["sy"])
    assert meta["pad_x"] == pytest.approx(0.0, abs=0.5)
    assert meta["pad_y"] == pytest.approx(0.0, abs=0.5)
    assert out[0]["keypoints"][0][0] == pytest.approx(200.0, abs=0.5)
    assert out[0]["keypoints"][0][1] == pytest.approx(100.0, abs=0.5)


def test_scale_pillarbox_preserves_center_when_aspect_differs():
    """4:3 推理 → 16:9 标定：左右留边，中心仍落在标定中心，不会被拉扁。"""
    persons = [{"keypoints": [[320.0, 240.0, 0.9]]}]
    out, meta = scale_keypoints_to_calib(persons, 640, 480, 1280, 720)
    assert meta["mode"] == "letterbox"
    assert meta["sx"] == pytest.approx(meta["sy"])
    assert meta["pad_x"] == pytest.approx(160.0, abs=0.5)
    assert meta["pad_y"] == pytest.approx(0.0, abs=0.5)
    assert out[0]["keypoints"][0][0] == pytest.approx(640.0, abs=0.5)
    assert out[0]["keypoints"][0][1] == pytest.approx(360.0, abs=0.5)
    # 左上角不会被拉到标定 (0,0)，而是落在内容区
    corner, _ = scale_keypoints_to_calib(
        [{"keypoints": [[0.0, 0.0, 1.0]]}], 640, 480, 1280, 720,
    )
    assert corner[0]["keypoints"][0][0] == pytest.approx(160.0, abs=0.5)


def test_scale_normalize_fallback_when_infer_missing_and_uv_unit():
    persons = [{"keypoints": [[0.5, 0.25, 0.9]]}]
    out, meta = scale_keypoints_to_calib(persons, 0, 0, 1280, 720)
    assert meta["mode"] == "normalize"
    assert out[0]["keypoints"][0][0] == pytest.approx(640.0)
    assert out[0]["keypoints"][0][1] == pytest.approx(180.0)


def test_scale_unknown_when_infer_missing_but_already_pixels():
    persons = [{"keypoints": [[400.0, 200.0, 0.9]]}]
    out, meta = scale_keypoints_to_calib(persons, 0, 0, 1280, 720)
    assert meta["mode"] == "unknown"
    assert out[0]["keypoints"][0][0] == 400.0


def test_pair_window_follows_pose_interval(monkeypatch):
    from services.dualcam_config import DEFAULT_DUALCAM

    monkeypatch.setattr(
        "services.dualcam_config.get_dualcam_section",
        lambda app=None: {**DEFAULT_DUALCAM, "pair_window_min_sec": 0.01, "pair_window_max_sec": 2.0},
    )
    w1 = pair_window_sec(frame_rate=15, pose_frame_interval=1)
    w3 = pair_window_sec(frame_rate=15, pose_frame_interval=3)
    assert w3 == pytest.approx(w1 * 3, rel=0.05)
    assert w3 > 0.2


def test_token_uses_shelf_and_box_not_aisle_or_wall_prefix():
    assert collision_token("S-A", "A-01") == "S-A:A-01"
    assert collision_token("货架1", "r0c0") == "货架1:r0c0"
    assert "w1-" not in collision_token("S-A", "r0c0")


def test_pose_interval_not_camera_override():
    assert "inference.pose_frame_interval" not in CAMERA_OVERRIDE_KEYS


def test_require_inference_ready_needs_solve_mesh_and_shelf(tmp_path, monkeypatch):
    d = tmp_path / "json"
    (d / "aisles").mkdir(parents=True)
    monkeypatch.setenv("JSON_DIR", str(d))
    bind_group("aisle-x", "cam-l", "cam-r", str(d))
    ok, err = require_grouped("cam-l", str(d))
    assert err is None
    ok, err = require_inference_ready("cam-l", str(d))
    assert ok is None
    assert "尚未反解" in (err or "")

    from services.aisle_store import load_aisle

    data = load_aisle("aisle-x", str(d))
    data["solved"] = {"ok": True, "cameras": {"L": {}, "R": {}}}
    data["required_wall_ids"] = [1]
    save_aisle(data, str(d))
    ok, err = require_inference_ready("cam-l", str(d))
    assert ok is None
    assert "层线" in (err or "")

    data["slot_meshes"] = [{"wall_id": 1, "rows": 1, "cols": 1, "vertices": []}]
    save_aisle(data, str(d))
    ok, err = require_inference_ready("cam-l", str(d))
    assert ok is None
    assert "货架号" in (err or "")

    data["slot_meshes"] = [{"wall_id": 1, "rows": 1, "cols": 1, "vertices": [], "shelf_code": "S1"}]
    save_aisle(data, str(d))
    ok, err = require_inference_ready("cam-l", str(d))
    assert err is None
    assert ok["aisle_id"] == "aisle-x"


def test_require_inference_ready_single_wall_ok_without_wall2(tmp_path, monkeypatch):
    d = tmp_path / "json"
    (d / "aisles").mkdir(parents=True)
    monkeypatch.setenv("JSON_DIR", str(d))
    bind_group("aisle-y", "cam-l", "cam-r", str(d))
    from services.aisle_store import load_aisle

    data = load_aisle("aisle-y", str(d))
    data["solved"] = {"ok": True, "cameras": {"L": {}, "R": {}}}
    data["required_wall_ids"] = [1]
    data["slot_meshes"] = [{"wall_id": 1, "rows": 1, "cols": 1, "vertices": [], "shelf_code": "左货架"}]
    save_aisle(data, str(d))
    ok, err = require_inference_ready("cam-l", str(d))
    assert err is None
    assert ok["required_wall_ids"] == [1]

    data["required_wall_ids"] = [1, 2]
    save_aisle(data, str(d))
    ok, err = require_inference_ready("cam-l", str(d))
    assert ok is None
    assert "墙2" in (err or "")
