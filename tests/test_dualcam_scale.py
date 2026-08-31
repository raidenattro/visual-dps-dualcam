"""UV 缩放、配对窗、开推理校验、3D token。"""

from __future__ import annotations

from pathlib import Path

import pytest

from services.aisle_store import bind_group, require_grouped, require_inference_ready, save_aisle
from services.dualcam_config import pair_window_sec, scale_keypoints_to_calib
from services.dualcam_overlay import collision_token, token_box_id


def test_scale_identity_when_sizes_match():
    persons = [{"keypoints": [[100.0, 50.0, 0.9]]}]
    out, meta = scale_keypoints_to_calib(persons, 1280, 720, 1280, 720)
    assert meta["mode"] == "identity"
    assert out[0]["keypoints"][0][0] == 100.0


def test_scale_ratio_when_infer_differs():
    persons = [{"keypoints": [[427.0, 240.0, 0.8]]}]
    out, meta = scale_keypoints_to_calib(persons, 854, 480, 1280, 720)
    assert meta["mode"] == "ratio"
    assert meta["sx"] == pytest.approx(1280 / 854)
    assert out[0]["keypoints"][0][0] == pytest.approx(427.0 * 1280 / 854)
    assert out[0]["keypoints"][0][1] == pytest.approx(240.0 * 720 / 480)


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


def test_token_includes_wall():
    assert token_box_id(1, "r0c0") == "w1-r0c0"
    assert token_box_id(2, "w2-r0c0") == "w2-r0c0"
    assert collision_token("aisle-1", 1, "r0c0") == "aisle-1:w1-r0c0"


def test_require_inference_ready_needs_solve_and_mesh(tmp_path, monkeypatch):
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
    save_aisle(data, str(d))
    ok, err = require_inference_ready("cam-l", str(d))
    assert ok is None
    assert "层线" in (err or "")

    data["slot_meshes"] = [{"wall_id": 1, "rows": 1, "cols": 1, "vertices": []}]
    save_aisle(data, str(d))
    ok, err = require_inference_ready("cam-l", str(d))
    assert err is None
    assert ok["aisle_id"] == "aisle-x"
