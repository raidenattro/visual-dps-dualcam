"""单路 Δt：开启时比同一只手腕的像素位移；关闭时仍用 n 帧进框。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from services.event_engine.collision import CollisionProcessor, lag_frames
from services.event_engine.worker import EventRedisWorker, _CameraContext
from services.runtime_config_service import get_legacy_section, get_public_settings, patch_public_settings


def test_lag_matches_experiment_15fps_and_25fps():
    assert lag_frames(15, 0.24) == 4
    assert lag_frames(25, 0.24) == 6


def test_legacy_section_defaults_dwell_off():
    section = get_legacy_section({})
    assert section["dwell_enabled"] is False
    assert section["dwell_delta_sec"] == pytest.approx(0.24)
    assert section["dwell_tau_px"] == pytest.approx(20.0)
    assert section["alarm_min_consecutive_frames"] == 3


def test_public_settings_roundtrip(tmp_path: Path):
    path = tmp_path / "runtime_config.json"
    saved = patch_public_settings(
        {
            "legacy.dwell_enabled": True,
            "legacy.dwell_delta_sec": 0.48,
            "legacy.dwell_tau_px": 12,
            "legacy.frame_rate": 12,
        },
        path=str(path),
    )
    assert saved["status"] == "success"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["legacy"]["dwell_enabled"] is True
    items = get_public_settings({"inference": {}, "legacy": {}}, path=str(path))["items"]
    assert items["legacy.dwell_enabled"] is True
    assert items["legacy.dwell_delta_sec"] == pytest.approx(0.48)
    assert items["legacy.frame_rate"] == 12
    rejected = patch_public_settings({"legacy.dwell_tau_px": 0}, path=str(path))
    assert rejected["status"] == "error"


def _box():
    contour = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]], np.int32)
    return {"shelf_code": "S", "box_id": "1", "orig_contour": contour}


def _person(wrist_xy, wrist_conf=0.9):
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    kps[5] = [40.0, 40.0, 0.9]
    kps[6] = [60.0, 40.0, 0.9]
    kps[9] = [float(wrist_xy[0]), float(wrist_xy[1]), float(wrist_conf)]
    kps[10] = [70.0, 200.0, 0.1]
    return {"keypoints": kps}


def _proc(**kwargs):
    params = dict(
        video_fps=15.0,
        pose_frame_interval=2,
        dwell_enabled=True,
        dwell_delta_sec=0.24,
        dwell_tau_px=20.0,
        alarm_min_consecutive_frames=3,
        alarm_cooldown_frames=0,
    )
    params.update(kwargs)
    return CollisionProcessor([_box()], **params)


def _feed(proc: CollisionProcessor, frames: list[tuple[int, tuple[float, float], float]]):
    out = []
    for frame_idx, xy, conf in frames:
        out.append(proc.process({"frame_idx": frame_idx, "persons": [_person(xy, conf)]}))
    return out


def test_still_wrist_alarms_without_waiting_n_frames():
    proc = _proc()
    assert proc.alarm_min_consecutive_frames == 3
    results = _feed(proc, [(0, (50, 50), 0.9), (2, (50, 50), 0.9), (4, (50, 50), 0.9)])
    assert results[0]["alarm_collisions"] == []
    assert results[2]["collisions"] == ["S:1"]
    assert results[2]["alarm_collisions"] == ["S:1"]


def test_moving_wrist_inside_box_is_not_a_hit():
    proc = _proc()
    results = _feed(proc, [(0, (50, 50), 0.9), (4, (80, 50), 0.9)])
    assert results[1]["collisions"] == []


def test_missing_comparison_frame_is_not_a_hit():
    proc = _proc()
    results = _feed(proc, [(0, (50, 50), 0.9), (8, (50, 50), 0.9)])
    assert results[1]["collisions"] == []


def test_dwell_disabled_needs_n_consecutive_frames():
    proc = _proc(dwell_enabled=False)
    results = _feed(proc, [(0, (50, 50), 0.9), (1, (50, 50), 0.9), (2, (50, 50), 0.9)])
    assert results[0]["collisions"] == ["S:1"]
    assert results[0]["alarm_collisions"] == []
    assert results[1]["alarm_collisions"] == []
    assert results[2]["alarm_collisions"] == ["S:1"]


def test_worker_hot_reads_legacy_dwell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUNTIME_CONFIG_FILE", str(tmp_path / "missing.json"))
    worker = EventRedisWorker({"paths": {"json_dir": str(tmp_path)}, "legacy": {"dwell_enabled": False}})
    proc = CollisionProcessor([])
    worker._contexts["cam"] = _CameraContext("none.json", processor=proc)
    worker._apply_runtime_settings(
        {
            "frame_rate": 12,
            "pose_frame_interval": 1,
            "alarm_min_consecutive_frames": 3,
            "alarm_cooldown_frames": 0,
            "dwell_enabled": True,
            "dwell_delta_sec": 0.5,
            "dwell_tau_px": 12,
        }
    )
    assert proc.dwell_enabled is True
    assert proc.dwell_delta_sec == pytest.approx(0.5)
    assert proc.dwell_tau_px == pytest.approx(12.0)
    assert proc.video_fps == pytest.approx(12.0)
