"""runtime_config 与 pose_frame_interval 注入。"""

from __future__ import annotations

import json
from pathlib import Path

from services.runtime_config_service import (
    DEFAULT_POSE_FRAME_INTERVAL,
    ensure_runtime_overlay,
    get_effective_settings,
    get_merged_inference_section,
)


def test_ensure_runtime_overlay_writes_interval_2(tmp_path: Path):
    path = tmp_path / "runtime_config.json"
    app = {"inference": {"frame_rate": 15, "height": 480, "pose_frame_interval": 2}}
    ensure_runtime_overlay(app, path=str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["inference"]["pose_frame_interval"] == 2
    # 已有文件不覆盖
    data["inference"]["pose_frame_interval"] = 9
    path.write_text(json.dumps(data), encoding="utf-8")
    ensure_runtime_overlay(app, path=str(path))
    again = json.loads(path.read_text(encoding="utf-8"))
    assert again["inference"]["pose_frame_interval"] == 9


def test_effective_settings_runtime_overrides_app_config(tmp_path: Path):
    path = tmp_path / "runtime_config.json"
    path.write_text(
        json.dumps({"inference": {"pose_frame_interval": 2, "frame_rate": 15}}),
        encoding="utf-8",
    )
    app = {"inference": {"pose_frame_interval": 1, "frame_rate": 15, "height": 480}}
    items = get_effective_settings(app, camera=None, path=str(path))
    assert int(items["inference.pose_frame_interval"]) == 2
    merged = get_merged_inference_section(app, path=str(path))
    assert int(merged["pose_frame_interval"]) == 2


def test_default_pose_interval_is_2():
    assert DEFAULT_POSE_FRAME_INTERVAL == 2
    app = {"inference": {"pose_frame_interval": 2}}
    items = get_effective_settings(app, camera=None, path="/no/such/runtime.json")
    assert int(items["inference.pose_frame_interval"]) == 2
