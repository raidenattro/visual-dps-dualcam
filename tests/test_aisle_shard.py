"""成组后 L/R 必须同一 logical shard；禁止再按 camera_id 各算各的。"""

from __future__ import annotations

import os

from services.aisle_store import bind_group
from services.event_engine.sharding import logical_shard_id, stream_key_for_aisle, stream_key_for_camera


def test_same_aisle_same_shard(tmp_path, monkeypatch):
    d = tmp_path / "json"
    (d / "aisles").mkdir(parents=True)
    monkeypatch.setenv("JSON_DIR", str(d))
    monkeypatch.setenv("POSE_LOGICAL_SHARD_COUNT", "16")
    monkeypatch.delenv("AISLE_ID", raising=False)
    bind_group("pair-a", "left-cam", "right-cam", str(d))

    sl = stream_key_for_camera("left-cam")
    sr = stream_key_for_camera("right-cam")
    assert sl == sr
    assert sl == stream_key_for_aisle("pair-a")
    assert logical_shard_id("pair-a") == logical_shard_id("pair-a")


def test_ungrouped_cameras_may_differ(tmp_path, monkeypatch):
    d = tmp_path / "json"
    (d / "aisles").mkdir(parents=True)
    monkeypatch.setenv("JSON_DIR", str(d))
    monkeypatch.setenv("POSE_LOGICAL_SHARD_COUNT", "16")
    monkeypatch.delenv("AISLE_ID", raising=False)
    # 未成组仍按 camera_id；这两路名字几乎必然落不同 shard
    a = stream_key_for_camera("cam-aaa-1")
    b = stream_key_for_camera("cam-zzz-2")
    # 不断言一定不同（哈希碰撞可能），只断言成组后被 aisle 覆盖
    bind_group("pair-b", "cam-aaa-1", "cam-zzz-2", str(d))
    assert stream_key_for_camera("cam-aaa-1") == stream_key_for_camera("cam-zzz-2")


def test_aisle_env_overrides_camera_hash(monkeypatch):
    monkeypatch.setenv("POSE_LOGICAL_SHARD_COUNT", "16")
    monkeypatch.setenv("AISLE_ID", "forced-aisle")
    assert stream_key_for_camera("whatever") == stream_key_for_aisle("forced-aisle")


def test_cam1_cam2_must_use_aisle_shard(monkeypatch):
    """现场 cam1/cam2 按 camera_id 会落到 6 和 12，aisle-1 在 7；不设 AISLE_ID 就永远配不上。"""
    monkeypatch.setenv("POSE_LOGICAL_SHARD_COUNT", "16")
    monkeypatch.delenv("AISLE_ID", raising=False)
    assert logical_shard_id("cam1") != logical_shard_id("cam2")
    assert logical_shard_id("cam1") != logical_shard_id("aisle-1")
    assert logical_shard_id("cam2") != logical_shard_id("aisle-1")
    monkeypatch.setenv("AISLE_ID", "aisle-1")
    assert stream_key_for_camera("cam1") == stream_key_for_camera("cam2") == stream_key_for_aisle("aisle-1")
