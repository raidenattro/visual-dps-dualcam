"""巷道成组：未成组禁推理；一台相机只属于一组。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from services.aisle_store import (
    bind_group,
    camera_group,
    empty_aisle,
    require_grouped,
    save_aisle,
    unbind_group,
)


@pytest.fixture()
def json_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    d = tmp_path / "json"
    (d / "aisles").mkdir(parents=True)
    monkeypatch.setenv("JSON_DIR", str(d))
    return str(d)


def test_bind_group_and_lookup(json_dir: str):
    bind_group("aisle-13", "cam-l", "cam-r", json_dir)
    assert camera_group("cam-l", json_dir) == {"aisle_id": "aisle-13", "role": "L"}
    assert camera_group("cam-r", json_dir) == {"aisle_id": "aisle-13", "role": "R"}
    ok, err = require_grouped("cam-l", json_dir)
    assert err is None
    assert ok["L"] == "cam-l" and ok["R"] == "cam-r"


def test_ungrouped_cannot_infer(json_dir: str):
    ok, err = require_grouped("lonely", json_dir)
    assert ok is None
    assert "禁止开推理" in (err or "")


def test_camera_cannot_join_two_groups(json_dir: str):
    bind_group("a1", "cam-l", "cam-r", json_dir)
    with pytest.raises(ValueError, match="已属于"):
        bind_group("a2", "cam-l", "cam-x", json_dir)


def test_same_camera_both_roles_rejected(json_dir: str):
    with pytest.raises(ValueError, match="同一台"):
        bind_group("a1", "cam1", "cam1", json_dir)


def test_unbind_clears_group(json_dir: str):
    bind_group("aisle-13", "cam-l", "cam-r", json_dir)
    unbind_group("aisle-13", json_dir)
    ok, err = require_grouped("cam-l", json_dir)
    assert ok is None
    assert err


def test_save_keeps_both_wall_quads(json_dir: str):
    """整份巷道落盘必须同时保留两面墙的四角，不能只留下当前墙。"""
    data = empty_aisle("aisle-keep")
    data["views"]["L"]["walls"][0]["quad"] = [[10, 10], [20, 10], [20, 20], [10, 20]]
    data["views"]["L"]["walls"][1]["quad"] = [[80, 10], [90, 10], [90, 20], [80, 20]]
    data["views"]["R"]["walls"][0]["quad"] = [[11, 11], [21, 11], [21, 21], [11, 21]]
    data["views"]["R"]["walls"][1]["quad"] = [[81, 11], [91, 11], [91, 21], [81, 21]]
    saved = save_aisle(data, json_dir)
    assert saved["views"]["L"]["walls"][0]["quad"][0] == [10, 10]
    assert saved["views"]["L"]["walls"][1]["quad"][0] == [80, 10]
    assert saved["views"]["R"]["walls"][1]["quad"][0] == [81, 11]


def test_each_wall_keeps_own_grid(json_dir: str):
    data = empty_aisle("aisle-grid")
    data["views"]["L"]["walls"][0]["n_layers"] = 5
    data["views"]["L"]["walls"][0]["n_cols"] = 3
    data["views"]["L"]["walls"][1]["n_layers"] = 2
    data["views"]["L"]["walls"][1]["n_cols"] = 6
    saved = save_aisle(data, json_dir)
    w1 = saved["views"]["L"]["walls"][0]
    w2 = saved["views"]["L"]["walls"][1]
    assert (w1["n_layers"], w1["n_cols"]) == (5, 3)
    assert (w2["n_layers"], w2["n_cols"]) == (2, 6)
