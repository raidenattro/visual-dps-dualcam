"""巷道成组：未成组禁推理；一台相机只属于一组。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from services.aisle_store import (
    bind_group,
    camera_group,
    require_grouped,
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
