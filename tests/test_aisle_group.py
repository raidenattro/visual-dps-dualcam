"""巷道成组：未成组禁推理；一台相机只属于一组。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from services.aisle_store import (
    bind_group,
    camera_group,
    create_aisle_with_cameras,
    empty_aisle,
    load_aisle,
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


def test_create_aisle_with_cameras_binds_pair(tmp_path: Path, json_dir: str):
    cam_file = str(tmp_path / "cameras.json")
    mtx = str(tmp_path / "mediamtx.yml")
    Path(cam_file).write_text("[]", encoding="utf-8")
    out = create_aisle_with_cameras(
        "aisle-9",
        {
            "path": "aisle-9-L",
            "name": "左",
            "source_type": "rtsp_pull",
            "pull_url": "rtsp://10.0.0.1:554/s1",
        },
        {
            "path": "aisle-9-R",
            "name": "右",
            "source_type": "rtsp_pull",
            "pull_url": "rtsp://10.0.0.2:554/s1",
        },
        camera_file=cam_file,
        mediamtx_config_path=mtx,
        json_dir=json_dir,
    )
    assert out.get("status") == "success", out
    aisle = load_aisle("aisle-9", json_dir)
    assert aisle["cameras"]["L"]["camera_id"] == "aisle-9-L"
    assert aisle["cameras"]["R"]["camera_id"] == "aisle-9-R"
    assert camera_group("aisle-9-L", json_dir)["role"] == "L"
    dup = create_aisle_with_cameras(
        "aisle-9",
        {"path": "x-L", "name": "x", "source_type": "rtsp_pull", "pull_url": "rtsp://10.0.0.3:554/s"},
        {"path": "x-R", "name": "y", "source_type": "rtsp_pull", "pull_url": "rtsp://10.0.0.4:554/s"},
        camera_file=cam_file,
        mediamtx_config_path=mtx,
        json_dir=json_dir,
    )
    assert "已存在" in (dup.get("error") or "")


def test_delete_aisle_with_cameras_unbinds(tmp_path: Path, json_dir: str, monkeypatch: pytest.MonkeyPatch):
    from services.aisle_store import delete_aisle_with_cameras
    from services.camera_store import load_cameras
    import services.inference_container_service as infer_svc

    monkeypatch.setattr(infer_svc, "stop_inference_container", lambda *_a, **_k: None)

    cam_file = str(tmp_path / "cameras.json")
    mtx = str(tmp_path / "mediamtx.yml")
    Path(cam_file).write_text("[]", encoding="utf-8")
    created = create_aisle_with_cameras(
        "aisle-del",
        {
            "path": "aisle-del-L",
            "name": "左",
            "source_type": "rtsp_pull",
            "pull_url": "rtsp://10.0.0.1:554/s1",
        },
        {
            "path": "aisle-del-R",
            "name": "右",
            "source_type": "rtsp_pull",
            "pull_url": "rtsp://10.0.0.2:554/s1",
        },
        camera_file=cam_file,
        mediamtx_config_path=mtx,
        json_dir=json_dir,
    )
    assert created.get("status") == "success", created
    out = delete_aisle_with_cameras(
        "aisle-del",
        camera_file=cam_file,
        mediamtx_config_path=mtx,
        json_dir=json_dir,
    )
    assert out.get("status") == "success", out
    ids = {c["id"] for c in load_cameras(cam_file)}
    assert "aisle-del-L" not in ids
    assert "aisle-del-R" not in ids
    assert camera_group("aisle-del-L", json_dir) is None
    leftover = load_aisle("aisle-del", json_dir)
    assert leftover
    assert not leftover["cameras"]["L"]["camera_id"]
    assert not leftover["cameras"]["R"]["camera_id"]
