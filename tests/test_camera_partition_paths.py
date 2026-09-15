"""巷道编号与单路通道命名空间互斥。"""

import json
import os

import pytest

from services.aisle_store import bind_group, create_aisle_with_cameras, save_aisle, empty_aisle
from services.camera_partition import (
    aisle_id_legacy_path_conflict,
    legacy_path_aisle_conflict,
    run_partition_audit,
)
from services.camera_store import create_camera, load_cameras, save_cameras


@pytest.fixture
def partition_env(tmp_path):
    json_dir = tmp_path / "json"
    aisles_dir = json_dir / "aisles"
    aisles_dir.mkdir(parents=True)
    cam_file = tmp_path / "camera_ips.json"
    save_cameras(str(cam_file), [])
    mtx = str(tmp_path / "mediamtx.yml")
    with open(mtx, "w") as f:
        f.write("paths: {}\n")
    app_config = {"paths": {"json_dir": str(json_dir)}}
    return str(cam_file), str(mtx), str(json_dir), app_config


def test_legacy_path_conflicts_with_aisle_id(partition_env):
    cam_file, mtx, json_dir, _ = partition_env
    save_aisle(empty_aisle("aisle1"), json_dir)
    assert legacy_path_aisle_conflict("aisle1", json_dir)
    assert legacy_path_aisle_conflict("aisle1-L", json_dir)
    assert legacy_path_aisle_conflict("aisle1-R", json_dir)
    assert legacy_path_aisle_conflict("other", json_dir) is None

    r = create_camera(
        cam_file,
        mtx,
        {"path": "aisle1", "name": "bad", "source_type": "publisher"},
        json_dir=json_dir,
    )
    assert r.get("error")


def test_aisle_id_conflicts_with_legacy_path(partition_env):
    cam_file, mtx, json_dir, _ = partition_env
    create_camera(
        cam_file,
        mtx,
        {"path": "aisle1", "name": "legacy", "source_type": "publisher"},
        json_dir=None,
    )
    assert aisle_id_legacy_path_conflict("aisle1", cam_file, json_dir)
    r = create_aisle_with_cameras(
        "aisle1",
        {"path": "aisle1-L", "source_type": "publisher"},
        {"path": "aisle1-R", "source_type": "publisher"},
        camera_file=cam_file,
        mediamtx_config_path=mtx,
        json_dir=json_dir,
    )
    assert r.get("error")


def test_partition_audit_flags_legacy_aisle1_path(partition_env):
    cam_file, mtx, json_dir, app_config = partition_env
    save_cameras(
        cam_file,
        [
            {"id": "1", "path": "aisle1-L", "name": "L", "source_type": "publisher", "enabled": True},
            {"id": "2", "path": "aisle1-R", "name": "R", "source_type": "publisher", "enabled": True},
            {"id": "13", "path": "aisle1", "name": "legacy", "source_type": "publisher", "enabled": True},
        ],
    )
    save_aisle(empty_aisle("aisle1"), json_dir)
    bind_group("aisle1", "1", "2", json_dir)
    report = run_partition_audit(app_config, cam_file)
    assert report["legacy_aisle_path_conflicts"]
    assert not report["ok"]
