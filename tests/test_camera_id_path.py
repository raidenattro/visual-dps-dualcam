"""摄像头主键 id 自增，path 只是 MediaMTX 通道名。"""

from __future__ import annotations

from pathlib import Path

import pytest

from services.camera_store import create_camera, load_cameras, update_camera


@pytest.fixture()
def cam_files(tmp_path: Path) -> tuple[str, str]:
    cam_file = str(tmp_path / "cameras.json")
    mtx = str(tmp_path / "mediamtx.yml")
    Path(cam_file).write_text("[]", encoding="utf-8")
    return cam_file, mtx


@pytest.fixture(autouse=True)
def _no_live_mediamtx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "services.mediamtx_service.reload_mediamtx_runtime",
        lambda cameras: {"reloaded": False, "skipped": True, "reason": "test"},
    )


def test_create_assigns_incrementing_id_keeps_path(cam_files: tuple[str, str]):
    cam_file, mtx = cam_files
    a = create_camera(
        cam_file,
        mtx,
        {"path": "aisle-1-L", "name": "左", "source_type": "publisher"},
    )
    b = create_camera(
        cam_file,
        mtx,
        {"path": "aisle-1-R", "name": "右", "source_type": "publisher"},
    )
    assert a["status"] == "success"
    assert a["camera"]["id"] == "1"
    assert a["camera"]["path"] == "aisle-1-L"
    assert b["camera"]["id"] == "2"
    assert b["camera"]["path"] == "aisle-1-R"


def test_create_rejects_duplicate_path(cam_files: tuple[str, str]):
    cam_file, mtx = cam_files
    create_camera(cam_file, mtx, {"path": "cam-a", "name": "A", "source_type": "publisher"})
    dup = create_camera(cam_file, mtx, {"path": "cam-a", "name": "B", "source_type": "publisher"})
    assert "已被使用" in (dup.get("error") or "")


def test_update_can_rename_path_keeps_id(cam_files: tuple[str, str]):
    cam_file, mtx = cam_files
    created = create_camera(
        cam_file,
        mtx,
        {"path": "old-path", "name": "A", "source_type": "publisher"},
    )
    cid = created["camera"]["id"]
    out = update_camera(cam_file, mtx, cid, {"path": "new-path", "name": "A", "source_type": "publisher"})
    assert out.get("status") == "success", out
    rec = out["camera"]
    assert rec["id"] == cid
    assert rec["path"] == "new-path"
    assert rec["url"] == "rtsp://127.0.0.1:8554/new-path"
    loaded = {c["id"]: c for c in load_cameras(cam_file)}
    assert loaded[cid]["path"] == "new-path"


def test_load_legacy_id_equals_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "services.mediamtx_service.reload_mediamtx_runtime",
        lambda cameras: {"reloaded": False, "skipped": True, "reason": "test"},
    )
    cam_file = str(tmp_path / "cameras.json")
    Path(cam_file).write_text(
        '[{"id":"aisle4-L","path":"aisle4-L","name":"左","url":"rtsp://127.0.0.1:8554/aisle4-L",'
        '"source_type":"publisher","enabled":true,"pull_url":""}]',
        encoding="utf-8",
    )
    items = load_cameras(cam_file)
    assert items[0]["id"] == "aisle4-L"
    assert items[0]["path"] == "aisle4-L"
    mtx = str(tmp_path / "mediamtx.yml")
    out = update_camera(
        cam_file,
        mtx,
        "aisle4-L",
        {"path": "left-new", "name": "左", "source_type": "publisher"},
    )
    assert out["camera"]["id"] == "aisle4-L"
    assert out["camera"]["path"] == "left-new"
