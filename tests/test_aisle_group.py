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


@pytest.fixture(autouse=True)
def _no_live_mediamtx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "services.mediamtx_service.reload_mediamtx_runtime",
        lambda cameras: {"reloaded": False, "skipped": True, "reason": "test"},
    )


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
    id_l = out["camera_l"]["id"]
    id_r = out["camera_r"]["id"]
    assert id_l != "aisle-9-L" and id_l.isdigit()
    assert out["camera_l"]["path"] == "aisle-9-L"
    assert out["camera_r"]["path"] == "aisle-9-R"
    assert aisle["cameras"]["L"]["camera_id"] == id_l
    assert aisle["cameras"]["R"]["camera_id"] == id_r
    assert camera_group(id_l, json_dir)["role"] == "L"
    assert out["aisle"].get("id") == 1
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
    cam_json = Path(json_dir) / "cameras"
    cam_json.mkdir(parents=True, exist_ok=True)
    (cam_json / f"{created['camera_l']['id']}.json").write_text("{}", encoding="utf-8")
    (cam_json / f"{created['camera_r']['id']}.json").write_text("{}", encoding="utf-8")
    out = delete_aisle_with_cameras(
        "aisle-del",
        camera_file=cam_file,
        mediamtx_config_path=mtx,
        json_dir=json_dir,
    )
    assert out.get("status") == "success", out
    ids = {c["id"] for c in load_cameras(cam_file)}
    assert created["camera_l"]["id"] not in ids
    assert created["camera_r"]["id"] not in ids
    assert camera_group(created["camera_l"]["id"], json_dir) is None
    leftover = load_aisle("aisle-del", json_dir)
    assert leftover is None
    from services.aisle_store import list_aisles
    assert "aisle-del" not in {a["aisle_id"] for a in list_aisles(json_dir, bound_only=False)}
    assert not (cam_json / f"{created['camera_l']['id']}.json").is_file()
    assert not (cam_json / f"{created['camera_r']['id']}.json").is_file()


def test_list_aisles_hides_unbound(json_dir: str):
    from services.aisle_store import list_aisles

    bind_group("aisle-ok", "cam-l", "cam-r", json_dir)
    save_aisle(empty_aisle("ghost"), json_dir)
    bound = {a["aisle_id"] for a in list_aisles(json_dir)}
    all_ids = {a["aisle_id"] for a in list_aisles(json_dir, bound_only=False)}
    assert bound == {"aisle-ok"}
    assert "ghost" in all_ids
    unbind_group("aisle-ok", json_dir)
    assert list_aisles(json_dir) == []


def test_purge_unbound_aisles_and_orphan_cameras(tmp_path: Path, json_dir: str, monkeypatch: pytest.MonkeyPatch):
    from services.aisle_store import purge_unbound_aisles_and_cameras
    from services.camera_store import create_camera, load_cameras
    import services.inference_container_service as infer_svc

    monkeypatch.setattr(infer_svc, "stop_inference_container", lambda *_a, **_k: None)

    cam_file = str(tmp_path / "cameras.json")
    mtx = str(tmp_path / "mediamtx.yml")
    Path(cam_file).write_text("[]", encoding="utf-8")
    created = create_aisle_with_cameras(
        "aisle-keep",
        {
            "path": "keep-L",
            "name": "左",
            "source_type": "rtsp_pull",
            "pull_url": "rtsp://10.0.0.1:554/s1",
        },
        {
            "path": "keep-R",
            "name": "右",
            "source_type": "rtsp_pull",
            "pull_url": "rtsp://10.0.0.2:554/s1",
        },
        camera_file=cam_file,
        mediamtx_config_path=mtx,
        json_dir=json_dir,
    )
    assert created.get("status") == "success", created
    save_aisle(empty_aisle("ghost"), json_dir)
    lonely = create_camera(
        cam_file,
        mtx,
        {"path": "lonely", "name": "无巷道", "source_type": "publisher"},
    )
    assert lonely.get("status") == "success", lonely
    cam_json = Path(json_dir) / "cameras"
    cam_json.mkdir(parents=True, exist_ok=True)
    (cam_json / "orphan-old.json").write_text("{}", encoding="utf-8")
    (cam_json / f"{created['camera_l']['id']}.json").write_text("{}", encoding="utf-8")

    out = purge_unbound_aisles_and_cameras(
        camera_file=cam_file,
        mediamtx_config_path=mtx,
        json_dir=json_dir,
    )
    assert out.get("status") == "success", out
    assert "ghost" in out["removed_aisles"]
    assert lonely["camera"]["id"] in out["removed_cameras"]
    assert "orphan-old" in out["removed_annotations"]
    assert load_aisle("ghost", json_dir) is None
    assert load_aisle("aisle-keep", json_dir)
    ids = {c["id"] for c in load_cameras(cam_file)}
    assert created["camera_l"]["id"] in ids
    assert created["camera_r"]["id"] in ids
    assert lonely["camera"]["id"] not in ids
    assert not (cam_json / "orphan-old.json").is_file()
    assert (cam_json / f"{created['camera_l']['id']}.json").is_file()


def test_update_aisle_cameras_changes_stream_not_ids(tmp_path: Path, json_dir: str):
    from services.aisle_store import update_aisle_cameras
    from services.camera_store import load_cameras

    cam_file = str(tmp_path / "cameras.json")
    mtx = str(tmp_path / "mediamtx.yml")
    Path(cam_file).write_text("[]", encoding="utf-8")
    created = create_aisle_with_cameras(
        "aisle-up",
        {
            "path": "aisle-up-L",
            "name": "左",
            "source_type": "rtsp_pull",
            "pull_url": "rtsp://10.0.0.1:554/s1",
        },
        {
            "path": "aisle-up-R",
            "name": "右",
            "source_type": "rtsp_pull",
            "pull_url": "rtsp://10.0.0.2:554/s1",
        },
        camera_file=cam_file,
        mediamtx_config_path=mtx,
        json_dir=json_dir,
    )
    assert created.get("status") == "success", created
    id_l = created["camera_l"]["id"]
    id_r = created["camera_r"]["id"]
    out = update_aisle_cameras(
        "aisle-up",
        {
            "name": "左路新",
            "path": "aisle-up-L2",
            "source_type": "rtsp_pull",
            "pull_url": "rtsp://10.0.0.8:554/live",
            "url": "rtsp://127.0.0.1:8554/aisle-up-L",
        },
        {
            "name": "右路新",
            "source_type": "publisher",
            "url": "rtsp://127.0.0.1:8554/aisle-up-R",
        },
        camera_file=cam_file,
        mediamtx_config_path=mtx,
        json_dir=json_dir,
    )
    assert out.get("status") == "success", out
    by_id = {c["id"]: c for c in load_cameras(cam_file)}
    assert set(by_id) == {id_l, id_r}
    assert by_id[id_l]["name"] == "左路新"
    assert by_id[id_l]["path"] == "aisle-up-L2"
    assert by_id[id_l]["pull_url"] == "rtsp://10.0.0.8:554/live"
    assert by_id[id_r]["name"] == "右路新"
    assert by_id[id_r]["source_type"] == "publisher"
    aisle = load_aisle("aisle-up", json_dir)
    assert aisle["cameras"]["L"]["camera_id"] == id_l
    missing = update_aisle_cameras(
        "no-such",
        {"name": "x"},
        {"name": "y"},
        camera_file=cam_file,
        mediamtx_config_path=mtx,
        json_dir=json_dir,
    )
    assert "不存在" in (missing.get("error") or "")


def test_rename_aisle_number_keeps_pk(tmp_path: Path, json_dir: str, monkeypatch: pytest.MonkeyPatch):
    from services.aisle_store import update_aisle_cameras
    import services.inference_container_service as infer_svc

    monkeypatch.setattr(infer_svc, "stop_inference_container", lambda *_a, **_k: {"status": "success"})
    cam_file = str(tmp_path / "cameras.json")
    mtx = str(tmp_path / "mediamtx.yml")
    Path(cam_file).write_text("[]", encoding="utf-8")
    created = create_aisle_with_cameras(
        "aisle-old",
        {
            "path": "aisle-old-L",
            "name": "左",
            "source_type": "rtsp_pull",
            "pull_url": "rtsp://10.0.0.1:554/s1",
        },
        {
            "path": "aisle-old-R",
            "name": "右",
            "source_type": "rtsp_pull",
            "pull_url": "rtsp://10.0.0.2:554/s1",
        },
        camera_file=cam_file,
        mediamtx_config_path=mtx,
        json_dir=json_dir,
    )
    assert created.get("status") == "success", created
    pk = created["aisle"]["id"]
    id_l = created["camera_l"]["id"]
    out = update_aisle_cameras(
        "aisle-old",
        {
            "name": "左",
            "source_type": "rtsp_pull",
            "pull_url": "rtsp://10.0.0.1:554/s1",
        },
        {
            "name": "右",
            "source_type": "rtsp_pull",
            "pull_url": "rtsp://10.0.0.2:554/s1",
        },
        camera_file=cam_file,
        mediamtx_config_path=mtx,
        json_dir=json_dir,
        new_aisle_id="aisle-new",
    )
    assert out.get("status") == "success", out
    assert out["aisle"]["aisle_id"] == "aisle-new"
    assert out["aisle"]["id"] == pk
    assert out.get("renamed_from") == "aisle-old"
    assert load_aisle("aisle-old", json_dir) is None
    assert load_aisle("aisle-new", json_dir)["cameras"]["L"]["camera_id"] == id_l
    assert camera_group(id_l, json_dir)["aisle_id"] == "aisle-new"


def test_apply_capture_sizes_writes_pixels_and_invalidates_solve_on_aspect_change(json_dir: str):
    from services.aisle_store import apply_capture_sizes, empty_aisle

    data = empty_aisle("aisle-sz")
    data["views"]["L"]["image_size"] = [1280, 720]
    data["views"]["L"]["walls"][0]["quad"] = [
        [160.0, 0.0], [1120.0, 0.0], [1120.0, 720.0], [160.0, 720.0],
    ]
    data["solved"] = {"ok": True, "cameras": {"L": {"f": 700.0, "cx": 640.0, "cy": 360.0}}}
    save_aisle(data, json_dir)
    out, changed = apply_capture_sizes(
        "aisle-sz",
        {"L": {"width": 960, "height": 720}},
        json_dir,
    )
    assert changed is True
    assert out["views"]["L"]["image_size"] == [960, 720]
    assert out["views"]["L"]["walls"][0]["quad"][0][0] == pytest.approx(0.0, abs=0.5)
    assert out["solved"].get("ok") is not True


def test_apply_capture_sizes_keeps_solved_on_uniform_scale(json_dir: str):
    from services.aisle_store import apply_capture_sizes, empty_aisle

    data = empty_aisle("aisle-scale")
    data["views"]["L"]["image_size"] = [1280, 720]
    data["views"]["R"]["image_size"] = [1280, 720]
    data["views"]["L"]["walls"][0]["quad"] = [[100.0, 50.0], [200.0, 50.0], [200.0, 150.0], [100.0, 150.0]]
    data["solved"] = {
        "ok": True,
        "cameras": {
            "L": {"f": 800.0, "cx": 640.0, "cy": 360.0},
            "R": {"f": 760.0, "cx": 640.0, "cy": 360.0},
        },
    }
    save_aisle(data, json_dir)
    out, changed = apply_capture_sizes(
        "aisle-scale",
        {"L": {"width": 640, "height": 360}, "R": {"width": 640, "height": 360}},
        json_dir,
    )
    assert changed is True
    assert out["solved"]["ok"] is True
    assert out["solved"]["cameras"]["L"]["f"] == pytest.approx(400.0)
    assert out["solved"]["cameras"]["L"]["cx"] == pytest.approx(320.0)
