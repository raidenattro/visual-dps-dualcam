"""3D 贴墙碰撞：伸进报警，停在通道不报。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dualcam.geom import contact_slots, make_layer_mesh, signed_wall_dist, wall_by_id
from dualcam.lift import CONTACT_SRC
from services.event_engine.dualcam_processor import DualcamProcessor

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "dual_1-3.json"


@pytest.fixture()
def calib():
    if not FIXTURE.is_file():
        pytest.skip("缺少 fixtures/dual_1-3.json")
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    solved = data.get("solved") or {}
    if not solved.get("ok"):
        pytest.skip("标定无效")
    return data


def test_into_wall_hits_slot(calib):
    solved = calib["solved"]
    wall = wall_by_id(solved, 1)
    mesh = make_layer_mesh(1, wall["corners"], n_layers=4, cols=4)
    # 贴墙内侧一点点（伸进）
    p0 = np.array(wall["corners"][0], float)
    inward = np.array([1.0 if int(wall["sign"]) < 0 else -1.0, 0.0, 0.0])
    into = p0 - 0.05 * inward
    into[1] = float(np.mean([c[1] for c in wall["corners"]]))
    into[2] = float(np.mean([c[2] for c in wall["corners"]]))
    hits = contact_slots(into, [mesh], solved, contact_m=0.0)
    assert hits, "伸进墙面应命中货格"


def test_in_aisle_no_hit(calib):
    solved = calib["solved"]
    wall = wall_by_id(solved, 1)
    mesh = make_layer_mesh(1, wall["corners"], n_layers=4, cols=4)
    p0 = np.array(wall["corners"][0], float)
    inward = np.array([1.0 if int(wall["sign"]) < 0 else -1.0, 0.0, 0.0])
    far = p0 + 0.50 * inward
    far[1] = float(np.mean([c[1] for c in wall["corners"]]))
    far[2] = float(np.mean([c[2] for c in wall["corners"]]))
    assert signed_wall_dist(far, wall) > 0.2
    assert contact_slots(far, [mesh], solved, contact_m=0.0) == []


def test_processor_empty_without_solve():
    proc = DualcamProcessor({"aisle_id": "x", "solved": {"ok": False}, "cameras": {}})
    out = proc.process_pair({"frame_idx": 1, "persons": []}, {"frame_idx": 1, "persons": []})
    assert out["alarm_collisions"] == []
    assert out["persons_3d"] == []
    preview = proc.process_single("L", {"frame_idx": 1, "persons": []})
    assert preview["persons_3d"] == []
    assert preview["alarm_collisions"] == []


def test_process_pair_empty_without_prior_hold(calib):
    aisle = {
        "aisle_id": "t",
        "solved": calib["solved"],
        "cameras": {"L": {"camera_id": "cam1"}, "R": {"camera_id": "cam2"}},
        "slot_meshes": calib.get("slot_meshes") or [],
        "views": calib.get("views") or {},
    }
    proc = DualcamProcessor(aisle)
    out = proc.process_pair({"frame_idx": 2, "persons": []}, {"frame_idx": 2, "persons": []})
    assert out["persons_3d"] == []
    assert proc._holds == []


def test_hold_keeps_then_clears_like_dump_skel3d():
    """闪断沿用 8 帧，第 9 帧空才丢掉（dump_skel3d.HOLD_FRAMES）。"""
    proc = DualcamProcessor({"aisle_id": "x", "solved": {"ok": False}, "cameras": {}})
    xyz = [[0.0, 1.0, 1.0] for _ in range(17)]
    person = {"xyz": xyz, "src": ["stereo"] * 17, "preview": False, "wrist_alarm": {9: False, 10: False}}
    first = proc._apply_hold([person])
    assert len(first) == 1
    assert not first[0].get("held")
    for i in range(8):
        held = proc._apply_hold([])
        assert len(held) == 1, f"empty frame {i + 1} should hold"
        assert held[0].get("held") is True
        assert held[0].get("preview") is True
    assert proc._apply_hold([]) == []
    assert proc._holds == []
    assert proc._prev_xyz == {}
    assert proc._prefer == []


def test_contact_src_excludes_mono():
    assert "Lmono" not in CONTACT_SRC
    assert "stereo" in CONTACT_SRC


def test_process_single_skips_face_joints(calib):
    """单路预览不得抬鼻子/眼睛，否则 3D 会出现射向墙面的长线。"""
    aisle = {
        "aisle_id": "t",
        "solved": calib["solved"],
        "cameras": {"L": {"camera_id": "cam1"}, "R": {"camera_id": "cam2"}},
        "slot_meshes": calib.get("slot_meshes") or [],
        "views": calib.get("views") or {},
    }
    proc = DualcamProcessor(aisle)
    kpts = [[640.0, 360.0, 0.95] for _ in range(17)]
    pose = {
        "frame_idx": 1,
        "persons": [{"keypoints": kpts}],
        "infer_width": 1280,
        "infer_height": 720,
    }
    for role in ("L", "R"):
        people = proc.process_single(role, pose).get("persons_3d") or []
        if not people:
            continue
        xyz = people[0].get("xyz") or []
        for ji in range(5):
            assert xyz[ji] is None, f"{role} 单路不应抬五官关节 {ji}"
