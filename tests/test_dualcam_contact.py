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


def test_contact_src_excludes_mono():
    assert "Lmono" not in CONTACT_SRC
    assert "stereo" in CONTACT_SRC
