"""3D 贴墙碰撞：伸进报警，停在通道不报。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dualcam.geom import contact_slots, make_layer_mesh, signed_wall_dist, wall_by_id
from dualcam.lift import CONTACT_SRC
from services.event_engine.dualcam_processor import DualcamProcessor, probe_wrist_contacts

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


def _ready_aisle(calib):
    meshes = []
    for m in calib.get("slot_meshes") or []:
        if not isinstance(m, dict):
            continue
        mesh = dict(m)
        wid = int(mesh.get("wall_id") or 0)
        mesh["shelf_code"] = str(mesh.get("shelf_code") or "").strip() or f"wall{wid}"
        meshes.append(mesh)
    return {
        "aisle_id": "t",
        "solved": calib["solved"],
        "cameras": {"L": {"camera_id": "cam1"}, "R": {"camera_id": "cam2"}},
        "slot_meshes": meshes,
        "views": calib.get("views") or {},
        "required_wall_ids": [int(m["wall_id"]) for m in meshes if m.get("wall_id")],
    }


def test_processor_drops_unchecked_wall_mesh(calib):
    aisle = _ready_aisle(calib)
    extra = dict((aisle["slot_meshes"] or [{}])[0] or {"wall_id": 2, "vertices": []})
    extra["wall_id"] = 2
    extra["shelf_code"] = "ignore-me"
    aisle["slot_meshes"] = list(aisle["slot_meshes"] or []) + [extra]
    aisle["required_wall_ids"] = [1]
    proc = DualcamProcessor(aisle)
    assert proc.required_walls == [1]
    assert all(int(m.get("wall_id") or 0) == 1 for m in proc.meshes)


def test_process_pair_empty_without_prior_hold(calib):
    proc = DualcamProcessor(_ready_aisle(calib))
    out = proc.process_pair({"frame_idx": 2, "persons": []}, {"frame_idx": 2, "persons": []})
    assert out["persons_3d"] == []
    assert proc._holds == []


def test_hold_keeps_then_clears_like_dump_skel3d():
    """dump 25fps：闪断沿用 0.32s（8 帧），第 9 帧空才丢掉。"""
    from dualcam.skel3d_smooth import DESIGN_DT, HOLD_SEC

    proc = DualcamProcessor({"aisle_id": "x", "solved": {"ok": False}, "cameras": {}})
    xyz = [[0.0, 1.0, 1.0] for _ in range(17)]
    person = {"xyz": xyz, "src": ["stereo"] * 17, "preview": False, "wrist_alarm": {9: False, 10: False}}
    first = proc._apply_hold([person], 0.0)
    assert len(first) == 1
    assert not first[0].get("held")
    n_hold = int(round(HOLD_SEC / DESIGN_DT))
    for i in range(n_hold):
        held = proc._apply_hold([], (i + 1) * DESIGN_DT)
        assert len(held) == 1, f"empty frame {i + 1} should hold"
        assert held[0].get("held") is True
        assert held[0].get("preview") is True
    assert proc._apply_hold([], (n_hold + 1) * DESIGN_DT) == []
    assert proc._holds == []
    assert proc._prev_xyz == {}
    assert proc._prefer == []


def test_hold_expires_in_two_live_pose_periods():
    """直播 ~7.5Hz：0.32s 只能续约 2 个 pose，不能冻满 8 帧。"""
    proc = DualcamProcessor({"aisle_id": "x", "solved": {"ok": False}, "cameras": {}})
    xyz = [[0.0, 1.0, 1.0] for _ in range(17)]
    person = {"xyz": xyz, "src": ["stereo"] * 17, "preview": False, "wrist_alarm": {9: False, 10: False}}
    proc._apply_hold([person], 0.0)
    assert len(proc._apply_hold([], 0.133)) == 1
    assert len(proc._apply_hold([], 0.266)) == 1
    assert proc._apply_hold([], 0.399) == []


def test_contact_src_excludes_mono():
    assert "Lmono" not in CONTACT_SRC
    assert "stereo" in CONTACT_SRC


def test_process_single_cold_start_does_not_lift_mono(calib):
    """对齐 dump_skel3d：没有上一帧立体人时，单路也不贴墙抬骨架。"""
    proc = DualcamProcessor(_ready_aisle(calib))
    kpts = [[640.0, 360.0, 0.95] for _ in range(17)]
    pose = {
        "frame_idx": 1,
        "persons": [{"keypoints": kpts}],
        "infer_width": 1280,
        "infer_height": 720,
    }
    for role in ("L", "R"):
        out = proc.process_single(role, pose)
        assert out.get("persons_3d") == []
        sk_key = "skeletons_l" if role == "L" else "skeletons_r"
        assert len(out.get(sk_key) or []) == 1


def test_process_single_follows_2d_with_hold_depth(calib):
    """有 hold 时单路应跟上当前检测，不要冻 3D、也不要清空 2D。"""
    proc = DualcamProcessor(_ready_aisle(calib))
    xyz = [[0.2, 1.1, 0.8] for _ in range(17)]
    proc._apply_hold([{
        "xyz": xyz,
        "src": ["stereo"] * 17,
        "preview": False,
        "wrist_alarm": {9: False, 10: False},
    }], 4.0)
    kpts = [[640.0, 360.0, 0.95] for _ in range(17)]
    pose = {
        "frame_idx": 4,
        "ts": 4.0,
        "persons": [{"keypoints": kpts}, {"keypoints": [[700.0, 400.0, 0.9] for _ in range(17)]}],
        "infer_width": 1280,
        "infer_height": 720,
    }
    out = proc.process_single("L", pose)
    people = out.get("persons_3d") or []
    assert len(people) == 1
    assert not people[0].get("held")
    assert people[0].get("preview") is True
    assert len(out.get("skeletons_l") or []) >= 1


def test_process_single_with_prefer_does_not_crash_on_numpy_pack(calib):
    """prefer 续帧时 pack['k'] 是 ndarray，不能用 `or []`。"""
    proc = DualcamProcessor(_ready_aisle(calib))
    kpts = [[640.0, 360.0, 0.95] for _ in range(17)]
    pose = {
        "frame_idx": 5,
        "ts": 5.0,
        "persons": [{"keypoints": kpts}],
        "infer_width": 1280,
        "infer_height": 720,
    }
    proc._prefer = [(np.array([640.0, 360.0]), np.array([640.0, 360.0]))]
    proc._apply_hold([{
        "xyz": [[0.2, 1.1, 0.8] for _ in range(17)],
        "src": ["stereo"] * 17,
        "preview": False,
        "wrist_alarm": {9: False, 10: False},
    }], 5.0)
    out = proc.process_single("L", pose)
    assert len(out.get("skeletons_l") or []) >= 1
    assert len(out.get("persons_3d") or []) >= 1


def _weak_pose(frame_idx: int) -> dict:
    """分数低于 KPT_MIN，NMS 为空，左右路配不上对。"""
    kpts = [[100.0, 100.0, 0.05] for _ in range(17)]
    return {
        "frame_idx": frame_idx,
        "ts": float(frame_idx),
        "persons": [{"keypoints": kpts}, {"keypoints": [[200.0, 120.0, 0.05] for _ in range(17)]}],
        "infer_width": 1280,
        "infer_height": 720,
    }


def test_unpaired_holds_one_person_not_l_and_r_mono(calib):
    """对齐 dump_skel3d：配不上时只续上一帧，不要把左右路各抬成两套乱骨架。"""
    proc = DualcamProcessor(_ready_aisle(calib))
    xyz = [[0.0, 1.0, 1.0] for _ in range(17)]
    proc._apply_hold([{
        "xyz": xyz,
        "src": ["stereo"] * 17,
        "preview": False,
        "wrist_alarm": {9: False, 10: False},
    }], 3.0)
    out = proc.process_pair(_weak_pose(3), _weak_pose(3))
    assert len(out["persons_3d"]) == 1
    assert out["persons_3d"][0].get("held") is True
    assert out["preview"] is True
    assert out["skeletons_l"] == []
    assert out["skeletons_r"] == []


def test_unpaired_does_not_overlay_all_detections(calib):
    """配不上时 2D 不要把两路全部检测画上去。"""
    proc = DualcamProcessor(_ready_aisle(calib))
    out = proc.process_pair(_weak_pose(1), _weak_pose(1))
    assert out["skeletons_l"] == []
    assert out["skeletons_r"] == []


def test_flying_wrist_clamped_when_far_from_shoulder():
    proc = DualcamProcessor({"aisle_id": "x", "solved": {"ok": False}, "cameras": {}})
    xyz = [None] * 17
    xyz[5] = [0.0, 1.4, 0.5]
    xyz[9] = [1.2, 1.4, 0.5]
    srcs = [None] * 17
    srcs[9] = "stereo"
    toks = {9: ["S:1"], 10: []}
    proc._clamp_flying_wrists(xyz, srcs)
    w = np.asarray(xyz[9], float)
    sh = np.asarray(xyz[5], float)
    assert abs(float(np.linalg.norm(w - sh)) - 0.85) < 1e-6
    assert toks[9] == ["S:1"]


def test_flying_elbow_clamped_when_far_from_shoulder():
    proc = DualcamProcessor({"aisle_id": "x", "solved": {"ok": False}, "cameras": {}})
    xyz = [None] * 17
    xyz[5] = [0.0, 1.4, 0.5]
    xyz[7] = [1.0, 1.4, 0.5]
    srcs = [None] * 17
    proc._clamp_flying_wrists(xyz, srcs)
    e = np.asarray(xyz[7], float)
    sh = np.asarray(xyz[5], float)
    assert abs(float(np.linalg.norm(e - sh)) - 0.42) < 1e-6


def test_collect_alarm_tokens_includes_held():
    proc = DualcamProcessor({"aisle_id": "x", "solved": {"ok": False}, "cameras": {}})
    people = [
        {"preview": True, "wrist_alarm": {9: True, 10: False}, "alarm_tokens": ["S1:1-1"]},
        {"held": True, "preview": True, "wrist_alarm": {9: True}, "alarm_tokens": ["S2:2-2"]},
    ]
    assert proc._collect_alarm_tokens(people) == ["S1:1-1", "S2:2-2"]


def test_contact_m_zero_is_exact(calib):
    aisle = _ready_aisle(calib)
    aisle["contact_m"] = 0
    proc = DualcamProcessor(aisle)
    assert proc.contact_m == 0.0


def test_contact_slots_nearest_cell_when_yz_miss(calib):
    """Y/Z 略出格但 d < contact_m 时，用最近货格兜底。"""
    solved = calib["solved"]
    wall = wall_by_id(solved, 1)
    mesh = make_layer_mesh(1, wall["corners"], n_layers=4, cols=4)
    p0 = np.array(wall["corners"][0], float)
    inward = np.array([1.0 if int(wall["sign"]) < 0 else -1.0, 0.0, 0.0])
    into = p0 - 0.05 * inward
    # 故意偏到格缝外
    into[1] = float(np.mean([c[1] for c in wall["corners"]])) + 999.0
    into[2] = float(np.mean([c[2] for c in wall["corners"]]))
    hits = contact_slots(into, [mesh], solved, contact_m=0.05)
    assert hits
    assert hits[0].get("nearest") is True


def test_hold_keeps_alarm_tokens():
    proc = DualcamProcessor({"aisle_id": "x", "solved": {"ok": False}, "cameras": {}})
    xyz = [[0.0, 1.0, 1.0] for _ in range(17)]
    person = {
        "xyz": xyz,
        "src": ["stereo"] * 17,
        "preview": False,
        "wrist_alarm": {9: True, 10: False},
        "alarm_tokens": ["S1:1-1"],
    }
    proc._apply_hold([person], 0.0)
    held = proc._apply_hold([], 0.04)
    assert len(held) == 1
    assert held[0].get("held") is True
    assert held[0].get("wrist_alarm", {}).get(9) is True
    assert held[0].get("alarm_tokens") == ["S1:1-1"]


def test_token_without_colon_is_kept(calib):
    """无 shelf_code 时 contact_slots 仍能命中货格。"""
    solved = calib["solved"]
    wall = wall_by_id(solved, 1)
    mesh = make_layer_mesh(1, wall["corners"], n_layers=4, cols=4)
    mesh.pop("shelf_code", None)
    p0 = np.array(wall["corners"][0], float)
    inward = np.array([1.0 if int(wall["sign"]) < 0 else -1.0, 0.0, 0.0])
    into = p0 - 0.05 * inward
    into[1] = float(np.mean([c[1] for c in wall["corners"]]))
    into[2] = float(np.mean([c[2] for c in wall["corners"]]))
    hits = contact_slots(into, [mesh], solved, contact_m=0.03)
    assert hits


def test_probe_wrist_contacts_reports_src_d_cell(calib):
    solved = calib["solved"]
    wall = wall_by_id(solved, 1)
    mesh = make_layer_mesh(1, wall["corners"], n_layers=4, cols=4)
    mesh["shelf_code"] = "S1"
    p0 = np.array(wall["corners"][0], float)
    inward = np.array([1.0 if int(wall["sign"]) < 0 else -1.0, 0.0, 0.0])
    into = p0 - 0.05 * inward
    into[1] = float(np.mean([c[1] for c in wall["corners"]]))
    into[2] = float(np.mean([c[2] for c in wall["corners"]]))
    xyz = [None] * 17
    srcs = [None] * 17
    xyz[9] = into.tolist()
    srcs[9] = "stereo"
    probes = probe_wrist_contacts(
        xyz, srcs, {9: True, 10: False}, [mesh], solved, contact_m=0.05,
    )
    lw = next(p for p in probes if p["wrist"] == "L")
    assert lw["src"] == "stereo"
    assert lw["d"] is not None and lw["d"] < 0.05
    assert lw["cell"]
    assert lw["wrist_alarm"] is True


def test_process_pair_preview_emits_lift_follow_tokens(calib):
    """pick_pairs 失败时，_lift_follow 已算的 token 应出现在 alarm_collisions。"""
    proc = DualcamProcessor(_ready_aisle(calib))
    xyz = [[0.2, 1.1, 0.8] for _ in range(17)]
    proc._prefer = [(np.array([640.0, 360.0]), np.array([640.0, 360.0]))]
    proc._apply_hold([{
        "xyz": xyz,
        "src": ["stereo"] * 17,
        "preview": False,
        "wrist_alarm": {9: False, 10: False},
    }], 1.0)

    def _fake_lift_joints(kl, sl, kr, sr, prev_key, prev_xyz=None):
        wrist_tok = {9: ["WALL1:3-2"], 10: []}
        return xyz, ["stereo"] * 17, wrist_tok

    proc._lift_joints = _fake_lift_joints  # type: ignore[method-assign]
    # 躯干落在 prefer 附近，确保 _lift_follow 会调用 mock
    kpts = [[640.0, 360.0, 0.95] for _ in range(17)]
    pose = {
        "frame_idx": 2,
        "ts": 2.0,
        "persons": [{"keypoints": kpts}, {"keypoints": [[645.0, 365.0, 0.95] for _ in range(17)]}],
        "infer_width": 1280,
        "infer_height": 720,
    }
    out = proc.process_pair(pose, pose)
    assert out["preview"] is True
    assert out["alarm_collisions"] == ["WALL1:3-2"]
    assert out["persons_3d"][0]["wrist_alarm"][9] is True
