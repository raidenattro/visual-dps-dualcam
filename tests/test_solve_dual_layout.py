"""同侧 / 对向双路标定：角序与法向距先验。"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.solve_scene import (
    LAYOUT_OPPOSITE,
    LAYOUT_SAME_SIDE,
    _OPP_CORNER,
    _SAME_CORNER,
    _cam_x_guess,
    _corner_order,
    _normalize_layout,
    _wall_corners,
    _wall_dist,
    cam_axes,
    solve,
    solve_dual,
)

IW, IH = 1280, 720
AISLE, WIDTH, HEIGHT, BASE = 2.0, 2.2, 2.0, 0.0


def _project_world(pts, C, pitch, yaw, roll, fov_h, img_w=IW, img_h=IH):
    f = (img_w / 2.0) / math.tan(math.radians(fov_h / 2.0))
    right, down, fwd = cam_axes(math.radians(pitch), math.radians(yaw), math.radians(roll))
    v = np.asarray(pts, float) - np.asarray(C, float)
    zc = v @ fwd
    cx, cy = img_w / 2.0, img_h / 2.0
    return np.column_stack([cx + f * (v @ right) / zc, cy + f * (v @ down) / zc])


def _aisle_walls(znear=0.0):
    spec = {"width": WIDTH, "height": HEIGHT, "base": BASE}
    return (
        _wall_corners(spec, -1, AISLE, znear),
        _wall_corners(spec, 1, AISLE, znear),
    )


def _view(name, C, pitch, yaw, prior, fov_h=90.0):
    w1, w2 = _aisle_walls()
    q1 = _project_world(w1, C, pitch, yaw, 0.0, fov_h).tolist()
    q2 = _project_world(w2, C, pitch, yaw, 0.0, fov_h).tolist()
    return {
        "name": name,
        "image_size": [IW, IH],
        "prior": prior,
        "walls": [
            {"wall_id": 1, "width": WIDTH, "height": HEIGHT, "base": BASE, "quad": q1},
            {"wall_id": 2, "width": WIDTH, "height": HEIGHT, "base": BASE, "quad": q2},
        ],
    }


def test_layout_helpers():
    assert _normalize_layout(None) == LAYOUT_OPPOSITE
    assert _normalize_layout("same-side") == LAYOUT_SAME_SIDE
    assert _corner_order(LAYOUT_OPPOSITE) == _OPP_CORNER
    assert _corner_order(LAYOUT_SAME_SIDE) == _SAME_CORNER
    prior = {"wallDist81": 0.8, "wallDist2": 1.2}
    assert _wall_dist(prior, 1) == 0.8
    assert _wall_dist(prior, 2) == 1.2
    assert _wall_dist({"wallDists": {1: 0.5}}, 1) == 0.5
    xs = _cam_x_guess(
        [{"wall_id": 1}, {"wall_id": 2}],
        [-1, 1],
        2.0,
        {"wallDist1": 0.8, "wallDist2": 1.2},
    )
    assert abs(xs - (-0.2)) < 1e-9


def test_wall_dist_pins_camx_off_center():
    """有两侧法向距时 camX 应离开巷道中线，贴近卷尺。"""
    C = np.array([-0.2, 2.84, -1.56])
    view = _view(
        "L", C, 45.0, 0.0,
        {"camH": 2.84, "camDist": 1.56, "wallDist1": 0.8, "wallDist2": 1.2, "pitch": 45, "yaw": 0},
    )
    res = solve({"aisle": AISLE, "prior": view["prior"], "walls": view["walls"]}, IW, IH)
    assert res["ok"]
    assert abs(res["camera"]["camX"] + 0.2) < 0.12
    assert abs(res["camera"]["camH"] - 2.84) < 0.15


def test_same_side_keeps_cameras_facing_same_way():
    """同端同向、不同站位：对齐后两路都朝 +Z，沿巷道错开。"""
    c_l = np.array([-0.2, 2.84, -1.56])
    c_r = np.array([0.25, 2.90, -2.80])
    payload = {
        "aisle": AISLE,
        "layout": "same_side",
        "views": [
            _view("L", c_l, 45.0, 0.0, {
                "camH": 2.84, "camDist": 1.56, "pitch": 45, "yaw": 0,
                "wallDist1": 0.8, "wallDist2": 1.2,
            }),
            _view("R", c_r, 42.0, 3.0, {
                "camH": 2.90, "camDist": 2.80, "pitch": 42, "yaw": 3,
                "wallDist1": 1.25, "wallDist2": 0.75,
            }),
        ],
    }
    res = solve_dual(payload)
    assert res["ok"], res.get("error")
    assert res["layout"] == LAYOUT_SAME_SIDE
    assert res["align_rms_m"] < 0.08
    rr = (res.get("wall_reproj_px") or {}).get("R") or {}
    assert rr.get(2, 999) < 2.0, f"右路墙2 重投影 {rr.get(2)} px"
    rr = (res.get("wall_reproj_px") or {}).get("R") or {}
    assert rr.get(2, 999) < 2.0, f"右路墙2 重投影应贴近标注，实际 {rr.get(2)} px"
    fwd_l = np.array(res["cameras"]["L"]["fwd"])
    fwd_r = np.array(res["cameras"]["R"]["fwd"])
    assert fwd_l[2] > 0.5 and fwd_r[2] > 0.5
    assert float(fwd_l @ fwd_r) > 0.7
    dz = abs(res["cameras"]["L"]["C"][2] - res["cameras"]["R"]["C"][2])
    assert abs(dz - 1.24) < 0.35


def test_wrong_opposite_layout_on_same_side_pair():
    """同侧数据误用对向角序：不是尺度炸掉，就是两路被拧成对打。"""
    c_l = np.array([-0.2, 2.84, -1.56])
    c_r = np.array([0.25, 2.90, -2.80])
    views = [
        _view("L", c_l, 45.0, 0.0, {
            "camH": 2.84, "camDist": 1.56, "pitch": 45, "yaw": 0,
            "wallDist1": 0.8, "wallDist2": 1.2,
        }),
        _view("R", c_r, 42.0, 3.0, {
            "camH": 2.90, "camDist": 2.80, "pitch": 42, "yaw": 3,
            "wallDist1": 1.25, "wallDist2": 0.75,
        }),
    ]
    wrong = solve_dual({"aisle": AISLE, "layout": "opposite", "views": views})
    assert wrong.get("layout") == LAYOUT_OPPOSITE
    if not wrong.get("ok"):
        assert abs(float(wrong.get("align_scale") or 0) - 1.0) > 0.1
        return
    fwd_l = np.array(wrong["cameras"]["L"]["fwd"])
    fwd_r = np.array(wrong["cameras"]["R"]["fwd"])
    assert float(fwd_l @ fwd_r) < 0.3


def test_same_side_one_shared_wall():
    """左右路都只标墙1：仍能对齐，基线保持，另一面不出现。"""
    c_l = np.array([-0.55, 2.84, -1.40])
    c_r = np.array([0.70, 2.90, -1.10])
    views = [
        _view("L", c_l, 45.0, 0.0, {
            "camH": 2.84, "camDist": 1.40, "pitch": 45, "yaw": 0, "wallDist1": 0.45,
        }),
        _view("R", c_r, 42.0, 4.0, {
            "camH": 2.90, "camDist": 1.10, "pitch": 42, "yaw": 4, "wallDist1": 1.70,
        }),
    ]
    for v in views:
        for w in v["walls"]:
            if w["wall_id"] != 1:
                w["quad"] = []
    res = solve_dual({"aisle": AISLE, "layout": "same_side", "views": views})
    assert res["ok"], res.get("error")
    assert res["align_wall_ids"] == [1]
    assert [w["wall_id"] for w in res["walls"]] == [1]
    assert res["align_rms_m"] < 0.05
    got = np.array(res["cameras"]["R"]["C"]) - np.array(res["cameras"]["L"]["C"])
    true = c_r - c_l
    assert abs(float(np.linalg.norm(got)) - float(np.linalg.norm(true))) < 0.05
    fwd_l = np.array(res["cameras"]["L"]["fwd"])
    fwd_r = np.array(res["cameras"]["R"]["fwd"])
    assert float(fwd_l @ fwd_r) > 0.7


def test_same_side_different_walls_do_not_align():
    """左路只标墙1、右路只标墙2：没有共同墙，不能硬对齐。"""
    c_l = np.array([-0.2, 2.84, -1.56])
    c_r = np.array([0.25, 2.90, -2.80])
    left = _view("L", c_l, 45.0, 0.0, {
        "camH": 2.84, "camDist": 1.56, "pitch": 45, "yaw": 0, "wallDist1": 0.8,
    })
    right = _view("R", c_r, 42.0, 3.0, {
        "camH": 2.90, "camDist": 2.80, "pitch": 42, "yaw": 3, "wallDist2": 0.75,
    })
    for w in left["walls"]:
        if w["wall_id"] != 1:
            w["quad"] = []
    for w in right["walls"]:
        if w["wall_id"] != 2:
            w["quad"] = []
    res = solve_dual({"aisle": AISLE, "layout": "same_side", "views": [left, right]})
    assert not res.get("ok")
    assert "同一面墙" in (res.get("error") or "")


def test_default_layout_is_opposite():
    c_l = np.array([-0.2, 2.84, -1.56])
    c_r = np.array([0.25, 2.90, -2.80])
    views = {
        "L": _view("L", c_l, 45.0, 0.0, {
            "camH": 2.84, "camDist": 1.56, "pitch": 45, "yaw": 0,
            "wallDist1": 0.8, "wallDist2": 1.2,
        }),
        "R": _view("R", c_r, 42.0, 3.0, {
            "camH": 2.90, "camDist": 2.80, "pitch": 42, "yaw": 3,
            "wallDist1": 1.25, "wallDist2": 0.75,
        }),
    }
    res = solve_dual({"aisle": AISLE, "views": views})
    assert res.get("layout") == LAYOUT_OPPOSITE


def test_existing_1_3_opposite_still_solves():
    """已入库的对向标定不传 layout，应仍能解且两路朝向相反。"""
    path = ROOT / "output/calib/dual_1-3.json"
    if not path.is_file():
        pytest.skip("缺少 dual_1-3.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    views = data.get("views") or {}
    if "L" not in views or "R" not in views:
        pytest.skip("dual_1-3.json 没有左右路")
    res = solve_dual({
        "aisle": data.get("aisle"),
        "prior": data.get("prior"),
        "views": views,
    })
    assert res["ok"], res.get("error")
    assert res["layout"] == LAYOUT_OPPOSITE
    fwd_l = np.array(res["cameras"]["L"]["fwd"])
    fwd_r = np.array(res["cameras"]["R"]["fwd"])
    assert float(fwd_l @ fwd_r) < 0.0
