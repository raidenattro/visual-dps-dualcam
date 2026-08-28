"""3D 骨架时序平滑：压抖、零相位、骨长夹紧。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.skel3d_smooth import (
    _clamp_bone,
    _gauss_smooth,
    _reject_speed,
    assign_tracks,
    smooth_frames,
    torso_centroid,
    wrist_jump_stats,
)


def test_gauss_reduces_noise_without_lag():
    t = np.linspace(0, 2.0, 51)
    true = np.stack([0.4 * t, np.ones_like(t), np.zeros_like(t)], axis=1)
    rng = np.random.default_rng(0)
    noisy = true + rng.normal(0, 0.04, true.shape)
    mask = np.ones(len(t), dtype=bool)
    sm, sm_mask = _gauss_smooth(noisy, mask, t, 0.10)
    assert sm_mask.all()
    rms_in = float(np.sqrt(np.mean((noisy - true) ** 2)))
    rms_out = float(np.sqrt(np.mean((sm - true) ** 2)))
    assert rms_out < 0.55 * rms_in
    # 零相位：中点不该被单向 EMA 拖在后面
    mid = len(t) // 2
    assert abs(sm[mid, 0] - true[mid, 0]) < 0.03


def test_reject_speed_drops_one_frame_spike():
    t = np.arange(8) * 0.04
    x = np.zeros((8, 3))
    x[:, 1] = 1.0
    x[4] = [0.0, 1.0, 0.8]  # 一帧飞 0.8m → 20 m/s
    mask = np.ones(8, dtype=bool)
    _reject_speed(x, mask, t, vmax=4.0)
    assert not mask[4]
    assert mask[3] and mask[5]


def test_bone_clamp_pulls_flying_wrist():
    n = 4
    elbow = np.tile(np.array([0.0, 1.2, 0.0]), (n, 1))
    wrist = np.tile(np.array([0.8, 1.2, 0.0]), (n, 1))  # 80cm 前臂
    mp = np.ones(n, dtype=bool)
    md = np.ones(n, dtype=bool)
    _clamp_bone(elbow, wrist, mp, md, 0.24)
    dist = np.linalg.norm(wrist[0] - elbow[0])
    assert abs(dist - 0.24) < 1e-6
    assert wrist[0, 0] > 0  # 方向不变


def test_assign_tracks_keeps_two_people():
    def person(x, y, z, tid=None):
        xyz = [None] * 17
        for i in (5, 6, 11, 12):
            xyz[i] = [x, y, z]
        p = {"xyz": xyz}
        if tid is not None:
            p["track_id"] = tid
        return p

    frames = [
        {"t": 0.00, "persons": [person(0.0, 1.0, 0.5), person(1.5, 1.0, 0.5)]},
        {"t": 0.04, "persons": [person(1.51, 1.0, 0.5), person(0.02, 1.0, 0.5)]},
    ]
    n = assign_tracks(frames)
    assert n == 2
    a0 = frames[0]["persons"][0]["track_id"]
    b0 = frames[0]["persons"][1]["track_id"]
    # 第二帧人序对调，track 应跟着位置走
    assert frames[1]["persons"][1]["track_id"] == a0
    assert frames[1]["persons"][0]["track_id"] == b0


def test_smooth_frames_cuts_wrist_jitter():
    rng = np.random.default_rng(1)
    frames = []
    true_w = []
    for i in range(40):
        t = i * 0.04
        base = np.array([0.2, 1.1, 0.4 + 0.15 * t])  # 沿巷道慢走
        wrist = base + np.array([0.25, 0.05, 0.0]) + rng.normal(0, 0.05, 3)
        true_w.append(base + np.array([0.25, 0.05, 0.0]))
        xyz = [None] * 17
        xyz[5] = (base + [0.0, 0.35, 0.0]).tolist()
        xyz[6] = (base + [0.0, 0.35, 0.15]).tolist()
        xyz[7] = (base + [0.12, 0.15, 0.0]).tolist()
        xyz[9] = wrist.tolist()
        xyz[11] = (base + [0.0, -0.05, 0.0]).tolist()
        xyz[12] = (base + [0.0, -0.05, 0.15]).tolist()
        frames.append({"t": t, "persons": [{"xyz": xyz, "vis": [1] * 17}]})
    before = wrist_jump_stats(frames, 9)
    smooth_frames(frames)
    after = wrist_jump_stats(frames, 9)
    assert after["p90"] < 0.65 * before["p90"]
    assert after["frac_gt_0.15"] <= before["frac_gt_0.15"]
    # 慢走趋势还在
    w0 = np.array(frames[0]["persons"][0]["xyz"][9], float)
    w1 = np.array(frames[-1]["persons"][0]["xyz"][9], float)
    assert w1[2] - w0[2] > 0.10


def test_torso_mean_and_empty():
    xyz = [None] * 17
    assert torso_centroid(xyz) is None
    xyz[5] = [0.0, 1.4, 0.2]
    # 只有一个点时退化为该点，方便单人遮挡帧还能续上 track
    c = torso_centroid(xyz)
    assert c is not None and abs(c[1] - 1.4) < 1e-9
    xyz[11] = [0.0, 0.9, 0.2]
    c = torso_centroid(xyz)
    assert c is not None
    assert abs(c[1] - 1.15) < 1e-9
