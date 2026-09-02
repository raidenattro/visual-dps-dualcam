"""对齐 pick-state：2D 短窗 + 3D 轻滤波压抖，分数不改。"""

from __future__ import annotations

import numpy as np

from dualcam.skel3d_smooth import (
    FOLLOW_SPEED_3D,
    LivePose2DSmoother,
    LivePose3DSmoother,
    _clamp_bone,
    _clamp_bones_causal,
    _gauss_smooth,
    _reject_speed,
    _restore_raw_bone_lengths,
    _one_euro_series,
    _speed_gated_blend,
    adapt_sigma,
    assign_tracks,
    copy_pose_pack,
    live_adapt_sigma,
    repair_arm_chains,
    smooth_frames,
    smooth_pose2d,
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
    mid = len(t) // 2
    assert abs(sm[mid, 0] - true[mid, 0]) < 0.03


def test_gauss_keeps_current_when_causal_window_is_sparse():
    t = np.array([0.0, 1.0], dtype=np.float64)
    x = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.4]], dtype=np.float64)
    mask = np.ones(2, dtype=bool)
    sm, sm_mask = _gauss_smooth(x, mask, t, 0.04, causal=True)
    assert sm_mask[1]
    assert abs(sm[1, 2] - 0.4) < 1e-9


def test_reject_speed_drops_one_frame_spike():
    t = np.arange(8) * 0.04
    x = np.zeros((8, 3))
    x[:, 1] = 1.0
    x[4] = [0.0, 1.0, 0.8]
    mask = np.ones(8, dtype=bool)
    _reject_speed(x, mask, t, vmax=4.0)
    assert not mask[4]
    assert mask[3] and mask[5]


def test_bone_clamp_pulls_flying_wrist():
    n = 4
    elbow = np.tile(np.array([0.0, 1.2, 0.0]), (n, 1))
    wrist = np.tile(np.array([0.8, 1.2, 0.0]), (n, 1))
    mp = np.ones(n, dtype=bool)
    md = np.ones(n, dtype=bool)
    _clamp_bone(elbow, wrist, mp, md, 0.24)
    dist = np.linalg.norm(wrist[0] - elbow[0])
    assert abs(dist - 0.24) < 1e-6
    assert wrist[0, 0] > 0


def test_clamp_bones_keeps_in_range_scale():
    """合法骨长不要被拉到成人中位，否则四肢和躯干比例会拧。"""
    n = 6
    shoulder = np.tile(np.array([0.0, 1.4, 0.0]), (n, 1))
    # 上臂 0.38m，落在 (0.18, 0.42) 内，比区间中位 0.30 长
    elbow = np.tile(np.array([0.38, 1.4, 0.0]), (n, 1))
    mp = np.ones(n, dtype=bool)
    md = np.ones(n, dtype=bool)
    _clamp_bones_causal(shoulder, elbow, mp, md, 0.18, 0.42, min_samples=16)
    d = float(np.linalg.norm(elbow[0] - shoulder[0]))
    assert abs(d - 0.38) < 1e-6


def test_clamp_bones_caps_flying_to_range():
    n = 3
    hip = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    knee = np.tile(np.array([1.2, 1.0, 0.0]), (n, 1))
    mp = np.ones(n, dtype=bool)
    md = np.ones(n, dtype=bool)
    _clamp_bones_causal(hip, knee, mp, md, 0.28, 0.55, min_samples=16)
    d = float(np.linalg.norm(knee[-1] - hip[-1]))
    assert abs(d - 0.55) < 1e-6


def test_restore_raw_bone_lengths_undoes_uneven_lag():
    """腕比肩多滤一截时，应收回本帧生骨长，而不是橡皮泥拉长。"""
    n = 1
    sho_sm = np.array([[0.0, 1.4, 0.0]], float)
    wr_sm = np.array([[0.0, 1.4, 0.6]], float)
    sho_raw = np.array([[0.0, 1.4, 0.0]], float)
    wr_raw = np.array([[0.0, 1.4, 0.28]], float)
    mk = np.ones(n, dtype=bool)
    _restore_raw_bone_lengths(
        {5: (sho_sm, mk), 7: (wr_sm, mk)},
        {5: (sho_raw, mk), 7: (wr_raw, mk)},
        bones=((5, 7),),
    )
    assert abs(float(np.linalg.norm(wr_sm[0] - sho_sm[0])) - 0.28) < 1e-6


def _torso_xyz(wrist_z: float) -> list:
    xyz = [None] * 17
    xyz[5] = [0.0, 1.4, 0.5]
    xyz[6] = [0.3, 1.4, 0.5]
    xyz[7] = [0.0, 1.25, 0.5 + 0.5 * wrist_z]
    xyz[9] = [0.0, 1.4, 0.5 + wrist_z]
    xyz[11] = [0.0, 1.0, 0.5]
    xyz[12] = [0.3, 1.0, 0.5]
    return xyz


def test_causal_sparse_keeps_raw_arm_shape():
    """7.5Hz 因果：腕在两帧间伸缩时，不得把骨头插值成中间长度。"""
    frames = []
    zs = [0.28, 0.70, 0.28, 0.70, 0.28, 0.70]
    for i, z in enumerate(zs):
        frames.append({
            "t": i * 0.133,
            "persons": [{"xyz": _torso_xyz(z), "vis": [1] * 17}],
        })
    smooth_frames(frames, causal=True, drop_short=False)
    last = frames[-1]["persons"][0]["xyz"]
    d = float(np.linalg.norm(np.array(last[9], float) - np.array(last[5], float)))
    assert abs(d - 0.70) < 0.03
    e = np.array(last[7], float)
    s = np.array(last[5], float)
    # 肘相对肩的偏移应保持本帧生抬点，不能被两帧平均
    assert abs((e[1] - s[1]) - (-0.15)) < 0.02
    assert abs((e[2] - s[2]) - 0.35) < 0.02


def _arm_xyz(sho, elb, wri):
    xyz = [None] * 17
    xyz[5], xyz[6] = [0.2, 1.4, 0.5], [0.5, 1.4, 0.5]
    xyz[11], xyz[12] = [0.2, 1.0, 0.5], [0.5, 1.0, 0.5]
    xyz[5] = list(sho)
    xyz[7] = list(elb)
    xyz[9] = list(wri)
    return xyz


def test_repair_arm_keeps_elbow_in_front():
    """伸手向前时肘在胸前是合法的，不能按胸口先验翻到身后。"""
    s = np.array([0.2, 1.4, 0.5])
    w = np.array([0.2, 1.15, 0.85])
    e_front = np.array([0.2, 1.28, 0.72])
    xyz = _arm_xyz(s, e_front, w)
    repair_arm_chains(xyz)
    e = np.asarray(xyz[7])
    assert e[2] > 0.55


def test_repair_arm_only_pulls_extreme_wrist():
    """只有肩腕明显撑破骨长才收腕；贴墙伸手不要沿肘收回。"""
    s = np.array([0.2, 1.4, 0.5])
    e = np.array([0.2, 1.25, 0.7])
    w = np.array([0.2, 1.2, 2.5])
    xyz = _arm_xyz(s, e, w)
    srcs = [None] * 17
    srcs[7], srcs[9] = "stereo", "Lmono"
    repair_arm_chains(xyz, srcs)
    w2 = np.asarray(xyz[9])
    assert float(np.linalg.norm(w2 - s)) < 1.0
    assert w2[2] > e[2]


def test_repair_arm_keeps_natural_hang():
    """自然下垂：不要拧。"""
    s = np.array([0.2, 1.4, 0.5])
    w = np.array([0.2, 1.05, 0.55])
    e = np.array([0.2, 1.22, 0.38])
    xyz = _arm_xyz(s, e, w)
    before = np.asarray(e).copy()
    repair_arm_chains(xyz)
    after = np.asarray(xyz[7])
    assert float(np.linalg.norm(after - before)) < 0.12


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
    assert frames[1]["persons"][1]["track_id"] == a0
    assert frames[1]["persons"][0]["track_id"] == b0


def test_smooth_frames_cuts_wrist_jitter():
    rng = np.random.default_rng(1)
    frames = []
    for i in range(40):
        t = i * 0.04
        base = np.array([0.2, 1.1, 0.4 + 0.15 * t])
        wrist = base + np.array([0.25, 0.05, 0.0]) + rng.normal(0, 0.05, 3)
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
    w0 = np.array(frames[0]["persons"][0]["xyz"][9], float)
    w1 = np.array(frames[-1]["persons"][0]["xyz"][9], float)
    assert w1[2] - w0[2] > 0.10


def test_smooth_pose2d_reduces_pixel_jitter_keeps_scores():
    rng = np.random.default_rng(2)
    pack = []
    for i in range(40):
        t = i * 0.04
        u = 400.0 + 2.0 * i
        k = np.zeros((17, 2))
        k[5] = [u, 200]
        k[6] = [u + 40, 200]
        k[11] = [u, 400]
        k[12] = [u + 40, 400]
        k[9] = [u + 20 + rng.normal(0, 6), 280 + rng.normal(0, 6)]
        s = np.full(17, 0.8)
        pack.append({
            "i": i,
            "t": t,
            "L": {"k": [k.copy()], "s": [s.copy()]},
            "R": {"k": [k.copy()], "s": [s.copy()]},
        })
    raw = [float(fr["L"]["k"][0][9][0]) for fr in pack]
    copied = copy_pose_pack(pack)
    info = smooth_pose2d(copied)
    assert info["n_wrist_in"] > 0
    sm = [float(fr["L"]["k"][0][9][0]) for fr in copied]
    jump_raw = np.max(np.abs(np.diff(raw)))
    jump_sm = np.max(np.abs(np.diff(sm)))
    assert jump_sm < 0.7 * jump_raw
    assert copied[-1]["L"]["k"][0][9][0] - copied[0]["L"]["k"][0][9][0] > 50
    assert np.allclose(copied[0]["L"]["s"][0], 0.8)
    raw10 = float(pack[10]["L"]["k"][0][9][0])
    smooth_pose2d(copied)
    assert float(pack[10]["L"]["k"][0][9][0]) == raw10


def test_torso_mean_and_empty():
    xyz = [None] * 17
    assert torso_centroid(xyz) is None
    xyz[5] = [0.0, 1.4, 0.2]
    c = torso_centroid(xyz)
    assert c is not None and abs(c[1] - 1.4) < 1e-9
    xyz[11] = [0.0, 0.9, 0.2]
    c = torso_centroid(xyz)
    assert c is not None and 0.9 < c[1] < 1.4


def test_adapt_sigma_scales_when_pose_is_sparse():
    assert adapt_sigma(0.07, 0.04) == 0.07
    assert adapt_sigma(0.07, 0.08) == 0.07 * 2.0
    assert adapt_sigma(0.07, 0.133) == 0.07 * 2.0


def test_live_adapt_sigma_covers_previous_pose_at_7hz():
    """7.5Hz 时质心窗必须看见上一帧；不再对各关节用这个加宽 σ。"""
    dt = 0.133
    sig = live_adapt_sigma(0.030, dt)
    assert sig >= 2.5 * dt / 3.0
    # 上一帧落在 3σ 内
    assert 3.0 * sig >= dt
    assert live_adapt_sigma(0.030, 0.04) < 0.04


def test_live_pose3d_cuts_joint_jitter_at_sparse_dt():
    """直播约 7.5Hz：平滑质心应压整身抖，但不能把姿态插值烂掉。"""
    rng = np.random.default_rng(4)
    sm = LivePose3DSmoother()
    raw_z = []
    out_z = []
    base = np.array([0.2, 1.1, 0.8], dtype=np.float64)
    for i in range(24):
        t = i * 0.133
        noise = rng.normal(0, 0.05, 3)
        p = (base + noise).tolist()
        xyz = [None] * 17
        for j in (5, 6, 11, 12, 9, 10):
            xyz[j] = list(p)
        raw_z.append(p[2])
        out = sm.update(t, [{"xyz": xyz, "preview": False, "wrist_alarm": {9: False, 10: False}}])
        oz = out[0]["xyz"][5]
        out_z.append(float(oz[2]))
    jump_raw = float(np.max(np.abs(np.diff(raw_z[8:]))))
    jump_sm = float(np.max(np.abs(np.diff(out_z[8:]))))
    assert jump_sm < 0.85 * jump_raw


def test_live_pose2d_cuts_wrist_jitter_at_sparse_dt():
    """直播 pose 约 7.5Hz，因果窗也应压腕点抽搐。"""
    rng = np.random.default_rng(3)
    sm = LivePose2DSmoother()
    raw_u = []
    out_u = []
    for i in range(24):
        t = i * 0.133
        u = 400.0 + 8.0 * i
        kpts = [[0.0, 0.0, 0.8] for _ in range(17)]
        kpts[5] = [u, 200, 0.8]
        kpts[6] = [u + 40, 200, 0.8]
        kpts[11] = [u, 400, 0.8]
        kpts[12] = [u + 40, 400, 0.8]
        wu = u + 20 + float(rng.normal(0, 8))
        kpts[9] = [wu, 280, 0.8]
        raw_u.append(wu)
        persons = sm.update(t, [{"keypoints": kpts}])
        out_u.append(float(persons[0]["keypoints"][9][0]))
    jump_raw = np.max(np.abs(np.diff(raw_u[8:])))
    jump_sm = np.max(np.abs(np.diff(out_u[8:])))
    assert jump_sm < 0.85 * jump_raw
    assert out_u[-1] - out_u[8] > 40
    assert persons[0]["keypoints"][9][2] == 0.8


def test_one_euro_cuts_jitter_follows_reach():
    """1-Euro：静止噪声下降，快速伸手仍能跟过去。"""
    rng = np.random.default_rng(6)
    t = np.arange(20, dtype=float) * 0.133
    rest = np.column_stack([80.0 + rng.normal(0, 10, 20), np.full(20, 280.0)])
    mk = np.ones(20, dtype=bool)
    sm = rest.copy()
    _one_euro_series(sm, mk, t)
    assert float(np.max(np.abs(np.diff(sm[4:, 0])))) < 0.85 * float(np.max(np.abs(np.diff(rest[4:, 0]))))
    reach = np.column_stack([80.0 + 25.0 * np.arange(20), np.full(20, 280.0)])
    out = reach.copy()
    _one_euro_series(out, mk, t)
    assert out[-1, 0] - out[4, 0] > 250


def test_speed_gated_follows_fast_keeps_slow():
    """快动作跟本帧；慢抖才掺上一帧。"""
    prev = np.array([0.0, 0.0, 0.0])
    fast = np.array([0.20, 0.0, 0.0])  # 0.20/0.133 ≈ 1.5 m/s > 0.85
    out = _speed_gated_blend(prev, fast, 0.133, FOLLOW_SPEED_3D)
    assert float(np.linalg.norm(out - fast)) < 1e-9
    slow = np.array([0.02, 0.0, 0.0])  # 0.15 m/s
    out = _speed_gated_blend(prev, slow, 0.133, FOLLOW_SPEED_3D, 0.42)
    assert 0.005 < float(out[0]) < 0.02


def test_dense_causal_matches_dump_joint_window():
    """15～25Hz 因果应走 dump 逐关节短窗，腕点抖应明显下降。"""
    rng = np.random.default_rng(5)
    frames = []
    for i in range(40):
        t = i * 0.04
        base = np.array([0.2, 1.1, 0.4 + 0.12 * t])
        wrist = base + np.array([0.25, 0.05, 0.0]) + rng.normal(0, 0.05, 3)
        xyz = [None] * 17
        xyz[5] = (base + [0.0, 0.35, 0.0]).tolist()
        xyz[6] = (base + [0.0, 0.35, 0.15]).tolist()
        xyz[7] = (base + [0.12, 0.15, 0.0]).tolist()
        xyz[9] = wrist.tolist()
        xyz[11] = (base + [0.0, -0.05, 0.0]).tolist()
        xyz[12] = (base + [0.0, -0.05, 0.15]).tolist()
        frames.append({"t": t, "persons": [{"xyz": xyz, "vis": [1] * 17}]})
    before = wrist_jump_stats(frames, 9)
    smooth_frames(frames, causal=True, drop_short=False)
    after = wrist_jump_stats(frames, 9)
    assert after["p90"] < 0.80 * before["p90"]
