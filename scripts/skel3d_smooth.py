"""双路 3D 骨架时序平滑。只用于 /play 验证，不进推理。

腕点是贴墙判定的输入，三角化又会把 2D 噪声放大，所以腕/肘比躯干滤得更狠。
离线零相位（时间高斯），伸手动作不往后拖。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scripts.dualcam_lift import LWRIST, RWRIST, signed_x

# COCO-17
LSHO, RSHO, LELB, RELB = 5, 6, 7, 8
LHIP, RHIP = 11, 12
N_JOINTS = 17
TORSO = (LSHO, RSHO, LHIP, RHIP)
# 近端 → 远端，平滑之后把远端收到骨长中位上，避免腕点飞离肘
ARM_BONES = ((LSHO, LELB), (LELB, LWRIST), (RSHO, RELB), (RELB, RWRIST))
BONE_LEN_RANGE = {
    (LSHO, LELB): (0.18, 0.42),
    (RSHO, RELB): (0.18, 0.42),
    (LELB, LWRIST): (0.15, 0.38),
    (RELB, RWRIST): (0.15, 0.38),
}

# 秒。腕 100ms 能压掉 10Hz 级抖，0.4s 级伸手还在。
SIGMA_BODY = 0.045
SIGMA_ELBOW = 0.070
SIGMA_WRIST = 0.100
VMAX_BODY = 3.0
VMAX_WRIST = 4.0
TRACK_MAX_M = 0.60
TRACK_MAX_GAP_S = 0.48
BONE_TOL = 0.12  # 相对中位，超出才拉回
BONE_MIN_SAMPLES = 16


def _as3(p) -> np.ndarray | None:
    if p is None:
        return None
    a = np.asarray(p, dtype=np.float64).reshape(-1)
    if a.size < 3 or not np.all(np.isfinite(a[:3])):
        return None
    return a[:3].copy()


def torso_centroid(xyz: list) -> np.ndarray | None:
    pts = [_as3(xyz[i]) for i in TORSO if xyz and i < len(xyz)]
    pts = [p for p in pts if p is not None]
    if len(pts) < 2:
        pts = [_as3(p) for p in (xyz or [])]
        pts = [p for p in pts if p is not None]
    if not pts:
        return None
    return np.mean(pts, axis=0)


def joint_sigma(j: int) -> float:
    if j in (LWRIST, RWRIST):
        return SIGMA_WRIST
    if j in (LELB, RELB):
        return SIGMA_ELBOW
    return SIGMA_BODY


def joint_vmax(j: int) -> float:
    if j in (LWRIST, RWRIST, LELB, RELB):
        return VMAX_WRIST
    return VMAX_BODY


@dataclass
class _Track:
    tid: int
    last_torso: np.ndarray
    last_t: float
    members: list[tuple[int, int]] = field(default_factory=list)


def assign_tracks(
    frames: list[dict],
    *,
    max_dist: float = TRACK_MAX_M,
    max_gap_s: float = TRACK_MAX_GAP_S,
) -> int:
    """按躯干 3D 位置把跨帧的人串成 track，写入 person['track_id']。返回 track 数。"""
    active: list[_Track] = []
    next_id = 0
    for fi, fr in enumerate(frames):
        t = float(fr.get("t") or 0.0)
        persons = fr.get("persons") or []
        cents: list[np.ndarray | None] = []
        for p in persons:
            cents.append(torso_centroid(p.get("xyz") or []))
        alive = [tr for tr in active if (t - tr.last_t) <= max_gap_s]
        cands: list[tuple[float, int, int]] = []
        for pi, c in enumerate(cents):
            if c is None:
                continue
            for ti, tr in enumerate(alive):
                cands.append((float(np.linalg.norm(c - tr.last_torso)), pi, ti))
        cands.sort()
        used_p: set[int] = set()
        used_t: set[int] = set()
        for dist, pi, ti in cands:
            if pi in used_p or ti in used_t or dist > max_dist:
                continue
            tr = alive[ti]
            tr.members.append((fi, pi))
            tr.last_torso = cents[pi]  # type: ignore[assignment]
            tr.last_t = t
            persons[pi]["track_id"] = tr.tid
            used_p.add(pi)
            used_t.add(ti)
        for pi, c in enumerate(cents):
            if pi in used_p:
                continue
            seed = c if c is not None else np.zeros(3)
            tr = _Track(next_id, seed, t, [(fi, pi)])
            persons[pi]["track_id"] = next_id
            next_id += 1
            alive.append(tr)
        active = alive
    return next_id


def _drop_short_tracks(frames: list[dict], min_frames: int = 4) -> None:
    """闪现 1～3 帧的幽灵骨架丢掉。"""
    tracks = _tracks_from_ids(frames)
    drop = {tid for tid, mem in tracks.items() if len(mem) < min_frames}
    if not drop:
        return
    for fr in frames:
        fr["persons"] = [
            p for p in (fr.get("persons") or []) if int(p.get("track_id") or -1) not in drop
        ]


def _reject_speed(values: np.ndarray, mask: np.ndarray, times: np.ndarray, vmax: float) -> None:
    last = -1
    for i in range(len(mask)):
        if not mask[i]:
            continue
        if last >= 0:
            dt = float(times[i] - times[last])
            if dt > 1e-6:
                spd = float(np.linalg.norm(values[i] - values[last]) / dt)
                if spd > vmax:
                    mask[i] = False
                    continue
        last = i


def _gauss_smooth(
    values: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    sigma_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """非均匀时间的归一化高斯。缺测处若邻域够近则补上。"""
    n, dim = values.shape
    out = np.full_like(values, np.nan)
    out_mask = np.zeros(n, dtype=bool)
    if n == 0 or sigma_s <= 0:
        return values.copy(), mask.copy()
    radius = 3.0 * sigma_s
    j0 = 0
    for i in range(n):
        while j0 < n and times[i] - times[j0] > radius:
            j0 += 1
        j1 = i
        while j1 < n and times[j1] - times[i] <= radius:
            j1 += 1
        wsum = 0.0
        acc = np.zeros(dim)
        for j in range(j0, j1):
            if not mask[j]:
                continue
            dlt = float(times[j] - times[i])
            w = float(np.exp(-0.5 * (dlt / sigma_s) ** 2))
            wsum += w
            acc += w * values[j]
        # 中心权约为 1；邻域太稀则不算有效
        if wsum < 0.45:
            continue
        out[i] = acc / wsum
        out_mask[i] = True
    return out, out_mask


def _median_len(a: np.ndarray, b: np.ndarray, ma: np.ndarray, mb: np.ndarray, lo: float, hi: float) -> float | None:
    lens = []
    for i in range(len(ma)):
        if not (ma[i] and mb[i]):
            continue
        d = float(np.linalg.norm(a[i] - b[i]))
        if lo <= d <= hi:
            lens.append(d)
    if len(lens) < BONE_MIN_SAMPLES:
        return None
    return float(np.median(lens))


def _clamp_bone(prox: np.ndarray, dist: np.ndarray, mp: np.ndarray, md: np.ndarray, L: float) -> None:
    for i in range(len(mp)):
        if not (mp[i] and md[i]) or L <= 1e-6:
            continue
        v = dist[i] - prox[i]
        d = float(np.linalg.norm(v))
        if d < 1e-6:
            continue
        if abs(d - L) / L <= BONE_TOL:
            continue
        dist[i] = prox[i] + v * (L / d)


def _series_from_track(frames: list[dict], members: list[tuple[int, int]], j: int):
    times, vals, mask = [], [], []
    for fi, pi in members:
        persons = frames[fi].get("persons") or []
        if pi >= len(persons):
            continue
        xyz = persons[pi].get("xyz") or []
        t = float(frames[fi].get("t") or 0.0)
        p = _as3(xyz[j]) if j < len(xyz) else None
        times.append(t)
        if p is None:
            vals.append(np.zeros(3))
            mask.append(False)
        else:
            vals.append(p)
            mask.append(True)
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(vals, dtype=np.float64).reshape(-1, 3),
        np.asarray(mask, dtype=bool),
    )


def _write_joint(frames: list[dict], members: list[tuple[int, int]], j: int, values: np.ndarray, mask: np.ndarray) -> None:
    k = 0
    for fi, pi in members:
        persons = frames[fi].get("persons") or []
        if pi >= len(persons):
            continue
        person = persons[pi]
        xyz = person.setdefault("xyz", [None] * N_JOINTS)
        vis = person.setdefault("vis", [0] * N_JOINTS)
        while len(xyz) < N_JOINTS:
            xyz.append(None)
        while len(vis) < N_JOINTS:
            vis.append(0)
        if mask[k]:
            xyz[j] = [round(float(v), 3) for v in values[k]]
            vis[j] = 1
        else:
            xyz[j] = None
            vis[j] = 0
        k += 1


def _tracks_from_ids(frames: list[dict]) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for fi, fr in enumerate(frames):
        for pi, p in enumerate(fr.get("persons") or []):
            tid = p.get("track_id")
            if tid is None:
                continue
            out.setdefault(int(tid), []).append((fi, pi))
    return out


def smooth_frames(frames: list[dict], plane: dict | None = None) -> dict:
    """就地平滑 frames[*].persons[*].xyz，并按平滑后的腕点重算 dL/dR。"""
    already = any(
        p.get("track_id") is not None for fr in frames for p in (fr.get("persons") or [])
    )
    if not already:
        assign_tracks(frames)
    _drop_short_tracks(frames, min_frames=4)
    n_tracks = len(_tracks_from_ids(frames))
    tracks = _tracks_from_ids(frames)
    n_wrist_in = n_wrist_out = 0
    for tid, members in tracks.items():
        if len(members) < 2:
            continue
        series: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for j in range(N_JOINTS):
            times, vals, mask = _series_from_track(frames, members, j)
            if j in (LWRIST, RWRIST):
                n_wrist_in += int(mask.sum())
            _reject_speed(vals, mask, times, joint_vmax(j))
            sm, sm_mask = _gauss_smooth(vals, mask, times, joint_sigma(j))
            series[j] = (sm, sm_mask, times)
        # 骨长：先上臂再前臂，腕跟着已经稳住的肘走
        for a, b in ARM_BONES:
            sa, ma, _ = series[a]
            sb, mb, _ = series[b]
            lo, hi = BONE_LEN_RANGE[(a, b)]
            L = _median_len(sa, sb, ma, mb, lo, hi)
            if L is None:
                continue
            _clamp_bone(sa, sb, ma, mb, L)
        for j in range(N_JOINTS):
            sm, sm_mask, _ = series[j]
            if j in (LWRIST, RWRIST):
                n_wrist_out += int(sm_mask.sum())
            _write_joint(frames, members, j, sm, sm_mask)

    if plane is not None:
        for fr in frames:
            for p in fr.get("persons") or []:
                xyz = p.get("xyz") or []
                lw, rw = _as3(xyz[LWRIST] if LWRIST < len(xyz) else None), _as3(
                    xyz[RWRIST] if RWRIST < len(xyz) else None
                )
                p["dL"] = round(signed_x(lw, plane), 3) if lw is not None else None
                p["dR"] = round(signed_x(rw, plane), 3) if rw is not None else None

    for fr in frames:
        persons = fr.get("persons") or []
        persons.sort(key=lambda p: int(p.get("track_id") or 0))
        fr["persons"] = persons

    return {
        "n_tracks": n_tracks,
        "n_wrist_in": n_wrist_in,
        "n_wrist_out": n_wrist_out,
        "sigma_wrist_s": SIGMA_WRIST,
        "sigma_elbow_s": SIGMA_ELBOW,
        "sigma_body_s": SIGMA_BODY,
        "vmax_wrist": VMAX_WRIST,
        "bone_clamp": True,
    }


def wrist_jump_stats(frames: list[dict], joint: int = LWRIST) -> dict[str, float]:
    """按 track 统计腕点帧间位移，用来对照平滑前后。"""
    tracks = _tracks_from_ids(frames)
    if not tracks:
        tracks = {
            0: [
                (fi, 0)
                for fi, fr in enumerate(frames)
                if (fr.get("persons") or [])
            ]
        }
    jumps: list[float] = []
    for members in tracks.values():
        prev = None
        for fi, pi in members:
            persons = frames[fi].get("persons") or []
            if pi >= len(persons):
                prev = None
                continue
            xyz = persons[pi].get("xyz") or []
            p = _as3(xyz[joint] if joint < len(xyz) else None)
            if p is None:
                prev = None
                continue
            if prev is not None:
                jumps.append(float(np.linalg.norm(p - prev)))
            prev = p
    if not jumps:
        return {"n": 0}
    a = np.asarray(jumps)
    return {
        "n": int(a.size),
        "p50": round(float(np.percentile(a, 50)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
        "p99": round(float(np.percentile(a, 99)), 4),
        "frac_gt_0.15": round(float(np.mean(a > 0.15)), 4),
    }
