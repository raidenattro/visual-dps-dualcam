"""骨架时序平滑（对齐 visual-dps-pick-state/scripts/skel3d_smooth.py）。

先滤左右路 2D（短窗），再三角化；3D 滤波 + 飞骨裁到合法区间。分数不平滑。

pick-state dump 按 25fps（DESIGN_DT=40ms）写 σ：高斯窗是「几十毫秒的检测噪声」，
不是「几帧姿势」。本仓直播相机 15fps × interval=2 ≈ 7.5Hz，同一套时间 σ 几乎
看不见上一帧；若按帧数加宽，两帧真动作会被插成橡皮泥。

因此因果路径按 pose 周期分流：
- 密（dt≤DENSE_DT，约 ≥15Hz）：复现 dump 的逐关节因果高斯 + 骨长中位；
- 稀（约 7.5Hz）：只滑躯干质心，形状贴本帧，再用速度门压慢抖（快动作不掺）。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from dualcam.lift import KPT_MIN, LWRIST, RWRIST, _torso_xy

# COCO-17
LSHO, RSHO, LELB, RELB = 5, 6, 7, 8
LHIP, RHIP, LKNE, RKNE, LANK, RANK = 11, 12, 13, 14, 15, 16
N_JOINTS = 17
TORSO = (LSHO, RSHO, LHIP, RHIP)
# 近端 → 远端，平滑之后把远端收到骨长中位上，避免腕/踝飞离
ARM_BONES = ((LSHO, LELB), (LELB, LWRIST), (RSHO, RELB), (RELB, RWRIST))
LEG_BONES = ((LHIP, LKNE), (LKNE, LANK), (RHIP, RKNE), (RKNE, RANK))
LIMB_BONES = ARM_BONES + LEG_BONES
# 近端→远端，用生抬点骨长沿平滑方向放回，避免腕比肩滞后把骨头拉成橡皮泥
TORSO_BONES = ((LSHO, RSHO), (LHIP, RHIP), (LSHO, LHIP), (RSHO, RHIP))
RIGID_BONES = TORSO_BONES + LIMB_BONES
BONE_LEN_RANGE = {
    (LSHO, LELB): (0.18, 0.42),
    (RSHO, RELB): (0.18, 0.42),
    (LELB, LWRIST): (0.15, 0.38),
    (RELB, RWRIST): (0.15, 0.38),
    (LHIP, LKNE): (0.28, 0.55),
    (RHIP, RKNE): (0.28, 0.55),
    (LKNE, LANK): (0.28, 0.52),
    (RKNE, RANK): (0.28, 0.52),
}

# 秒。2D 先滤短窗；dump 25fps 的 3D 只收残差。直播质心窗按周期放大，关节形状不跨帧混。
SIGMA_2D_BODY = 0.045
SIGMA_2D_ELBOW = 0.055
SIGMA_2D_WRIST = 0.070
SIGMA_BODY = 0.030
SIGMA_ELBOW = 0.045
SIGMA_WRIST = 0.055
VMAX_2D_PX = 1200.0
VMAX_BODY = 3.0
VMAX_WRIST = 4.0
TRACK_2D_PX = 140.0
TRACK_MAX_M = 0.60
TRACK_MAX_GAP_S = 0.48
BONE_TOL = 0.12  # 相对中位，超出才拉回
BONE_MIN_SAMPLES = 16
BONE_MIN_SAMPLES_LIVE = 4
# dump 设计周期 40ms。σ 是时间常数，不要按直播帧数加宽到「几帧姿势」。
DESIGN_DT = 0.04
LIVE_KEEP_S = 2.4
LIVE_3D_SPAN_FRAMES = 2.5
EVENT_SKELETON_STALE_S = 0.40
# dump hold 8 帧 @25fps = 0.32s；直播按墙钟对齐，7.5Hz 只能续约 2 帧。
HOLD_SEC = 8 * DESIGN_DT
# 因果 3D：周期接近 dump 走逐关节短窗；更稀则质心 + 速度门。
DENSE_DT = 0.075
# 超过则认定是真动作，不和上一帧姿势掺（m/s、px/s）
FOLLOW_SPEED_3D = 0.85
EMA_ALPHA_SLOW = 0.42
# 因果 2D：3σ 最多刚盖住约 2 个 pose，不再 2× 加宽糊 400ms
LIVE_2D_SPAN_FRAMES = 2.0
# 1-Euro（像素）：静止压抖，伸手跟手。beta 单位 1/(px/s)
EURO_MINCUTOFF = 1.05
EURO_BETA = 0.012
EURO_DCUTOFF = 1.0


def signed_x(p: np.ndarray, plane: dict) -> float:
    """巷道内侧为正：人在通道里 >0，伸进墙为 ≤0。"""
    return float((np.asarray(p, dtype=np.float64).reshape(-1)[:3] - np.asarray(plane["p0"], dtype=np.float64)) @ np.asarray(plane["n"], dtype=np.float64))


def median_dt(times) -> float:
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if t.size < 2:
        return DESIGN_DT
    d = np.diff(np.sort(t))
    d = d[d > 1e-6]
    if d.size == 0:
        return DESIGN_DT
    return float(np.median(d))


def adapt_sigma(sigma_s: float, dt: float | None, cap: float | None = 2.0) -> float:
    if not dt or dt <= 0:
        return float(sigma_s)
    scale = max(1.0, float(dt) / DESIGN_DT)
    if cap is not None:
        scale = min(float(cap), scale)
    return float(sigma_s) * scale


def live_adapt_sigma(sigma_s: float, dt: float | None) -> float:
    """质心窗：按 pose 周期放大，且 3σ 至少盖住 LIVE_3D_SPAN_FRAMES 帧。不对各关节用。"""
    base = adapt_sigma(sigma_s, dt, cap=None)
    if not dt or dt <= 0:
        return base
    need = LIVE_3D_SPAN_FRAMES * float(dt) / 3.0
    return float(max(base, need))


def _smoothing_factor(dt: float, cutoff: float) -> float:
    r = 2.0 * math.pi * float(cutoff) * float(dt)
    return r / (r + 1.0)


def _one_euro_series(
    values: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    *,
    mincutoff: float = EURO_MINCUTOFF,
    beta: float = EURO_BETA,
    dcutoff: float = EURO_DCUTOFF,
) -> None:
    """就地 1-Euro（One Euro Filter）：慢动截止低、快动截止高，不会像速度门那样回弹。"""
    hatx: np.ndarray | None = None
    hatdx: np.ndarray | None = None
    last_t: float | None = None
    for i in range(len(mask)):
        if not mask[i]:
            hatx = None
            hatdx = None
            last_t = None
            continue
        x = np.asarray(values[i], dtype=np.float64)
        t = float(times[i])
        if hatx is None or hatdx is None or last_t is None:
            hatx = x.copy()
            hatdx = np.zeros_like(x)
            last_t = t
            values[i] = hatx
            continue
        dt = t - last_t
        if dt <= 1e-6:
            values[i] = hatx
            continue
        dx = (x - hatx) / dt
        ad = _smoothing_factor(dt, dcutoff)
        hatdx = ad * dx + (1.0 - ad) * hatdx
        cutoff = mincutoff + beta * float(np.linalg.norm(hatdx))
        a = _smoothing_factor(dt, cutoff)
        hatx = a * x + (1.0 - a) * hatx
        last_t = t
        values[i] = hatx


def _speed_gated_blend(
    prev: np.ndarray,
    cur: np.ndarray,
    dt: float,
    follow_speed: float,
    alpha_slow: float = EMA_ALPHA_SLOW,
) -> np.ndarray:
    """慢抖掺上一帧；位移速度过快则跟本帧，避免两帧姿势插成橡皮泥。"""
    if dt <= 1e-6:
        return np.asarray(cur, dtype=np.float64)
    prev = np.asarray(prev, dtype=np.float64)
    cur = np.asarray(cur, dtype=np.float64)
    spd = float(np.linalg.norm(cur - prev) / dt)
    if spd >= follow_speed:
        return cur
    a = float(alpha_slow)
    return a * cur + (1.0 - a) * prev


def _apply_speed_gated_series(
    series: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    follow_speed: float,
    alpha_slow: float = EMA_ALPHA_SLOW,
) -> None:
    """就地按关节做速度门。"""
    for sm, mk, times in series.values():
        last = -1
        for i in range(len(mk)):
            if not mk[i]:
                continue
            if last >= 0:
                dt = float(times[i] - times[last])
                sm[i] = _speed_gated_blend(sm[last], sm[i], dt, follow_speed, alpha_slow)
            last = i


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


# 肘几乎贴在肩-腕轴上时才用上一帧铰链侧，避免三角化把臂折成一根棍。
_ELBOW_AXIS_M = 0.025
_ARM_CHAINS = ((LSHO, LELB, LWRIST), (RSHO, RELB, RWRIST))
_STRETCH_EPS = 0.08


def _two_bone_place(
    shoulder: np.ndarray,
    wrist: np.ndarray,
    lu: float,
    ll: float,
    elbow_hint: np.ndarray | None,
    prefer: np.ndarray | None,
) -> np.ndarray:
    """肩、腕固定，肘落到两球面交线；prefer / hint 只用来选铰链哪一侧，不按胸口强翻。"""
    sw = wrist - shoulder
    d = float(np.linalg.norm(sw))
    lo = abs(lu - ll) + 1e-3
    hi = lu + ll - 1e-3
    if hi <= lo:
        return shoulder + 0.5 * sw
    if d < 1e-6:
        return shoulder + (prefer if prefer is not None else np.array([0.0, 1.0, 0.0])) * lu
    if d < lo or d > hi:
        d2 = float(np.clip(d, lo, hi))
        wrist = shoulder + sw * (d2 / d)
        sw = wrist - shoulder
        d = d2
    a = (lu * lu - ll * ll + d * d) / (2.0 * d)
    h2 = lu * lu - a * a
    axis = sw / d
    mid = shoulder + a * axis
    if h2 <= 1e-8:
        return mid
    h = float(np.sqrt(max(0.0, h2)))

    def _in_plane(v: np.ndarray) -> np.ndarray:
        return v - axis * float(np.dot(v, axis))

    side = None
    if elbow_hint is not None:
        side = _in_plane(elbow_hint - mid)
    if (side is None or float(np.linalg.norm(side)) < 1e-6) and prefer is not None:
        side = _in_plane(prefer)
    sn = float(np.linalg.norm(side)) if side is not None else 0.0
    if sn < 1e-6:
        side = np.cross(axis, np.array([0.0, 1.0, 0.0]))
        sn = float(np.linalg.norm(side))
        if sn < 1e-6:
            side = np.cross(axis, np.array([1.0, 0.0, 0.0]))
            sn = float(np.linalg.norm(side))
    return mid + side * (h / (sn + 1e-12))


def repair_arm_chains(
    xyz: list,
    srcs: list | None = None,
    prev_xyz: list | None = None,
) -> None:
    """臂链铰链投影。不要接到直播抬点上：用教科书骨长把肘钉到圆上，会把立体尺度拉成哈哈镜。

    仅保留给单测/离线实验。直播只靠三角化 + 飞腕上限 + 出界骨长裁切。
    """
    if not xyz or len(xyz) < 17:
        return
    for sho_i, elb_i, wri_i in _ARM_CHAINS:
        s = _as3(xyz[sho_i])
        e = _as3(xyz[elb_i])
        w = _as3(xyz[wri_i])
        if s is None or w is None:
            continue
        prev_e = _as3(prev_xyz[elb_i]) if prev_xyz and elb_i < len(prev_xyz) else None
        lu_lo, lu_hi = BONE_LEN_RANGE[(sho_i, elb_i)]
        ll_lo, ll_hi = BONE_LEN_RANGE[(elb_i, wri_i)]
        lu = float(np.clip(np.linalg.norm(e - s), lu_lo, lu_hi)) if e is not None else 0.5 * (lu_lo + lu_hi)
        ll = (
            float(np.clip(np.linalg.norm(w - e), ll_lo, ll_hi))
            if e is not None
            else 0.5 * (ll_lo + ll_hi)
        )
        dsw = float(np.linalg.norm(w - s))
        max_r = lu + ll
        if dsw > max_r + _STRETCH_EPS and dsw > 1e-6:
            w = s + (w - s) * (max_r / dsw)
            xyz[wri_i] = w.tolist()

        sw = w - s
        d = float(np.linalg.norm(sw))
        if d < 1e-6:
            continue
        axis = sw / d
        a = (lu * lu - ll * ll + d * d) / (2.0 * d)
        mid = s + a * axis
        hint = e
        if e is not None:
            perp = (e - mid) - axis * float(np.dot(e - mid, axis))
            if float(np.linalg.norm(perp)) < _ELBOW_AXIS_M and prev_e is not None:
                hint = prev_e
        prefer = None
        if prev_e is not None and (e is None or float(np.linalg.norm(
            (e - mid) - axis * float(np.dot(e - mid, axis))
        )) < _ELBOW_AXIS_M):
            prefer = prev_e - mid
        e2 = _two_bone_place(s, w, lu, ll, hint, prefer)
        xyz[elb_i] = e2.tolist()


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


def joint_sigma_2d(j: int) -> float:
    if j in (LWRIST, RWRIST):
        return SIGMA_2D_WRIST
    if j in (LELB, RELB):
        return SIGMA_2D_ELBOW
    return SIGMA_2D_BODY


def copy_pose_pack(pack) -> list[dict]:
    """深拷贝左右路 k/s，避免改到 npz 原数组。"""
    out: list[dict] = []
    for fr in pack:
        rec: dict = {"i": fr["i"], "t": fr["t"]}
        for view in ("L", "R"):
            v = fr[view]
            rec[view] = {
                "k": [np.array(p, dtype=np.float64, copy=True) for p in v["k"]],
                "s": [np.array(sc, dtype=np.float64, copy=True) for sc in v["s"]],
            }
        out.append(rec)
    return out


def assign_tracks_2d(
    pack: list[dict],
    view: str,
    *,
    max_px: float = TRACK_2D_PX,
    max_gap_s: float = TRACK_MAX_GAP_S,
) -> dict[int, list[tuple[int, int]]]:
    """单路按躯干 2D 续 track，返回 tid → [(frame_i, det_i)]。"""
    active: list[_Track] = []
    next_id = 0
    tracks: dict[int, list[tuple[int, int]]] = {}
    for fi, fr in enumerate(pack):
        t = float(fr.get("t") or 0.0)
        klist, slist = fr[view]["k"], fr[view]["s"]
        xys = [_torso_xy(klist[i], slist[i]) for i in range(len(klist))]
        alive = [tr for tr in active if (t - tr.last_t) <= max_gap_s]
        cands: list[tuple[float, int, int]] = []
        for di, xy in enumerate(xys):
            if xy is None:
                continue
            for ti, tr in enumerate(alive):
                cands.append((float(np.linalg.norm(xy - tr.last_torso)), di, ti))
        cands.sort()
        used_d: set[int] = set()
        used_t: set[int] = set()
        for dist, di, ti in cands:
            if di in used_d or ti in used_t or dist > max_px:
                continue
            tr = alive[ti]
            tr.members.append((fi, di))
            tr.last_torso = xys[di]  # type: ignore[assignment]
            tr.last_t = t
            tracks[tr.tid].append((fi, di))
            used_d.add(di)
            used_t.add(ti)
        for di, xy in enumerate(xys):
            if di in used_d or xy is None:
                continue
            tr = _Track(next_id, xy, t, [(fi, di)])
            tracks[next_id] = [(fi, di)]
            next_id += 1
            alive.append(tr)
        active = alive
    return tracks


def smooth_pose2d(pack: list[dict], views: tuple[str, ...] = ("L", "R"), *, causal: bool = False) -> dict:
    """就地平滑 pack[*][view].k 的像素坐标，分数不动。"""
    n_tracks = 0
    n_in = n_out = 0
    dt = median_dt([float(fr.get("t") or 0.0) for fr in pack]) if causal else DESIGN_DT
    for view in views:
        tracks = assign_tracks_2d(pack, view)
        n_tracks += len(tracks)
        for members in tracks.values():
            if len(members) < 2:
                continue
            times = np.array([float(pack[fi]["t"]) for fi, _ in members], dtype=np.float64)
            for j in range(N_JOINTS):
                vals = np.zeros((len(members), 2), dtype=np.float64)
                mask = np.zeros(len(members), dtype=bool)
                for i, (fi, di) in enumerate(members):
                    k = pack[fi][view]["k"][di]
                    s = pack[fi][view]["s"][di]
                    if float(s[j]) < KPT_MIN:
                        continue
                    uv = np.asarray(k[j], dtype=np.float64).reshape(-1)
                    if uv.size < 2 or not np.all(np.isfinite(uv[:2])):
                        continue
                    vals[i] = uv[:2]
                    mask[i] = True
                if j in (LWRIST, RWRIST):
                    n_in += int(mask.sum())
                _reject_speed(vals, mask, times, VMAX_2D_PX)
                # dump 用设计 σ。直播只把窗扩到「刚看见上一两帧」，避免 2× 糊整段动作。
                if causal and dt > DENSE_DT:
                    need = LIVE_2D_SPAN_FRAMES * float(dt) / 3.0
                    sig2d = max(joint_sigma_2d(j), need)
                else:
                    sig2d = joint_sigma_2d(j)
                sm, sm_mask = _gauss_smooth(
                    vals, mask, times, sig2d, causal=causal,
                )
                if causal and dt > DENSE_DT and j in (LWRIST, RWRIST, LELB, RELB):
                    _one_euro_series(sm, sm_mask, times)
                if j in (LWRIST, RWRIST):
                    n_out += int(sm_mask.sum())
                for i, (fi, di) in enumerate(members):
                    if not sm_mask[i]:
                        continue
                    arr = np.array(pack[fi][view]["k"][di][j], dtype=np.float64, copy=True)
                    arr[0] = sm[i, 0]
                    arr[1] = sm[i, 1]
                    pack[fi][view]["k"][di][j] = arr
    return {"n_tracks": n_tracks, "n_wrist_in": n_in, "n_wrist_out": n_out}


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
    *,
    causal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """非均匀时间的归一化高斯。缺测处若邻域够近则补上。causal 只看过去+当前。"""
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
        j1 = i + 1 if causal else i
        if not causal:
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
            # 邻域太稀时保留当前观测，避免关节被写成空再闪回来
            if mask[i]:
                out[i] = values[i]
                out_mask[i] = True
            continue
        out[i] = acc / wsum
        out_mask[i] = True
    return out, out_mask


def _median_len(
    a: np.ndarray, b: np.ndarray, ma: np.ndarray, mb: np.ndarray, lo: float, hi: float,
    min_samples: int | None = None,
) -> float | None:
    lens = []
    for i in range(len(ma)):
        if not (ma[i] and mb[i]):
            continue
        d = float(np.linalg.norm(a[i] - b[i]))
        if lo <= d <= hi:
            lens.append(d)
    if len(lens) < (BONE_MIN_SAMPLES if min_samples is None else min_samples):
        return None
    return float(np.median(lens))


def _clamp_bones_causal(
    prox: np.ndarray,
    dist: np.ndarray,
    mp: np.ndarray,
    md: np.ndarray,
    lo: float,
    hi: float,
    *,
    min_samples: int | None = None,
    use_median: bool = True,
) -> None:
    """飞骨收到 [lo, hi]。dump 样本够了才按此人中位微调；直播 7.5Hz 不要用中位，
    否则几帧不同姿势会被收成同一根骨头（橡皮泥）。"""
    need = BONE_MIN_SAMPLES if min_samples is None else int(min_samples)
    hist: list[float] = []
    L_run: float | None = None
    for i in range(len(mp)):
        if not (mp[i] and md[i]):
            continue
        v = dist[i] - prox[i]
        d = float(np.linalg.norm(v))
        if d < 1e-6:
            continue
        if lo <= d <= hi:
            hist.append(d)
            if use_median and len(hist) >= need:
                L_run = float(np.median(hist))
        if d > hi:
            dist[i] = prox[i] + v * (hi / d)
            continue
        if d < lo:
            dist[i] = prox[i] + v * (lo / d)
            continue
        if not use_median or L_run is None or L_run <= 1e-6:
            continue
        if abs(d - L_run) / L_run <= BONE_TOL:
            continue
        dist[i] = prox[i] + v * (L_run / d)


def _centroid_from_raw(raw: dict[int, tuple[np.ndarray, np.ndarray]], i: int) -> np.ndarray | None:
    """第 i 帧躯干质心；躯干不够则用所有可见关节。"""
    pts = []
    for j in TORSO:
        if j not in raw:
            continue
        vals, mask = raw[j]
        if i < len(mask) and mask[i]:
            pts.append(vals[i])
    if len(pts) < 2:
        pts = []
        for vals, mask in raw.values():
            if i < len(mask) and mask[i]:
                pts.append(vals[i])
    if not pts:
        return None
    return np.mean(pts, axis=0)


def _apply_rigid_com_smooth(
    series: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    raw: dict[int, tuple[np.ndarray, np.ndarray]],
    times: np.ndarray,
    sigma_s: float,
    *,
    causal: bool,
) -> None:
    """只平滑质心，再把本帧相对质心的偏移贴回去。各关节不再独立插值。"""
    n = int(len(times))
    if n == 0 or sigma_s <= 0:
        return
    cents = np.zeros((n, 3), dtype=np.float64)
    cm = np.zeros(n, dtype=bool)
    for i in range(n):
        c = _centroid_from_raw(raw, i)
        if c is None:
            continue
        cents[i] = c
        cm[i] = True
    _reject_speed(cents, cm, times, VMAX_BODY)
    sm, sm_mask = _gauss_smooth(cents, cm, times, sigma_s, causal=causal)
    for i in range(n):
        if not (cm[i] and sm_mask[i]):
            continue
        delta = sm[i] - cents[i]
        for j, (sv, smk, _) in series.items():
            if j not in raw:
                continue
            rv, rm = raw[j]
            if not rm[i]:
                continue
            sv[i] = rv[i] + delta
            smk[i] = True


def _restore_raw_bone_lengths(
    smoothed: dict[int, tuple[np.ndarray, np.ndarray]],
    raw: dict[int, tuple[np.ndarray, np.ndarray]],
    bones: tuple[tuple[int, int], ...] = RIGID_BONES,
) -> None:
    """近端已平滑后，远端沿平滑方向收到本帧生抬点的骨长。"""
    n = 0
    for sm, mk in smoothed.values():
        n = len(sm)
        break
    if n == 0:
        return
    for i in range(n):
        for a, b in bones:
            if a not in smoothed or b not in smoothed or a not in raw or b not in raw:
                continue
            sa, ma = smoothed[a]
            sb, mb = smoothed[b]
            ra, rma = raw[a]
            rb, rmb = raw[b]
            if not (ma[i] and mb[i] and rma[i] and rmb[i]):
                continue
            L = float(np.linalg.norm(rb[i] - ra[i]))
            if L < 1e-4:
                continue
            v = sb[i] - sa[i]
            d = float(np.linalg.norm(v))
            if d < 1e-6:
                continue
            sb[i] = sa[i] + v * (L / d)


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


def smooth_frames(
    frames: list[dict],
    plane: dict | None = None,
    *,
    causal: bool = False,
    drop_short: bool = True,
) -> dict:
    """就地平滑 frames[*].persons[*].xyz，并按平滑后的腕点重算 dL/dR。"""
    already = any(
        p.get("track_id") is not None for fr in frames for p in (fr.get("persons") or [])
    )
    if not already:
        assign_tracks(frames)
    if drop_short:
        _drop_short_tracks(frames, min_frames=4)
    n_tracks = len(_tracks_from_ids(frames))
    tracks = _tracks_from_ids(frames)
    n_wrist_in = n_wrist_out = 0
    dt = median_dt([float(fr.get("t") or 0.0) for fr in frames]) if causal else DESIGN_DT
    bone_min = BONE_MIN_SAMPLES if (not causal or dt <= DENSE_DT) else BONE_MIN_SAMPLES_LIVE
    for tid, members in tracks.items():
        if len(members) < 2:
            continue
        series: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        raw: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        times_ref: np.ndarray | None = None
        for j in range(N_JOINTS):
            times, vals, mask = _series_from_track(frames, members, j)
            times_ref = times
            if j in (LWRIST, RWRIST):
                n_wrist_in += int(mask.sum())
            _reject_speed(vals, mask, times, joint_vmax(j))
            raw[j] = (vals.copy(), mask.copy())
            if causal and dt > DENSE_DT:
                # 稀采样：先留本帧形状，后面只滑质心 + 速度门
                series[j] = (vals.copy(), mask.copy(), times)
                continue
            sm, sm_mask = _gauss_smooth(
                vals, mask, times, joint_sigma(j), causal=causal,
            )
            series[j] = (sm, sm_mask, times)
        if causal and dt > DENSE_DT and times_ref is not None:
            # 7.5Hz：只滑质心，再对慢抖做速度门。加宽逐关节高斯会插成橡皮泥。
            _apply_rigid_com_smooth(
                series, raw, times_ref, live_adapt_sigma(SIGMA_BODY, dt), causal=True,
            )
            _apply_speed_gated_series(series, FOLLOW_SPEED_3D)
        elif not causal:
            _restore_raw_bone_lengths(
                {j: (series[j][0], series[j][1]) for j in series},
                raw,
            )
        use_median = (not causal) or (causal and dt <= DENSE_DT)
        for a, b in LIMB_BONES:
            sa, ma, _ = series[a]
            sb, mb, _ = series[b]
            lo, hi = BONE_LEN_RANGE[(a, b)]
            _clamp_bones_causal(
                sa, sb, ma, mb, lo, hi, min_samples=bone_min, use_median=use_median,
            )
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


def _persons_to_ks(persons: list | None) -> tuple[list, list]:
    ks: list[np.ndarray] = []
    ss: list[np.ndarray] = []
    for p in persons or []:
        xy = np.zeros((N_JOINTS, 2), dtype=np.float64)
        sc = np.zeros(N_JOINTS, dtype=np.float64)
        for j, kp in enumerate((p.get("keypoints") or [])[:N_JOINTS]):
            if not isinstance(kp, (list, tuple)) or len(kp) < 2:
                continue
            xy[j, 0] = float(kp[0])
            xy[j, 1] = float(kp[1])
            sc[j] = float(kp[2]) if len(kp) > 2 else 0.0
        ks.append(xy)
        ss.append(sc)
    return ks, ss


def _ks_to_persons(persons: list | None, ks: list, ss: list) -> list[dict]:
    out: list[dict] = []
    src = list(persons or [])
    for i, p in enumerate(src):
        q = dict(p)
        old = list(p.get("keypoints") or [])
        kpts = []
        for j in range(N_JOINTS):
            score = float(old[j][2]) if j < len(old) and isinstance(old[j], (list, tuple)) and len(old[j]) > 2 else 0.0
            if i < len(ss):
                score = float(ss[i][j]) if score <= 0 else score
            x = float(ks[i][j, 0]) if i < len(ks) else 0.0
            y = float(ks[i][j, 1]) if i < len(ks) else 0.0
            kpts.append([x, y, score])
        q["keypoints"] = kpts
        out.append(q)
    return out


def pose_time(pose: dict | None, frame_idx: int = 0) -> float:
    if isinstance(pose, dict):
        t = float(pose.get("ts") or pose.get("captured_at") or 0.0)
        if t > 0:
            return t
    return float(frame_idx) * DESIGN_DT


class LivePose2DSmoother:
    """单路 2D 因果短窗。输入/输出都是推理像素的 persons（keypoints）。"""

    def __init__(self, keep_s: float = LIVE_KEEP_S):
        self.keep_s = float(keep_s)
        self._frames: list[dict] = []

    def reset(self) -> None:
        self._frames = []

    def update(self, t: float, persons: list | None) -> list[dict]:
        src = list(persons or [])
        k, s = _persons_to_ks(src)
        self._frames.append({"t": float(t), "k": k, "s": s, "persons": [dict(p) for p in src]})
        cutoff = float(t) - self.keep_s
        self._frames = [fr for fr in self._frames if float(fr["t"]) >= cutoff]
        if len(self._frames) < 2:
            return src
        pack = [
            {
                "t": fr["t"],
                "L": {
                    "k": [np.array(p, dtype=np.float64, copy=True) for p in fr["k"]],
                    "s": [np.array(sc, dtype=np.float64, copy=True) for sc in fr["s"]],
                },
            }
            for fr in self._frames
        ]
        smooth_pose2d(pack, views=("L",), causal=True)
        last = pack[-1]["L"]
        return _ks_to_persons(src, last["k"], last["s"])


class LivePose3DSmoother:
    """3D 因果滤波 + 骨长。密采样复现 dump；稀采样只滑质心再加速度门。"""

    def __init__(self, keep_s: float = LIVE_KEEP_S):
        self.keep_s = float(keep_s)
        self._frames: list[dict] = []

    def reset(self) -> None:
        self._frames = []

    def update(self, t: float, persons: list | None, plane: dict | None = None) -> list[dict]:
        src = list(persons or [])
        self._frames.append({"t": float(t), "persons": copy.deepcopy(src)})
        cutoff = float(t) - self.keep_s
        self._frames = [fr for fr in self._frames if float(fr["t"]) >= cutoff]
        if len(self._frames) < 2:
            return src
        work = copy.deepcopy(self._frames)
        # 直播不能 drop_short：hold 的人在窗里只有几帧时会被整段抹掉。
        smooth_frames(work, plane, causal=True, drop_short=False)
        return list(work[-1].get("persons") or [])

