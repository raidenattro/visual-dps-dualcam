"""两路骨架三角化：跨路匹配人 + 逐关节抬 3D。

相机 dict 接受 solve_dual 的 bundle（f/cx/cy/C/right/down/fwd），
也接受旧 lift 脚本的 {f, c0, C, ...}。
"""

from __future__ import annotations

from typing import Any

import numpy as np

LWRIST, RWRIST = 9, 10
KPT_MIN = 0.3
CONF_MARGIN = 0.12
PAIR_GAP_MAX = 0.18
JOINT_GAP_MAX = 0.20
_PAIR_TORSO = (5, 6, 11, 12)
_PAIR_JOINTS = _PAIR_TORSO + (LWRIST, RWRIST)
PAIR_MIN_VIS = 10
PAIR_MIN_JOINTS = 3
NMS_TORSO_PX = 50.0
PREFER_PX = 90.0
DUPLICATE_M = 0.40
AISLE_AABB = {"x": (-1.35, 1.35), "y": (0.50, 1.65), "z": (-0.12, 2.50)}
# 可贴墙的源：立体或高分路主导。hold/mono 只显示不报警。
CONTACT_SRC = frozenset({"stereo", "L", "R"})


def normalize_cam(cam: dict[str, Any]) -> dict[str, Any]:
    """统一成 lift 内部格式。"""
    if cam.get("c0") is not None and not isinstance(cam.get("cx"), (int, float)):
        f = cam["f"]
        if isinstance(f, (list, tuple, np.ndarray)):
            f = float(np.asarray(f).reshape(-1)[0])
        return {
            "f": float(f),
            "c0": np.asarray(cam["c0"], float).reshape(2),
            "C": np.asarray(cam["C"], float).reshape(3),
            "right": np.asarray(cam["right"], float).reshape(3),
            "down": np.asarray(cam["down"], float).reshape(3),
            "fwd": np.asarray(cam["fwd"], float).reshape(3),
        }
    f = cam["f"]
    if isinstance(f, (list, tuple, np.ndarray)):
        f = float(np.asarray(f).reshape(-1)[0])
    return {
        "f": float(f),
        "c0": np.array([float(cam["cx"]), float(cam["cy"])], float),
        "C": np.asarray(cam["C"], float).reshape(3),
        "right": np.asarray(cam["right"], float).reshape(3),
        "down": np.asarray(cam["down"], float).reshape(3),
        "fwd": np.asarray(cam["fwd"], float).reshape(3),
    }


def normalize_cams(cams: dict[str, Any]) -> dict[str, Any]:
    return {k: normalize_cam(v) for k, v in cams.items()}


def wall_plane_from_solved(solved: dict, wall_id: int = 1) -> dict | None:
    for w in solved.get("walls") or []:
        if int(w.get("wall_id") or 0) == int(wall_id):
            p0 = np.array(w["corners"][0], float)
            n = np.array([1.0 if int(w.get("sign", -1)) < 0 else -1.0, 0.0, 0.0])
            return {"p0": p0, "n": n, "x": float(p0[0])}
    return None


def ray(uv: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    cam = normalize_cam(cam)
    uv = np.asarray(uv, float).reshape(2)
    x = (uv[0] - cam["c0"][0]) / cam["f"]
    y = (uv[1] - cam["c0"][1]) / cam["f"]
    d = x * cam["right"] + y * cam["down"] + cam["fwd"]
    n = np.linalg.norm(d)
    return cam["C"], d / max(float(n), 1e-12)


def triangulate_ends(
    uv_l: np.ndarray, uv_r: np.ndarray, cams: dict
) -> tuple[np.ndarray, np.ndarray, float]:
    c1, d1 = ray(uv_l, cams["L"])
    c2, d2 = ray(uv_r, cams["R"])
    w0 = c1 - c2
    a, b, c = float(d1 @ d1), float(d1 @ d2), float(d2 @ d2)
    d, e = float(d1 @ w0), float(d2 @ w0)
    den = a * c - b * b
    if abs(den) < 1e-9:
        return c1 + d1, c2 + d2, 99.0
    t = (b * e - c * d) / den
    s = (a * e - b * d) / den
    p1, p2 = c1 + t * d1, c2 + s * d2
    return p1, p2, float(np.linalg.norm(p1 - p2))


def triangulate(uv_l: np.ndarray, uv_r: np.ndarray, cams: dict) -> tuple[np.ndarray, float]:
    p1, p2, g = triangulate_ends(uv_l, uv_r, cams)
    return 0.5 * (p1 + p2), g


def point_on_ray(uv: np.ndarray, cam: dict, ref: np.ndarray) -> np.ndarray:
    C, d = ray(uv, cam)
    t = float((np.asarray(ref, float) - C) @ d)
    if t < 0.05:
        t = 0.05
    return C + t * d


def ray_plane(uv: np.ndarray, cam: dict, plane: dict) -> np.ndarray | None:
    c, d = ray(uv, cam)
    den = float(d @ plane["n"])
    if abs(den) < 1e-6:
        return None
    t = float((plane["p0"] - c) @ plane["n"]) / den
    if t < 0.05:
        return None
    return c + t * d


def lift_point(
    uv_l,
    s_l: float,
    uv_r,
    s_r: float,
    cams: dict,
    plane: dict | None,
    prev: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float | None, str | None]:
    sl, sr = float(s_l), float(s_r)
    ok_l, ok_r = sl >= KPT_MIN, sr >= KPT_MIN
    if ok_l and ok_r:
        p1, p2, g = triangulate_ends(uv_l, uv_r, cams)
        if g <= JOINT_GAP_MAX:
            p = (sl * p1 + sr * p2) / (sl + sr)
            if abs(sl - sr) < CONF_MARGIN:
                return p, g, "stereo"
            return p, g, "L" if sl >= sr else "R"
        winner_l = sl >= sr
        uv, cam = (uv_l, cams["L"]) if winner_l else (uv_r, cams["R"])
        src = "L" if winner_l else "R"
        ref = prev if prev is not None else (p1 if winner_l else p2)
        return point_on_ray(uv, cam, ref), g, src
    if ok_l:
        if prev is not None:
            return point_on_ray(uv_l, cams["L"], prev), None, "Lhold"
        hit = ray_plane(uv_l, cams["L"], plane) if plane is not None else None
        return hit, None, "Lmono" if hit is not None else None
    if ok_r:
        if prev is not None:
            return point_on_ray(uv_r, cams["R"], prev), None, "Rhold"
        hit = ray_plane(uv_r, cams["R"], plane) if plane is not None else None
        return hit, None, "Rmono" if hit is not None else None
    return None, None, None


def _torso_xy(k, s) -> np.ndarray | None:
    pts = [np.asarray(k[i][:2], float) for i in _PAIR_TORSO if s[i] >= KPT_MIN]
    if not pts:
        return None
    return np.mean(pts, axis=0)


def _nvis(s) -> int:
    return int(np.sum(np.asarray(s, float) >= KPT_MIN))


def nms_indices(k, s, dist_px: float = NMS_TORSO_PX) -> list[int]:
    n = len(k)
    xy = [_torso_xy(k[i], s[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -float(np.mean(s[i])))
    kept: list[int] = []
    for i in order:
        if _nvis(s[i]) < PAIR_MIN_VIS or xy[i] is None:
            continue
        if any(
            float(np.linalg.norm(xy[i] - xy[j])) < dist_px
            for j in kept
            if xy[j] is not None
        ):
            continue
        kept.append(i)
    return kept


def _pair_gap(kl, sl, kr, sr, cams: dict) -> float | None:
    gap = []
    n_torso = 0
    for k in _PAIR_JOINTS:
        if sl[k] < KPT_MIN or sr[k] < KPT_MIN:
            continue
        _p, g = triangulate(kl[k], kr[k], cams)
        if g > JOINT_GAP_MAX * 2:
            continue
        gap.append(g)
        if k in _PAIR_TORSO:
            n_torso += 1
    if n_torso < 1 or len(gap) < PAIR_MIN_JOINTS:
        return None
    return float(np.median(gap))


def _torso_xyz(kl, sl, kr, sr, cams: dict) -> np.ndarray | None:
    pts = []
    for i in _PAIR_TORSO:
        if sl[i] < KPT_MIN or sr[i] < KPT_MIN:
            continue
        p, g = triangulate(kl[i], kr[i], cams)
        if g > JOINT_GAP_MAX:
            continue
        pts.append(p)
    if len(pts) < 2:
        return None
    return np.mean(pts, axis=0)


def in_aisle(p: np.ndarray | None) -> bool:
    if p is None:
        return False
    for ax, (lo, hi) in AISLE_AABB.items():
        v = float(p[{"x": 0, "y": 1, "z": 2}[ax]])
        if v < lo or v > hi:
            return False
    return True


def pick_pairs(
    fl: dict,
    fr: dict,
    cams: dict,
    gap_max: float = PAIR_GAP_MAX,
    prefer: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[tuple[int, int, float]]:
    """左右路匹配：先 NMS，再续上帧，再贪心。"""
    li = nms_indices(fl["k"], fl["s"])
    ri = nms_indices(fr["k"], fr["s"])
    lxy = {i: _torso_xy(fl["k"][i], fl["s"][i]) for i in li}
    rxy = {j: _torso_xy(fr["k"][j], fr["s"][j]) for j in ri}

    def gap_ij(i: int, j: int) -> float | None:
        return _pair_gap(fl["k"][i], fl["s"][i], fr["k"][j], fr["s"][j], cams)

    used_l: set[int] = set()
    used_r: set[int] = set()
    out: list[tuple[int, int, float]] = []

    def _nearest(xy: np.ndarray, pool: dict[int, np.ndarray | None], used: set[int], max_px: float) -> int | None:
        best, best_d = None, max_px
        for idx, p in pool.items():
            if idx in used or p is None:
                continue
            d = float(np.linalg.norm(xy - p))
            if d < best_d:
                best, best_d = idx, d
        return best

    for pl, pr in prefer or []:
        i = _nearest(pl, lxy, used_l, PREFER_PX)
        j = _nearest(pr, rxy, used_r, PREFER_PX)
        if i is None or j is None:
            continue
        g = gap_ij(i, j)
        if g is None or g > gap_max:
            continue
        used_l.add(i)
        used_r.add(j)
        out.append((i, j, g))

    cands: list[tuple[float, int, int]] = []
    for i in li:
        if i in used_l:
            continue
        for j in ri:
            if j in used_r:
                continue
            g = gap_ij(i, j)
            if g is None or g > gap_max:
                continue
            cands.append((g, i, j))
    cands.sort()
    for g, i, j in cands:
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        out.append((i, j, g))

    kept: list[tuple[int, int, float]] = []
    cents: list[np.ndarray] = []
    for i, j, g in sorted(out, key=lambda x: x[2]):
        c = _torso_xyz(fl["k"][i], fl["s"][i], fr["k"][j], fr["s"][j], cams)
        if not in_aisle(c):
            continue
        if any(float(np.linalg.norm(c - p)) < DUPLICATE_M for p in cents):
            continue
        kept.append((i, j, g))
        cents.append(c)
    return kept


def keypoints_to_ks(persons: list[dict]) -> dict:
    """PoseFrame persons → {k: (N,17,2), s: (N,17)}。"""
    ks = []
    ss = []
    for p in persons or []:
        kpts = p.get("keypoints") or []
        xy = np.zeros((17, 2), np.float32)
        sc = np.zeros(17, np.float32)
        for i, kp in enumerate(kpts[:17]):
            if not isinstance(kp, (list, tuple)) or len(kp) < 2:
                continue
            xy[i, 0] = float(kp[0])
            xy[i, 1] = float(kp[1])
            sc[i] = float(kp[2]) if len(kp) > 2 else 0.0
        ks.append(xy)
        ss.append(sc)
    if not ks:
        return {"k": np.zeros((0, 17, 2), np.float32), "s": np.zeros((0, 17), np.float32)}
    return {"k": np.stack(ks), "s": np.stack(ss)}
