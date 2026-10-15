#!/usr/bin/env python3
"""用人工标的拣货面四角 + 实测尺寸，反解相机位姿与焦距。

世界系：X 横跨巷道（两拣货面在 x=±aisle/2）、Y 向上、Z 沿巷道。
每面墙四角按「顶远 / 顶近 / 底近 / 底远」顺序标注，配合该墙的宽/高/底沿离地，
角点世界坐标只剩一个未知（该墙近端的 Z）。

未知：f, camX, camH, camZ, pitch, yaw, roll + 每面墙近端 Z。
观测：每面墙 4 个角点。卷尺量的相机位姿只作初值与软先验，焦距必须解。

双路 layout：opposite=巷道两端对打（B 近端=A 远端）；same_side=同端同向
（角序恒等）。同侧可用 prior.wallDist1/wallDist2（或 wallDists、wallDist81/82）
钉到两侧墙的俯视法向距，不再强迫 camX≈0。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares

CORNER_HINTS = ("顶沿·远端", "顶沿·近端", "底沿·近端", "底沿·远端")
LAYOUT_OPPOSITE = "opposite"
LAYOUT_SAME_SIDE = "same_side"
# 对向：路 B 的近端 = 路 A 的远端。同侧同向：近对近，角序恒等。
_OPP_CORNER = (1, 0, 3, 2)
_SAME_CORNER = (0, 1, 2, 3)
# 墙1=81、墙2=82（同巷道两侧）
_WALL_DIST_ALIAS = {1: ("wallDist1", "wallDist81"), 2: ("wallDist2", "wallDist82")}


def cam_axes(pitch: float, yaw: float, roll: float):
    """与查看器 camAxes 同一约定：图像 u 向右、v 向下，无滚转时相机不倾斜。"""
    fwd = np.array([
        math.sin(yaw) * math.cos(pitch),
        -math.sin(pitch),
        math.cos(yaw) * math.cos(pitch),
    ])
    r0 = np.cross(fwd, [0.0, 1.0, 0.0])
    n = np.linalg.norm(r0)
    if n < 1e-9:
        r0 = np.array([1.0, 0.0, 0.0])
    else:
        r0 = r0 / n
    d0 = np.cross(fwd, r0)
    cr, sr = math.cos(roll), math.sin(roll)
    return cr * r0 + sr * d0, -sr * r0 + cr * d0, fwd


def _wall_corners(wall: dict, sign: int, aisle: float, znear: float) -> np.ndarray:
    x = sign * aisle / 2.0
    w = float(wall["width"])
    h = float(wall["height"])
    b = float(wall["base"])
    zf = znear + w
    return np.array([
        [x, b + h, zf],
        [x, b + h, znear],
        [x, b, znear],
        [x, b, zf],
    ])


def _project(pts: np.ndarray, z: np.ndarray, cx: float, cy: float) -> np.ndarray:
    f, camX, camH, camZ, pitch, yaw, roll = z[:7]
    right, down, fwd = cam_axes(pitch, yaw, roll)
    v = pts - np.array([camX, camH, camZ])
    zc = v @ fwd
    zc = np.where(zc < 0.05, np.nan, zc)
    return np.column_stack([cx + f * (v @ right) / zc, cy + f * (v @ down) / zc])


def _finite_dist(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        d = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(d) or d <= 0.05:
        return None
    return d


def _wall_dist(prior: dict, wall_id: Any) -> float | None:
    """俯视法向距：wallDists[id] / wallDist1 / wallDist81。"""
    if not prior or wall_id is None:
        return None
    try:
        wid = int(wall_id)
    except (TypeError, ValueError):
        return None
    dists = prior.get("wallDists")
    if isinstance(dists, dict):
        for key in (wid, str(wid)):
            d = _finite_dist(dists.get(key))
            if d is not None:
                return d
    for key in _WALL_DIST_ALIAS.get(wid, ()):
        d = _finite_dist(prior.get(key))
        if d is not None:
            return d
    return None


def _cam_x_guess(walls: list, signs: list[int], aisle: float, prior: dict) -> float:
    """巷道内侧：从墙沿 -sign 走 wallDist。多面墙取平均。"""
    xs: list[float] = []
    for w, s in zip(walls, signs):
        wd = _wall_dist(prior, w.get("wall_id"))
        if wd is None:
            continue
        wall_x = s * aisle / 2.0
        xs.append(wall_x - s * wd)
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def _normalize_layout(raw: Any) -> str:
    s = str(raw or LAYOUT_OPPOSITE).strip().lower().replace("-", "_")
    if s in ("same_side", "sameside", "same"):
        return LAYOUT_SAME_SIDE
    return LAYOUT_OPPOSITE


def _as_view_list(views: Any) -> list:
    if isinstance(views, dict):
        if "L" in views and "R" in views:
            return [views["L"], views["R"]]
        return list(views.values())
    return list(views or [])


def _corner_order(layout: str) -> tuple[int, ...]:
    return _SAME_CORNER if layout == LAYOUT_SAME_SIDE else _OPP_CORNER


def _bounds(n: int, img_w: int):
    lo = np.full(n, -12.0)
    hi = np.full(n, 12.0)
    lo[0], hi[0] = img_w * 0.25, img_w * 4.0
    lo[1], hi[1] = -2.0, 2.0
    lo[2], hi[2] = 1.2, 6.5
    lo[3], hi[3] = -12.0, 3.0
    lo[4], hi[4] = math.radians(-5), math.radians(89)
    lo[5], hi[5] = math.radians(-75), math.radians(75)
    lo[6], hi[6] = math.radians(-35), math.radians(35)
    lo[7:], hi[7:] = -6.0, 25.0
    return lo, hi


def solve(
    calib: dict,
    img_w: int,
    img_h: int,
    *,
    force_signs: list[int] | None = None,
) -> dict:
    """calib: {aisle, prior:{...}, walls:[{wall_id, quad:[[u,v]x4], width, height, base}]}"""
    walls = [w for w in calib.get("walls") or [] if len(w.get("quad") or []) == 4]
    if not walls:
        return {"ok": False, "error": "没有完整的四角标注"}
    aisle = float(calib.get("aisle") or 2.0)
    prior = calib.get("prior") or {}
    cx, cy = img_w / 2.0, img_h / 2.0
    obs = np.array([p for w in walls for p in w["quad"]], float)

    sign_sets: list[list[int]]
    if force_signs is not None:
        if len(force_signs) != len(walls):
            return {"ok": False, "error": "force_signs 与墙数量不一致"}
        sign_sets = [list(force_signs)]
    elif len(walls) == 1:
        sign_sets = [[1], [-1]]
    else:
        sign_sets = [[1, -1], [-1, 1]]

    def make_resid(signs):
        def resid(z):
            pts = np.vstack([
                _wall_corners(w, s, aisle, z[7 + i]) for i, (w, s) in enumerate(zip(walls, signs))
            ])
            uv = _project(pts, z, cx, cy)
            r = np.nan_to_num(uv - obs, nan=400.0).ravel().tolist()
            # 卷尺值作软先验：偏离才罚，不钉死
            r.append(2.0 * (z[2] - float(prior.get("camH", 2.84))))
            r.append(1.0 * (math.degrees(z[4]) - float(prior.get("pitch", 45.0))) / 10.0)
            r.append(1.0 * (math.degrees(z[5]) - float(prior.get("yaw", 0.0))) / 10.0)
            wd_res = []
            for w, s in zip(walls, signs):
                wd = _wall_dist(prior, w.get("wall_id"))
                if wd is None:
                    continue
                wall_x = s * aisle / 2.0
                # 有向：墙内侧距 = sign * (wall_x - camX)
                wd_res.append(3.0 * (s * (wall_x - z[1]) - wd))
            if wd_res:
                r.extend(wd_res)
            else:
                r.append(3.0 * z[1])
            r.append(2.0 * math.degrees(z[6]) / 10.0)
            return r
        return resid

    best: dict[str, Any] | None = None
    n = 7 + len(walls)
    for signs in sign_sets:
        for pitch0 in (20.0, 35.0, 50.0, 65.0):
            for fov0 in (60.0, 90.0, 110.0):
                z0 = np.zeros(n)
                z0[0] = (img_w / 2) / math.tan(math.radians(fov0 / 2))
                z0[1] = _cam_x_guess(walls, signs, aisle, prior)
                z0[2] = float(prior.get("camH", 2.84))
                z0[3] = -float(prior.get("camDist", 1.56))
                z0[4] = math.radians(pitch0)
                z0[5] = math.radians(float(prior.get("yaw", 0.0)))
                z0[6] = 0.0
                z0[7:] = 0.0
                try:
                    sol = least_squares(
                        make_resid(signs), z0, method="trf",
                        bounds=_bounds(n, img_w), max_nfev=3000,
                    )
                except Exception:
                    continue
                r = np.array(sol.fun[: len(obs) * 2]).reshape(-1, 2)
                rms = float(np.sqrt(np.mean(np.sum(r * r, axis=1))))
                if best is None or rms < best["resid_px"]:
                    best = {"resid_px": rms, "z": sol.x.copy(), "signs": list(signs)}
    if best is None:
        return {"ok": False, "error": "求解未收敛"}

    z = best["z"]
    znear = [float(z[7 + i]) for i in range(len(walls))]
    # 把巷道近端挪到 z=0，方便查看器摆场景
    shift = min(znear)
    fov_h = 2 * math.degrees(math.atan((img_w / 2) / z[0]))
    out_walls = []
    for i, (w, s) in enumerate(zip(walls, best["signs"])):
        pts = _wall_corners(w, s, aisle, znear[i] - shift)
        out_walls.append({
            "wall_id": w.get("wall_id"),
            "sign": int(s),
            "z_near": round(znear[i] - shift, 4),
            "corners": [[round(float(c), 4) for c in p] for p in pts],
        })
    per_corner = np.linalg.norm(
        np.nan_to_num(
            _project(
                np.vstack([
                    _wall_corners(w, s, aisle, znear[i])
                    for i, (w, s) in enumerate(zip(walls, best["signs"]))
                ]),
                z, cx, cy,
            ) - obs,
            nan=400.0,
        ),
        axis=1,
    )
    return {
        "ok": True,
        "resid_px": round(best["resid_px"], 2),
        "corner_resid_px": [round(float(v), 1) for v in per_corner],
        "aisle": aisle,
        "camera": {
            "fovH": round(fov_h, 2),
            "camX": round(float(z[1]), 4),
            "camH": round(float(z[2]), 4),
            "camDist": round(-(float(z[3]) - shift), 4),
            "pitch": round(math.degrees(float(z[4])), 3),
            "yaw": round(math.degrees(float(z[5])), 3),
            "roll": round(math.degrees(float(z[6])), 3),
        },
        "walls": out_walls,
    }


def _xform_pts(pts: np.ndarray, R: np.ndarray, t: np.ndarray, scale: float) -> np.ndarray:
    pts = np.asarray(pts, float)
    return (scale * (R @ pts.T).T + t)


def _project_cam(pts: np.ndarray, cam: dict) -> np.ndarray:
    C = np.array(cam["C"], float)
    right = np.array(cam["right"], float)
    down = np.array(cam["down"], float)
    fwd = np.array(cam["fwd"], float)
    f, cx, cy = float(cam["f"]), float(cam["cx"]), float(cam["cy"])
    v = pts - C
    zc = v @ fwd
    zc = np.where(zc < 0.05, np.nan, zc)
    return np.column_stack([cx + f * (v @ right) / zc, cy + f * (v @ down) / zc])


def _signs_by_wall_id(sol: dict) -> dict[int, int]:
    out: dict[int, int] = {}
    for w in sol.get("walls") or []:
        wid = w.get("wall_id")
        if wid is not None:
            out[int(wid)] = int(w["sign"])
    return out


def _force_signs_for_view(sign_by_id: dict[int, int], walls: list) -> list[int] | None:
    ids = [int(w["wall_id"]) for w in walls if len(w.get("quad") or []) == 4]
    if not ids or any(i not in sign_by_id for i in ids):
        return None
    return [sign_by_id[i] for i in ids]


def _merge_dual_walls(
    sol_a: dict,
    sol_b: dict,
    layout: str,
    R: np.ndarray,
    t: np.ndarray,
    scale: float,
) -> list[dict]:
    """双路对齐后：每面墙 3D 取 A 与（变换后的 B）平均，避免只信左路导致右路层线飞出画面。"""
    order = list(_corner_order(layout))
    wa = {int(w["wall_id"]): w for w in sol_a["walls"]}
    wb = {int(w["wall_id"]): w for w in sol_b["walls"]}
    merged: list[dict] = []
    for wid in sorted(set(wa.keys()) | set(wb.keys())):
        if wid in wa and wid in wb:
            zn = 0.5 * (float(wa[wid]["z_near"]) + float(wb[wid]["z_near"]))
            sign = int(wa[wid]["sign"])
        elif wid in wa:
            zn = float(wa[wid]["z_near"])
            sign = int(wa[wid]["sign"])
        else:
            zn = float(wb[wid]["z_near"])
            sign = int(wb[wid]["sign"])
        merged.append({
            "wall_id": wid,
            "sign": sign,
            "z_near": zn,
            "corners": [],
        })
    return merged


def _wall_specs_from_views(views: list) -> dict[int, dict]:
    specs: dict[int, dict] = {}
    for v in views:
        for w in v.get("walls") or []:
            wid = w.get("wall_id")
            if wid is None or wid in specs:
                continue
            specs[int(wid)] = {
                "width": float(w.get("width") or 2.2),
                "height": float(w.get("height") or 2.0),
                "base": float(w.get("base") or 0.0),
            }
    return specs


def snap_solved_walls_parametric(solved: dict, views: list, aisle: float) -> dict:
    """对外：把 solved.walls 钉回参数化竖墙（保存/反解后兜底）。"""
    if not solved or not solved.get("ok") or not solved.get("walls"):
        return solved
    fixed = _snap_merged_walls_parametric(solved["walls"], views, float(aisle or 2.0))
    return {**solved, "walls": fixed}


def _wall_x_spread(walls: list[dict]) -> float:
    s = 0.0
    for w in walls:
        xs = [p[0] for p in w.get("corners") or []]
        if len(xs) >= 2:
            s = max(s, float(max(xs) - min(xs)))
    return s


def _snap_merged_walls_parametric(walls: list[dict], views: list, aisle: float) -> list[dict]:
    """双路融合后仍保持参数化墙面：x=±aisle/2、竖直、宽/高/底沿来自标注表格。"""
    specs = _wall_specs_from_views(views)
    znear: list[float] = []
    for w in walls:
        wid = int(w["wall_id"])
        spec = specs.get(wid, {"width": 2.2, "height": 2.0, "base": 0.0})
        sign = int(w["sign"])
        if w.get("z_near") is not None:
            zn = float(w["z_near"])
        else:
            raw = w.get("corners") or []
            if len(raw) == 4:
                c = np.array(raw, float)
                zn = float(0.5 * (c[1, 2] + c[2, 2]))
            else:
                zn = 0.0
        znear.append(zn)
    shift = min(znear) if znear else 0.0
    out: list[dict] = []
    for w, zn in zip(walls, znear):
        wid = int(w["wall_id"])
        spec = specs.get(wid, {"width": 2.2, "height": 2.0, "base": 0.0})
        sign = int(w["sign"])
        zn0 = zn - shift
        pts = _wall_corners(spec, sign, aisle, zn0)
        out.append({
            "wall_id": wid,
            "sign": sign,
            "z_near": round(zn0, 4),
            "corners": [[round(float(v), 4) for v in p] for p in pts],
        })
    return out


def _refine_merged_walls(
    walls: list[dict],
    views: list,
    _sols: list,
    cameras: dict[str, dict],
    _layout: str,
) -> list[dict]:
    """在固定双相机下微调墙四角，使两路像素同时拟合。"""
    names = [v.get("name") or ("L" if i == 0 else "R") for i, v in enumerate(views)]
    obs: dict[int, dict[str, np.ndarray]] = {}
    for v, name in zip(views, names):
        for w in v.get("walls") or []:
            q = w.get("quad") or []
            if len(q) != 4:
                continue
            obs.setdefault(int(w["wall_id"]), {})[name] = np.array(q, float)

    refine_ids = [w["wall_id"] for w in walls if w["wall_id"] in obs and len(obs[w["wall_id"]]) >= 2]
    if not refine_ids:
        return walls

    by_id = {w["wall_id"]: w for w in walls}
    init_blocks = [np.array(by_id[wid]["corners"], float) for wid in refine_ids]
    x0 = np.array([c for b in init_blocks for p in b for c in p], float)

    def resid(x: np.ndarray) -> np.ndarray:
        pts = x.reshape(-1, 3)
        r: list[float] = []
        i0 = 0
        for j, wid in enumerate(refine_ids):
            block = pts[i0 : i0 + 4]
            i0 += 4
            for name, cam in cameras.items():
                if name not in obs[wid]:
                    continue
                uv = _project_cam(block, cam)
                diff = np.nan_to_num(uv - obs[wid][name], nan=80.0)
                r.extend(diff.ravel().tolist())
            r.extend((0.15 * (block - init_blocks[j])).ravel().tolist())
        return r

    try:
        sol = least_squares(resid, x0, method="trf", max_nfev=800)
    except Exception:
        return walls
    pts = sol.x.reshape(-1, 3)
    i0 = 0
    out = []
    for w in walls:
        if w["wall_id"] not in refine_ids:
            out.append(w)
            continue
        block = pts[i0 : i0 + 4]
        i0 += 4
        out.append({
            **w,
            "corners": [[round(float(c), 4) for c in p] for p in block],
        })
    return out


def _wall_reproj_px(walls: list[dict], views: list, cameras: dict[str, dict]) -> dict[str, dict[int, float]]:
    """每路每面墙四角最大像素误差。"""
    names = [v.get("name") or ("L" if i == 0 else "R") for i, v in enumerate(views)]
    by_id = {int(w["wall_id"]): w for w in walls}
    out: dict[str, dict[int, float]] = {n: {} for n in names}
    for v, name in zip(views, names):
        cam = cameras.get(name)
        if not cam:
            continue
        for w in v.get("walls") or []:
            q = w.get("quad") or []
            if len(q) != 4:
                continue
            wid = int(w["wall_id"])
            wall = by_id.get(wid)
            if not wall:
                continue
            uv = _project_cam(np.array(wall["corners"], float), cam)
            err = np.linalg.norm(np.nan_to_num(uv - np.array(q, float), nan=400.0), axis=1)
            out[name][wid] = round(float(np.max(err)), 1)
    return out


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """刚体+均匀缩放：dst ≈ s R src + t。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    x, y = src - mu_s, dst - mu_d
    var = float(np.mean(np.sum(x * x, axis=1)))
    u, s, vt = np.linalg.svd(y.T @ x / len(src))
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1
    R = u @ np.diag(d) @ vt
    scale = float(np.sum(s * d) / var) if var > 1e-12 else 1.0
    t = mu_d - scale * R @ mu_s
    return R, t, scale


def _cam_center(sol: dict) -> np.ndarray:
    c = sol["camera"]
    # solve() 把近端挪到 z=0 后 camDist = -(camZ - shift)，故 camZ = -camDist
    return np.array([c["camX"], c["camH"], -c["camDist"]], float)


def _cam_bundle(sol: dict, img_w: int, img_h: int, R=None, t=None, scale: float = 1.0) -> dict:
    c = sol["camera"]
    right, down, fwd = cam_axes(
        math.radians(c["pitch"]), math.radians(c["yaw"]), math.radians(c["roll"])
    )
    C = _cam_center(sol)
    if R is not None:
        right, down, fwd = R @ right, R @ down, R @ fwd
        C = scale * (R @ C) + t
    f = (img_w / 2.0) / math.tan(math.radians(c["fovH"] / 2.0))
    return {
        "f": round(float(f), 3),
        "cx": round(img_w / 2.0, 2),
        "cy": round(img_h / 2.0, 2),
        "C": [round(float(v), 4) for v in C],
        "right": [round(float(v), 6) for v in right],
        "down": [round(float(v), 6) for v in down],
        "fwd": [round(float(v), 6) for v in fwd],
        "fovH": c["fovH"],
        "camH": round(float(C[1]), 4),
        "resid_px": sol["resid_px"],
        "corner_resid_px": sol.get("corner_resid_px") or [],
    }


def _marked_walls(view: dict) -> list:
    return [w for w in (view.get("walls") or []) if len(w.get("quad") or []) == 4]


def _wall_sign(wall_id: int) -> int:
    """墙1=81 在 x 负侧，墙2=82 在 x 正侧。"""
    return -1 if int(wall_id) == 1 else 1


def _ground_ready(rect: Any) -> bool:
    if not isinstance(rect, dict):
        return False
    if _finite_dist(rect.get("a")) is None or _finite_dist(rect.get("b")) is None:
        return False
    for key in ("L", "R"):
        pts = rect.get(key) or []
        if len(pts) != 4:
            return False
    return True


def _match_shelf_bottom(shelf_quad, ground_quad, max_px: float = 30.0) -> dict[str, int] | None:
    """货架底边（底近、底远）落在地面矩形的哪两个角上。对不上说明底边没和地面重合。"""
    q = np.asarray(shelf_quad, float)
    g = np.asarray(ground_quad, float)
    out: dict[str, int] = {}
    used: set[int] = set()
    for shelf_i, name in ((2, "near"), (3, "far")):
        dist = np.linalg.norm(g - q[shelf_i], axis=1)
        pick = None
        for j in np.argsort(dist):
            j = int(j)
            if j in used or float(dist[j]) > max_px:
                continue
            pick = j
            break
        if pick is None:
            return None
        used.add(pick)
        out[name] = pick
    return out


def _cam_x_from_wall_dists(prior: dict, wall_ids: list[int], aisle: float) -> float | None:
    """两面墙的法向距都能推出同一个 camX。标了的那一面和没标的那一面都用。"""
    xs: list[float] = []
    for wid in wall_ids:
        wd = _wall_dist(prior, wid)
        if wd is None:
            continue
        sign = _wall_sign(wid)
        wall_x = sign * aisle / 2.0
        xs.append(wall_x - sign * wd)
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _ground_corners(x: float, z: float, theta: float, a: float, b: float, flip: float) -> np.ndarray:
    """地面 y=0。①→② 长 a，②→③ 长 b。flip=±1 决定矩形往哪一侧转。"""
    c, s = math.cos(theta), math.sin(theta)
    u = np.array([c, 0.0, s])
    v = np.array([-s, 0.0, c]) * float(flip)
    p0 = np.array([x, 0.0, z])
    return np.stack([p0, p0 + a * u, p0 + a * u + b * v, p0 + b * v])


def _pack_camera(z7: np.ndarray, img_w: int, img_h: int, resid_px: float, shift: float) -> dict:
    f, cam_x, cam_h, cam_z, pitch, yaw, roll = [float(v) for v in z7]
    cam_z -= shift
    right, down, fwd = cam_axes(pitch, yaw, roll)
    fov_h = 2 * math.degrees(math.atan((img_w / 2.0) / f))
    return {
        "f": round(f, 3),
        "cx": round(img_w / 2.0, 2),
        "cy": round(img_h / 2.0, 2),
        "C": [round(cam_x, 4), round(cam_h, 4), round(cam_z, 4)],
        "right": [round(float(v), 6) for v in right],
        "down": [round(float(v), 6) for v in down],
        "fwd": [round(float(v), 6) for v in fwd],
        "fovH": round(fov_h, 2),
        "camH": round(cam_h, 4),
        "resid_px": round(resid_px, 2),
    }


def _init_cam_z(view: dict, aisle: float, img_w: int, img_h: int, wall: dict) -> np.ndarray:
    """单路只靠自己那面墙出初值。近端先放在 z=0，camZ = -离近端。"""
    prior = view.get("prior") or {}
    sign = _wall_sign(int(wall["wall_id"]))
    res = solve(
        {"aisle": aisle, "prior": prior, "walls": [wall]},
        img_w, img_h, force_signs=[sign],
    )
    if res.get("ok") and float(res["resid_px"]) < 12.0:
        c = res["camera"]
        f = (img_w / 2.0) / math.tan(math.radians(float(c["fovH"]) / 2.0))
        return np.array([
            f, c["camX"], c["camH"], -c["camDist"],
            math.radians(c["pitch"]), math.radians(c["yaw"]), math.radians(c["roll"]),
        ], float)
    cam_x = _cam_x_guess([wall], [sign], aisle, prior)
    f = (img_w / 2.0) / math.tan(math.radians(90.0 / 2.0))
    return np.array([
        f, cam_x, float(prior.get("camH", 2.84)), -float(prior.get("camDist", 1.56)),
        math.radians(float(prior.get("pitch", 45.0))),
        math.radians(float(prior.get("yaw", 0.0))),
        0.0,
    ], float)


def solve_split_ground(payload: dict, views: list) -> dict:
    """同侧、左右各标一面不同的货架，再用地面矩形接到同一个巷道系。

    货架给出宽高，地面矩形给出边①→②、边②→③的米制长度。
    相机高、离近端、到该货架的法向距只作软先验。
    """
    rect = payload.get("ground_rect") or {}
    if not _ground_ready(rect):
        return {
            "ok": False,
            "error": "左右路没有共同的墙。请在两路都点满同一块地面矩形的四个角，并填写边长",
        }
    aisle = float(payload.get("aisle") or 2.0)
    names = [v.get("name") or ("L" if i == 0 else "R") for i, v in enumerate(views)]
    sizes = [(v.get("image_size") or [1280, 720])[:2] for v in views]
    marked = [_marked_walls(v) for v in views]
    if any(len(ws) != 1 for ws in marked):
        return {"ok": False, "error": "分墙反解时每路只标一面货架的四个角"}
    wall_by_id: dict[int, dict] = {}
    for ws in marked:
        wall_by_id[int(ws[0]["wall_id"])] = ws[0]
    wall_ids = sorted(wall_by_id)
    if len(wall_ids) < 2:
        return {"ok": False, "error": "两路标的是同一面墙，请走共同墙对齐"}
    obs_wall = []
    for v, ws in zip(views, marked):
        obs_wall.append(np.array(ws[0]["quad"], float))
    obs_ground = [np.array(rect[name], float) for name in names]
    # 地面四个角 = 两面货架的底近、底远。底边在 y=base，和地面标注是同一组点。
    slots: list[tuple[int, str] | None] = [None] * 4
    for ws, gobs in zip(marked, obs_ground):
        wid = int(ws[0]["wall_id"])
        links = _match_shelf_bottom(ws[0]["quad"], gobs)
        if links is None:
            return {
                "ok": False,
                "error": "货架底边要和地面矩形重合。请把地面的两个角点标在这面货架的底边上",
            }
        for name, gi in links.items():
            prev = slots[gi]
            if prev is not None and prev != (wid, name):
                return {
                    "ok": False,
                    "error": "左右路同一个地面角对到了不同的货架底点，请按同一套角序点",
                }
            slots[gi] = (wid, name)
    if any(s is None for s in slots):
        return {
            "ok": False,
            "error": "地面四个角还没被两面货架的底边盖全。矩形应一头贴 81 底边、一头贴 82 底边",
        }
    z_cams = [
        _init_cam_z(v, aisle, int(sz[0]), int(sz[1]), ws[0])
        for v, sz, ws in zip(views, sizes, marked)
    ]
    for i, v in enumerate(views):
        prior = v.get("prior") or payload.get("prior") or {}
        cam_x = _cam_x_from_wall_dists(prior, wall_ids, aisle)
        if cam_x is not None:
            z_cams[i][1] = cam_x
    # 参数：两路各 7 + 共用的近端 Z。地面角由两面墙的底边决定，不再单独摆。
    n = 15

    def project_view(zcam: np.ndarray, pts: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        return _project(pts, zcam, img_w / 2.0, img_h / 2.0)

    def sign_of(wid: int, flipped: bool) -> int:
        s = _wall_sign(wid)
        return -s if flipped else s

    def shelf_bottoms(zn: float, flipped: bool) -> dict[tuple[int, str], np.ndarray]:
        out: dict[tuple[int, str], np.ndarray] = {}
        for wid in wall_ids:
            spec = wall_by_id[wid]
            x = sign_of(wid, flipped) * aisle / 2.0
            y = float(spec.get("base") or 0.0)
            width = float(spec["width"])
            out[(wid, "near")] = np.array([x, y, zn])
            out[(wid, "far")] = np.array([x, y, zn + width])
        return out

    def ground_at(zn: float, flipped: bool) -> np.ndarray:
        btm = shelf_bottoms(zn, flipped)
        return np.stack([btm[slots[i]] for i in range(4)])

    def pixel_resid(x: np.ndarray, flipped: bool) -> tuple[list[float], list[np.ndarray]]:
        zn = float(x[14])
        g = ground_at(zn, flipped)
        chunks = []
        r: list[float] = []
        for i, ws in enumerate(marked):
            wid = int(ws[0]["wall_id"])
            block = _wall_corners(wall_by_id[wid], sign_of(wid, flipped), aisle, zn)
            iw, ih = int(sizes[i][0]), int(sizes[i][1])
            zcam = x[i * 7 : i * 7 + 7]
            uv_w = project_view(zcam, block, iw, ih)
            uv_g = project_view(zcam, g, iw, ih)
            dw = np.nan_to_num(uv_w - obs_wall[i], nan=400.0)
            dg = np.nan_to_num(uv_g - obs_ground[i], nan=400.0)
            r.extend(dw.ravel().tolist())
            r.extend(dg.ravel().tolist())
            chunks.append(np.vstack([dw, dg]))
        return r, chunks

    def resid(x: np.ndarray, flipped: bool) -> list[float]:
        r, _chunks = pixel_resid(x, flipped)
        zn = float(x[14])
        for i, v in enumerate(views):
            prior = v.get("prior") or payload.get("prior") or {}
            zcam = x[i * 7 : i * 7 + 7]
            r.append(8.0 * (float(zcam[2]) - float(prior.get("camH", 2.84))))
            r.append(1.0 * (math.degrees(float(zcam[4])) - float(prior.get("pitch", 45.0))) / 10.0)
            r.append(1.0 * (math.degrees(float(zcam[5])) - float(prior.get("yaw", 0.0))) / 10.0)
            r.append(2.0 * math.degrees(float(zcam[6])) / 10.0)
            for wid in wall_ids:
                wd = _wall_dist(prior, wid)
                if wd is None:
                    continue
                sign = sign_of(wid, flipped)
                wall_x = sign * aisle / 2.0
                r.append(12.0 * (sign * (wall_x - float(zcam[1])) - wd))
            cd = _finite_dist(prior.get("camDist"))
            if cd is not None:
                # 相机在货架近端外侧 camDist 米，两面墙共用这个近端
                r.append(8.0 * (float(zcam[3]) - (zn - cd)))
        return r

    def pixel_rms(x: np.ndarray, flipped: bool) -> float:
        _r, chunks = pixel_resid(x, flipped)
        err = np.vstack(chunks)
        return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))

    def cam_x_of(prior: dict, flipped: bool) -> float | None:
        xs: list[float] = []
        for wid in wall_ids:
            wd = _wall_dist(prior, wid)
            if wd is None:
                continue
            sign = sign_of(wid, flipped)
            wall_x = sign * aisle / 2.0
            xs.append(wall_x - sign * wd)
        if not xs:
            return None
        return float(sum(xs) / len(xs))

    best = None
    for flipped in (False, True):
        x0 = np.zeros(n)
        x0[0:7] = z_cams[0]
        x0[7:14] = z_cams[1]
        lo = np.full(n, -8.0)
        hi = np.full(n, 12.0)
        for i, v in enumerate(views):
            prior = v.get("prior") or {}
            iw = int(sizes[i][0])
            base = i * 7
            lo[base], hi[base] = iw * 0.25, iw * 4.0
            cam_x = cam_x_of(prior, flipped)
            if cam_x is not None:
                x0[base + 1] = cam_x
                lo[base + 1], hi[base + 1] = cam_x - 0.35, cam_x + 0.35
            else:
                lo[base + 1], hi[base + 1] = -aisle, aisle
            lo[base + 2], hi[base + 2] = 1.2, 6.5
            lo[base + 3], hi[base + 3] = -12.0, 4.0
            lo[base + 4], hi[base + 4] = math.radians(-5), math.radians(89)
            lo[base + 5], hi[base + 5] = math.radians(-75), math.radians(75)
            lo[base + 6], hi[base + 6] = math.radians(-35), math.radians(35)
        lo[14], hi[14] = -1.0, 8.0
        x0 = np.minimum(np.maximum(x0, lo + 1e-6), hi - 1e-6)
        try:
            sol = least_squares(
                lambda x, fp=flipped: resid(x, fp), x0,
                method="trf", bounds=(lo, hi), max_nfev=2000,
            )
        except Exception:
            continue
        rms = pixel_rms(sol.x, flipped)
        if best is None or rms < best["rms"]:
            best = {"rms": rms, "x": sol.x.copy(), "flipped": flipped}
    if best is None:
        return {"ok": False, "error": "分墙地面反解未收敛"}
    x = best["x"]
    rms = best["rms"]
    flipped = best["flipped"]
    zn = float(x[14])
    shift = zn
    walls_out = []
    for wid in wall_ids:
        pts = _wall_corners(wall_by_id[wid], sign_of(wid, flipped), aisle, 0.0)
        walls_out.append({
            "wall_id": wid,
            "sign": sign_of(wid, flipped),
            "z_near": 0.0,
            "corners": [[round(float(c), 4) for c in p] for p in pts],
        })
    g = ground_at(0.0, flipped)
    cameras = {}
    per_view = {}
    ground_reproj: dict[str, float] = {}
    _r, chunks = pixel_resid(x, flipped)
    for i, name in enumerate(names):
        iw, ih = int(sizes[i][0]), int(sizes[i][1])
        zcam = x[i * 7 : i * 7 + 7].copy()
        view_rms = float(np.sqrt(np.mean(np.sum(chunks[i] * chunks[i], axis=1))))
        g_uv = project_view(zcam, ground_at(zn, flipped), iw, ih)
        g_err = np.linalg.norm(np.nan_to_num(g_uv - obs_ground[i], nan=400.0), axis=1)
        ground_reproj[name] = round(float(np.max(g_err)), 1)
        cameras[name] = _pack_camera(zcam, iw, ih, view_rms, shift)
        per_view[name] = {"resid_px": round(view_rms, 2), "camera": {
            "camX": cameras[name]["C"][0],
            "camH": cameras[name]["camH"],
            "camDist": round(-cameras[name]["C"][2], 4),
        }}
    reproj = _wall_reproj_px(walls_out, views, cameras)
    return {
        "ok": True,
        "method": "split_ground",
        "layout": LAYOUT_SAME_SIDE,
        "aisle": aisle,
        "align_scale": 1.0,
        "align_rms_m": 0.0,
        "align_wall_id": wall_ids[0],
        "align_wall_ids": wall_ids,
        "cameras": cameras,
        "walls": walls_out,
        "wall_reproj_px": reproj,
        "ground": {
            "a": float(rect.get("a") or 0.0),
            "b": float(rect.get("b") or 0.0),
            "corners": [[round(float(c), 4) for c in p] for p in g],
            "tied_to_shelf_base": True,
        },
        "ground_reproj_px": ground_reproj,
        "per_view": per_view,
        "pixel_rms": round(rms, 2),
    }


def solve_dual(payload: dict) -> dict:
    """两路各自反解，再把路 B 对到路 A 的巷道坐标。

    每路只保留四角标齐的墙。左右路可以都只标同一面（另一面 quad 为空），
    用这一面的角把两路相机放进同一个巷道系。
    同侧若左右各标一面不同的货架，则改用地面矩形把两路接到同一巷道。
    """
    views = _as_view_list(payload.get("views"))
    if len(views) != 2:
        return {"ok": False, "error": "需要恰好两路"}
    layout = _normalize_layout(payload.get("layout"))
    order = _corner_order(layout)
    sols = []
    for i, v in enumerate(views):
        w, h = (v.get("image_size") or [1280, 720])[:2]
        cal = {
            "aisle": payload.get("aisle"),
            "prior": v.get("prior") or payload.get("prior") or {},
            "walls": v.get("walls") or [],
        }
        force = None
        if i == 1 and sols:
            marked = [w for w in v.get("walls") or [] if len(w.get("quad") or []) == 4]
            force = _force_signs_for_view(_signs_by_wall_id(sols[0]), marked)
        res = solve(cal, int(w), int(h), force_signs=force)
        if not res.get("ok"):
            return {"ok": False, "error": f"{v.get('name')}: {res.get('error')}"}
        sols.append(res)

    wa = {w["wall_id"]: w for w in sols[0]["walls"]}
    wb = {w["wall_id"]: w for w in sols[1]["walls"]}
    common = [k for k in wa if k in wb]
    if not common:
        if layout == LAYOUT_SAME_SIDE and _ground_ready(payload.get("ground_rect")):
            return solve_split_ground(payload, views)
        return {
            "ok": False,
            "error": "左右路要标同一面墙（同一墙号的四角都齐）。现在两路没有共同的完整墙",
        }
    # 对向仍只拿一面（墙1 优先）做近远对调。同侧一面或两面都行：标齐的墙都参与对齐。
    ids = sorted(common, key=lambda k: (k != 1, k))
    if layout != LAYOUT_SAME_SIDE:
        ids = ids[:1]
    A = np.vstack([np.array(wa[i]["corners"], float) for i in ids])
    B = np.vstack([np.array(wb[i]["corners"], float)[list(order)] for i in ids])
    R, t, scale = _umeyama(B, A)
    fit = float(np.sqrt(np.mean(np.sum((scale * (R @ B.T).T + t - A) ** 2, axis=1))))
    if abs(scale - 1.0) > 0.15:
        return {
            "ok": False,
            "error": f"对齐尺度 {scale:.3f} 偏离 1 太多，检查是否标的同一面、尺寸是否一致",
            "layout": layout,
            "align_scale": round(scale, 3),
            "align_rms_m": round(fit, 4),
        }

    size_a = (views[0].get("image_size") or [1280, 720])[:2]
    size_b = (views[1].get("image_size") or [1280, 720])[:2]
    cam_a = _cam_bundle(sols[0], int(size_a[0]), int(size_a[1]))
    cam_b = _cam_bundle(sols[1], int(size_b[0]), int(size_b[1]), R, t, scale)
    name_a = views[0].get("name") or "L"
    name_b = views[1].get("name") or "R"
    cameras = {name_a: cam_a, name_b: cam_b}
    aisle_f = float(payload.get("aisle") or sols[0].get("aisle") or 2.0)
    walls = _merge_dual_walls(sols[0], sols[1], layout, R, t, scale)
    walls = _snap_merged_walls_parametric(walls, views, aisle_f)
    reproj = _wall_reproj_px(walls, views, cameras)
    return {
        "ok": True,
        "layout": layout,
        "aisle": sols[0]["aisle"],
        "align_scale": round(scale, 4),
        "align_rms_m": round(fit, 4),
        "align_wall_id": ids[0],
        "align_wall_ids": ids,
        "cameras": cameras,
        "walls": walls,
        "wall_reproj_px": reproj,
        "per_view": {
            name_a: {"resid_px": sols[0]["resid_px"], "camera": sols[0]["camera"]},
            name_b: {"resid_px": sols[1]["resid_px"], "camera": sols[1]["camera"]},
        },
    }
