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


def solve_dual(payload: dict) -> dict:
    """两路各自反解，再把路 B 对到路 A 的巷道坐标。

    每路只保留四角标齐的墙。左右路可以都只标同一面（另一面 quad 为空），
    用这一面的角把两路相机放进同一个巷道系。
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
