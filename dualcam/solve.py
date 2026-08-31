"""巷道四角反解（从 pick-state solve_scene 迁入）。

世界系：X 横跨巷道（两拣货面在 x=±aisle/2）、Y 向上、Z 沿巷道。
每面墙四角按「顶远 / 顶近 / 底近 / 底远」顺序标注，配合该墙的宽/高/底沿离地，
角点世界坐标只剩一个未知（该墙近端的 Z）。

未知：f, camX, camH, camZ, pitch, yaw, roll + 每面墙近端 Z。
观测：每面墙 4 个角点。卷尺量的相机位姿只作初值与软先验，焦距必须解。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares

CORNER_HINTS = ("顶沿·远端", "顶沿·近端", "底沿·近端", "底沿·远端")


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


def solve(calib: dict, img_w: int, img_h: int) -> dict:
    """calib: {aisle, prior:{...}, walls:[{wall_id, quad:[[u,v]x4], width, height, base}]}"""
    walls = [w for w in calib.get("walls") or [] if len(w.get("quad") or []) == 4]
    if not walls:
        return {"ok": False, "error": "没有完整的四角标注"}
    aisle = float(calib.get("aisle") or 2.0)
    prior = calib.get("prior") or {}
    cx, cy = img_w / 2.0, img_h / 2.0
    obs = np.array([p for w in walls for p in w["quad"]], float)

    sign_sets: list[list[int]]
    if len(walls) == 1:
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
                z0[1] = 0.0
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


# 对向双机：每路按「相对本机 顶远/顶近/底近/底远」标同一面墙。
# 路 B 的近端 = 路 A 的远端，对应角序 [1, 0, 3, 2]。
_OPP_CORNER = (1, 0, 3, 2)


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
    """两路各自反解，再把路 B 对到路 A 的巷道坐标。"""
    views = payload.get("views") or []
    if len(views) != 2:
        return {"ok": False, "error": "需要恰好两路"}
    sols = []
    for v in views:
        w, h = (v.get("image_size") or [1280, 720])[:2]
        cal = {
            "aisle": payload.get("aisle"),
            "prior": v.get("prior") or payload.get("prior") or {},
            "walls": v.get("walls") or [],
        }
        res = solve(cal, int(w), int(h))
        if not res.get("ok"):
            return {"ok": False, "error": f"{v.get('name')}: {res.get('error')}"}
        sols.append(res)

    wa = {w["wall_id"]: w for w in sols[0]["walls"]}
    wb = {w["wall_id"]: w for w in sols[1]["walls"]}
    common = [k for k in wa if k in wb]
    if not common:
        return {"ok": False, "error": "两路没有同 id 的墙，无法对齐"}
    wid = 1 if 1 in common else common[0]
    A = np.array(wa[wid]["corners"], float)
    B = np.array(wb[wid]["corners"], float)[list(_OPP_CORNER)]
    R, t, scale = _umeyama(B, A)
    fit = float(np.sqrt(np.mean(np.sum((scale * (R @ B.T).T + t - A) ** 2, axis=1))))
    if abs(scale - 1.0) > 0.15:
        return {
            "ok": False,
            "error": f"对齐尺度 {scale:.3f} 偏离 1 太多，检查是否标的同一面、尺寸是否一致",
            "align_scale": round(scale, 3),
            "align_rms_m": round(fit, 4),
        }

    size_a = (views[0].get("image_size") or [1280, 720])[:2]
    size_b = (views[1].get("image_size") or [1280, 720])[:2]
    cam_a = _cam_bundle(sols[0], int(size_a[0]), int(size_a[1]))
    cam_b = _cam_bundle(sols[1], int(size_b[0]), int(size_b[1]), R, t, scale)
    walls = []
    for w in sols[0]["walls"]:
        walls.append({
            "wall_id": w["wall_id"],
            "sign": w["sign"],
            "z_near": w["z_near"],
            "corners": w["corners"],
        })
    return {
        "ok": True,
        "aisle": sols[0]["aisle"],
        "align_scale": round(scale, 4),
        "align_rms_m": round(fit, 4),
        "align_wall_id": wid,
        "cameras": {views[0].get("name") or "L": cam_a, views[1].get("name") or "R": cam_b},
        "walls": walls,
        "per_view": {
            views[0].get("name") or "L": {"resid_px": sols[0]["resid_px"], "camera": sols[0]["camera"]},
            views[1].get("name") or "R": {"resid_px": sols[1]["resid_px"], "camera": sols[1]["camera"]},
        },
    }
