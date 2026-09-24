"""平面单应（homography）腕点碰撞：不解相机、不三角化，只用每路墙四角。

原理：墙面是平面，像素 ↔ 墙面坐标是一个 3×3 单应 H，4 个角恰好定死，
与焦距、相机位置无关。腕点若真在墙面上，两路各自过 H 得到的墙面坐标重合；
腕点悬在巷道里，两条视线各飞到墙面的落点分开，分开距离随离墙距离单调增。

    pc = PlaneContact.from_calib(calib)            # 每面墙、每路一个 H
    r = pc.wrist(uv_l, s_l, uv_r, s_r)             # → 最贴合那面墙的判定

墙面坐标：z 沿墙（近端 0 → 远端 width），y 离地高。与 solve_scene/dualcam_geom
的世界系一致：世界 = (wall_x, y, z_near + z)。

用途：判「手腕进哪个格、碰没碰」。同侧双机沿视线深度不可靠，这条路绕开了它。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from scripts.dualcam_geom import contact_slots, project_pix

LWRIST, RWRIST = 9, 10
KPT_MIN = 0.3
DEFAULT_TAU = 0.15   # 两路落点重合距离阈值（米）；按现场分布调
# 有向分离的参考方向：把墙面中心往巷道里推这么远，看两路落点朝哪边分开
_REF_INWARD_M = 0.5


def homography_to_wall(quad: Any, width: float, height: float, base: float = 0.0) -> np.ndarray:
    """像素四角（①顶远 ②顶近 ③底近 ④底远）→ 墙面坐标 (z, y) 的单应。"""
    src = np.asarray(quad, np.float64).reshape(4, 2)
    dst = np.array([
        [width, base + height],
        [0.0, base + height],
        [0.0, base],
        [width, base],
    ], np.float64)
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise ValueError("四角退化，无法建单应")
    return H


def apply_h(H: np.ndarray, uv: Any) -> np.ndarray:
    p = H @ np.array([float(uv[0]), float(uv[1]), 1.0])
    if abs(p[2]) < 1e-12:
        return np.array([np.nan, np.nan])
    return p[:2] / p[2]


class WallH:
    """一面墙：两路 H + 世界系位置（用于回 3D 与货格判定）。"""

    def __init__(
        self,
        wall_id: int,
        width: float,
        height: float,
        base: float,
        H: dict[str, np.ndarray],
        wall_x: float,
        z_near: float,
        sign: int,
        ref_dir: np.ndarray | None,
    ) -> None:
        self.wall_id = wall_id
        self.width = width
        self.height = height
        self.base = base
        self.H = H
        self.wall_x = wall_x
        self.z_near = z_near
        self.sign = sign
        self.ref_dir = ref_dir  # 墙面坐标里「落点 L − 落点 R」在腕悬于巷道时的方向（单位向量）

    def to_world(self, zy: Any) -> np.ndarray:
        return np.array([self.wall_x, float(zy[1]), self.z_near + float(zy[0])])

    def map(self, view: str, uv: Any) -> np.ndarray:
        return apply_h(self.H[view], uv)

    def inside(self, zy: Any, margin: float = 0.2) -> bool:
        z, y = float(zy[0]), float(zy[1])
        return (-margin <= z <= self.width + margin) and (self.base - margin <= y <= self.base + self.height + margin)


class PlaneContact:
    def __init__(self, walls: list[WallH], meshes: list[dict], solved: dict | None, tau: float = DEFAULT_TAU) -> None:
        self.walls = walls
        self.meshes = meshes or []
        self.solved = solved or {"walls": []}
        self.tau = float(tau)

    @classmethod
    def from_calib(cls, calib: dict, tau: float = DEFAULT_TAU) -> "PlaneContact":
        """只收两路四角都标齐的墙。z_near / wall_x 取 solved（有则用，无则按 x=±aisle/2、z_near=0）。"""
        views = calib.get("views") or {}
        solved = calib.get("solved") if isinstance(calib.get("solved"), dict) else None
        aisle = float(calib.get("aisle") or 2.0)
        solved_walls = {int(w["wall_id"]): w for w in (solved or {}).get("walls") or []}
        cams = (solved or {}).get("cameras") or {}
        out: list[WallH] = []
        specs: dict[int, dict] = {}
        quads: dict[int, dict[str, list]] = {}
        for v in ("L", "R"):
            for w in (views.get(v) or {}).get("walls") or []:
                q = w.get("quad") or []
                if len(q) != 4:
                    continue
                wid = int(w["wall_id"])
                quads.setdefault(wid, {})[v] = q
                specs.setdefault(wid, {
                    "width": float(w.get("width") or 2.2),
                    "height": float(w.get("height") or 2.0),
                    "base": float(w.get("base") or 0.0),
                })
        for wid, qs in sorted(quads.items()):
            if "L" not in qs or "R" not in qs:
                continue
            sp = specs[wid]
            H = {v: homography_to_wall(qs[v], sp["width"], sp["height"], sp["base"]) for v in ("L", "R")}
            sw = solved_walls.get(wid)
            if sw is not None:
                corners = np.asarray(sw["corners"], float)
                wall_x = float(corners[0][0])
                z_near = float(corners[1][2])
                sign = int(sw.get("sign", -1 if wall_x < 0 else 1))
            else:
                sign = -1 if wid == 1 else 1
                wall_x = sign * aisle / 2.0
                z_near = 0.0
            ref = None
            if cams.get("L") and cams.get("R"):
                # 墙面中心往巷道里推 0.5m 的点，投回两路再过 H，得到「悬空时 L−R 分离方向」
                inward = np.array([1.0 if sign < 0 else -1.0, 0.0, 0.0])
                center = np.array([wall_x, sp["base"] + sp["height"] / 2.0, z_near + sp["width"] / 2.0])
                p = center + _REF_INWARD_M * inward
                uvl, uvr = project_pix(p, cams["L"]), project_pix(p, cams["R"])
                if uvl is not None and uvr is not None:
                    d = apply_h(H["L"], uvl) - apply_h(H["R"], uvr)
                    n = float(np.linalg.norm(d))
                    if np.isfinite(n) and n > 1e-6:
                        ref = d / n
            out.append(WallH(wid, sp["width"], sp["height"], sp["base"], H, wall_x, z_near, sign, ref))
        meshes = [m for m in calib.get("slot_meshes") or [] if isinstance(m, dict)]
        return cls(out, meshes, solved, tau)

    @property
    def wall_ids(self) -> list[int]:
        return [w.wall_id for w in self.walls]

    def _cell(self, wall: WallH, zy: np.ndarray) -> str | None:
        p = wall.to_world(zy)
        meshes = [m for m in self.meshes if int(m.get("wall_id") or 0) == wall.wall_id]
        if not meshes or not self.solved.get("walls"):
            return None
        hits = contact_slots(p, meshes, self.solved, contact_m=1e9)
        return hits[0]["box_id"] if hits else None

    def wrist_on(self, wall: WallH, uv_l: Any, uv_r: Any) -> dict[str, Any]:
        a = wall.map("L", uv_l)
        b = wall.map("R", uv_r)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return {"wall": wall.wall_id, "dist": None}
        diff = a - b
        dist = float(np.linalg.norm(diff))
        mid = 0.5 * (a + b)
        sgn = None
        if wall.ref_dir is not None:
            # >0 悬在巷道里；<0 伸进了货架；|sgn|≈dist
            sgn = float(diff @ wall.ref_dir)
        return {
            "wall": wall.wall_id,
            "zy": [round(float(mid[0]), 3), round(float(mid[1]), 3)],
            "zyL": [round(float(a[0]), 3), round(float(a[1]), 3)],
            "zyR": [round(float(b[0]), 3), round(float(b[1]), 3)],
            "dist": round(dist, 3),
            "sgn": None if sgn is None else round(sgn, 3),
            "in": wall.inside(mid),
            "cell": self._cell(wall, mid) if wall.inside(mid) else None,
        }

    def wrist(self, uv_l: Any, s_l: float, uv_r: Any, s_r: float, kpt_min: float = KPT_MIN) -> dict[str, Any] | None:
        """两路都过门槛才判；多面墙取重合距离最小且落点在墙面范围内的那面。"""
        if uv_l is None or uv_r is None or float(s_l) < kpt_min or float(s_r) < kpt_min:
            return None
        best: dict[str, Any] | None = None
        for wall in self.walls:
            r = self.wrist_on(wall, uv_l, uv_r)
            if r.get("dist") is None:
                continue
            key = (0 if r.get("in") else 1, r["dist"])
            if best is None or key < (0 if best.get("in") else 1, best["dist"]):
                best = r
        if best is None:
            return None
        best["contact"] = bool(best.get("in") and best["dist"] <= self.tau and best.get("cell"))
        return best

    def person(self, k_l: Any, s_l: Any, k_r: Any, s_r: Any) -> dict[str, Any]:
        """两腕一次算完。返回 {"9": {...}|None, "10": {...}|None}。"""
        out: dict[str, Any] = {}
        for j in (LWRIST, RWRIST):
            r = None
            if k_l is not None and k_r is not None and s_l is not None and s_r is not None:
                r = self.wrist(k_l[j], float(s_l[j]), k_r[j], float(s_r[j]))
            out[str(j)] = r
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "tau": self.tau,
            "walls": [
                {
                    "wall_id": w.wall_id, "x": round(w.wall_x, 4), "z_near": round(w.z_near, 4),
                    "width": w.width, "height": w.height, "base": w.base, "sign": w.sign,
                    "ref_dir": None if w.ref_dir is None else [round(float(v), 4) for v in w.ref_dir],
                }
                for w in self.walls
            ],
        }
