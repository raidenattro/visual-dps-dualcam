"""双路货格几何：3D 网格为唯一真值，两路画面都是投影。

世界系与 solve_scene 相同：X 横跨巷道、Y 向上、Z 沿巷道。
墙角顺序与标注一致：①顶远 ②顶近 ③底近 ④底远（远近相对左路）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-8
MIN_DEPTH = 0.05


def _v(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)


def project_pix(p: Any, cam: dict) -> list[float] | None:
    """世界点 → 像素。相机背后返回 None。"""
    C = _v(cam["C"])
    fwd, right, down = _v(cam["fwd"]), _v(cam["right"]), _v(cam["down"])
    f, cx, cy = float(cam["f"]), float(cam["cx"]), float(cam["cy"])
    v = _v(p) - C
    zc = float(v @ fwd)
    if zc < MIN_DEPTH:
        return None
    return [cx + f * float(v @ right) / zc, cy + f * float(v @ down) / zc]


def pixel_ray(u: float, v: float, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """像素 → 世界系射线 (原点, 单位方向)。"""
    C = _v(cam["C"])
    fwd, right, down = _v(cam["fwd"]), _v(cam["right"]), _v(cam["down"])
    f, cx, cy = float(cam["f"]), float(cam["cx"]), float(cam["cy"])
    d = ((u - cx) / f) * right + ((v - cy) / f) * down + fwd
    n = float(np.linalg.norm(d))
    return C, d / max(n, EPS)


def wall_plane(corners: Any) -> tuple[np.ndarray, np.ndarray]:
    """墙四角 ①②③④ → 平面 (点, 单位法向)。"""
    c = np.asarray(corners, dtype=float)
    n = np.cross(c[1] - c[0], c[3] - c[0])
    n = n / max(float(np.linalg.norm(n)), EPS)
    return c[0], n


def ray_plane(u: float, v: float, cam: dict, p0: Any, n: Any) -> list[float] | None:
    """像素射线 ∩ 平面。不相交或朝后返回 None。"""
    C, d = pixel_ray(u, v, cam)
    p0, n = _v(p0), _v(n)
    den = float(d @ n)
    if abs(den) < 1e-8:
        return None
    t = float((p0 - C) @ n) / den
    if t < MIN_DEPTH:
        return None
    return (C + t * d).tolist()


def offset_corners(corners: Any, sign: int, inset: float) -> list[list[float]]:
    """墙角沿 X 朝巷道平移 inset 米，得到货格开口所在平面。"""
    dx = -float(sign) * float(inset)
    return [[float(p[0]) + dx, float(p[1]), float(p[2])] for p in corners]


def triangulate_rays(
    c1: Any, d1: Any, c2: Any, d2: Any, max_gap: float = 0.5, attach: str = "mid",
) -> list[float] | None:
    """两射线最近点。attach=first 则落在第一条射线上（拖哪路跟哪路）。"""
    c1, d1, c2, d2 = _v(c1), _v(d1), _v(c2), _v(d2)
    w0 = c1 - c2
    a, b, c = float(d1 @ d1), float(d1 @ d2), float(d2 @ d2)
    d, e = float(d1 @ w0), float(d2 @ w0)
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        return None
    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom
    if t < MIN_DEPTH or s < MIN_DEPTH:
        return None
    p1, p2 = c1 + t * d1, c2 + s * d2
    if float(np.linalg.norm(p1 - p2)) > max_gap:
        return None
    if attach == "first":
        return p1.tolist()
    if attach == "second":
        return p2.tolist()
    return (0.5 * (p1 + p2)).tolist()


def triangulate_pixels(
    u1: float, v1: float, cam1: dict, u2: float, v2: float, cam2: dict,
    attach: str = "mid",
) -> list[float] | None:
    c1, d1 = pixel_ray(u1, v1, cam1)
    c2, d2 = pixel_ray(u2, v2, cam2)
    return triangulate_rays(c1, d1, c2, d2, attach=attach)


def drag_vertex(
    u: float, v: float, cam: dict, other_cam: dict, current: Any, p0: Any, n: Any,
    stereo: bool = False,
) -> list[float] | None:
    """默认射线∩开口平面。stereo=True 时本路跟鼠标、用另一路像素三角化。"""
    if stereo:
        uv_o = project_pix(current, other_cam)
        if uv_o is not None:
            p = triangulate_pixels(u, v, cam, uv_o[0], uv_o[1], other_cam, attach="first")
            if p is not None:
                return p
    return ray_plane(u, v, cam, p0, n)


def make_grid_vertices(corners: Any, rows: int, cols: int) -> list[list[float]]:
    """在墙矩形上均匀切 rows×cols 格，顶点 (rows+1)*(cols+1)，行主序。

    r=0 顶、c=0 左路远端。双线性即矩形插值（墙在 3D 是矩形）。
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows/cols 至少为 1")
    c0, c1, c2, c3 = [_v(p) for p in corners]
    out: list[list[float]] = []
    for r in range(rows + 1):
        ty = r / rows
        for c in range(cols + 1):
            tz = c / cols
            p = (
                (1 - ty) * (1 - tz) * c0
                + (1 - ty) * tz * c1
                + ty * tz * c2
                + ty * (1 - tz) * c3
            )
            out.append([float(x) for x in p])
    return out


def vert_index(rows: int, cols: int, r: int, c: int) -> int:
    return r * (cols + 1) + c


def mesh_cells(mesh: dict) -> list[dict]:
    """网格 → 单元格列表，每格四角仍是 ①顶远 ②顶近 ③底近 ④底远。"""
    rows, cols = int(mesh["rows"]), int(mesh["cols"])
    verts = mesh["vertices"]
    cells = []
    for i in range(rows):
        for j in range(cols):
            corners = [
                verts[vert_index(rows, cols, i, j)],
                verts[vert_index(rows, cols, i, j + 1)],
                verts[vert_index(rows, cols, i + 1, j + 1)],
                verts[vert_index(rows, cols, i + 1, j)],
            ]
            cells.append({
                "row": i,
                "col": j,
                "box_id": f"r{i}c{j}",
                "corners": corners,
            })
    return cells


def wall_by_id(solved: dict, wall_id: int) -> dict | None:
    for w in solved.get("walls") or []:
        if w.get("wall_id") == wall_id:
            return w
    return None


def _point_in_poly(pt: Any, poly: Any) -> bool:
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def signed_wall_dist(p: Any, wall: dict) -> float:
    """巷道内侧为正：人在通道里 >0，贴墙/伸进 ≤0。"""
    p0 = _v(wall["corners"][0])
    inward = np.array([1.0 if int(wall.get("sign", -1)) < 0 else -1.0, 0.0, 0.0])
    return float((_v(p) - p0) @ inward)


def contact_slots(
    p: Any, meshes: list[dict], solved: dict, contact_m: float = 0.10,
) -> list[dict]:
    """腕点接触墙面且落在货格 (Y,Z) 多边形内 → 该货位报警。"""
    if p is None:
        return []
    hits = []
    for mesh in meshes or []:
        wall = wall_by_id(solved, mesh.get("wall_id"))
        if not wall:
            continue
        d = signed_wall_dist(p, wall)
        if d > contact_m:
            continue
        yz = [_v(p)[1], _v(p)[2]]
        for cell in mesh_cells(mesh):
            poly = [[c[1], c[2]] for c in cell["corners"]]
            if _point_in_poly(yz, poly):
                hits.append({
                    "wall_id": mesh["wall_id"],
                    "box_id": cell["box_id"],
                    "row": cell["row"],
                    "col": cell["col"],
                    "d": round(d, 3),
                })
    return hits
