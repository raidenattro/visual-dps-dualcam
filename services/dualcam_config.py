"""双路 3D 全局约束：标定分辨率、巷道 AABB、相机先验、配对窗。

优先级：巷道 JSON 覆盖 → runtime_config → app_config → 本文件默认值。
标定像素坐标系由 calib_width/height 约束，禁止在代码里写死 1280×720。
"""

from __future__ import annotations

from typing import Any

DEFAULT_DUALCAM: dict[str, Any] = {
    "calib_width": 1280,
    "calib_height": 720,
    "aabb_x_min": -1.35,
    "aabb_x_max": 1.35,
    "aabb_y_min": 0.50,
    "aabb_y_max": 1.65,
    "aabb_z_min": -0.12,
    "aabb_z_max": 2.50,
    "cam_h": 2.84,
    "cam_dist": 1.56,
    "pitch": 45.0,
    "yaw": 0.0,
    "contact_m": 0.0,
    # 配对窗 = pose 周期 × 该系数（pose 周期 = pose_frame_interval / frame_rate）
    "pair_window_periods": 1.5,
    "pair_window_min_sec": 0.08,
    "pair_window_max_sec": 1.0,
}

_AABB_KEYS = ("x", "y", "z")
_AABB_FLAT = {
    "x": ("aabb_x_min", "aabb_x_max"),
    "y": ("aabb_y_min", "aabb_y_max"),
    "z": ("aabb_z_min", "aabb_z_max"),
}


def _as_float(raw: Any, fallback: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(fallback)


def _as_int(raw: Any, fallback: int, *, min_v: int = 1) -> int:
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return int(fallback)
    return val if val >= min_v else int(fallback)


def _merge_section(base: dict, overlay: dict | None) -> dict:
    out = dict(base)
    if not isinstance(overlay, dict):
        return out
    for key, val in overlay.items():
        if val is None:
            continue
        out[key] = val
    return out


def get_dualcam_section(app_config: dict | None = None) -> dict[str, Any]:
    """合并 app_config.dualcam 与 runtime_config.dualcam。"""
    from services.runtime_config_service import DEFAULT_PATH, _load_json

    out = dict(DEFAULT_DUALCAM)
    app_sec = (app_config or {}).get("dualcam")
    out = _merge_section(out, app_sec if isinstance(app_sec, dict) else None)
    overlay = _load_json(DEFAULT_PATH)
    out = _merge_section(out, overlay.get("dualcam") if isinstance(overlay, dict) else None)
    out["calib_width"] = _as_int(out.get("calib_width"), DEFAULT_DUALCAM["calib_width"])
    out["calib_height"] = _as_int(out.get("calib_height"), DEFAULT_DUALCAM["calib_height"])
    for key in (
        "aabb_x_min",
        "aabb_x_max",
        "aabb_y_min",
        "aabb_y_max",
        "aabb_z_min",
        "aabb_z_max",
        "cam_h",
        "cam_dist",
        "pitch",
        "yaw",
        "contact_m",
        "pair_window_periods",
        "pair_window_min_sec",
        "pair_window_max_sec",
    ):
        out[key] = _as_float(out.get(key), DEFAULT_DUALCAM[key])
    return out


def aabb_from_section(section: dict | None = None, aisle: dict | None = None) -> dict[str, tuple[float, float]]:
    """巷道配对用的轴对齐包围盒。巷道 JSON 的 aabb 优先。"""
    sec = section or get_dualcam_section()
    raw = (aisle or {}).get("aabb") if isinstance(aisle, dict) else None
    out: dict[str, tuple[float, float]] = {}
    for axis, (lo_k, hi_k) in _AABB_FLAT.items():
        lo, hi = sec[lo_k], sec[hi_k]
        if isinstance(raw, dict) and axis in raw:
            pair = raw[axis]
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                lo, hi = _as_float(pair[0], lo), _as_float(pair[1], hi)
        if lo > hi:
            lo, hi = hi, lo
        out[axis] = (lo, hi)
    return out


def calib_size_from_view(view: dict | None, section: dict | None = None) -> tuple[int, int]:
    sec = section or get_dualcam_section()
    w, h = int(sec["calib_width"]), int(sec["calib_height"])
    size = (view or {}).get("image_size") if isinstance(view, dict) else None
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        w = _as_int(size[0], w)
        h = _as_int(size[1], h)
    return w, h


def default_prior(section: dict | None = None) -> dict[str, float]:
    sec = section or get_dualcam_section()
    return {
        "camH": float(sec["cam_h"]),
        "camDist": float(sec["cam_dist"]),
        "pitch": float(sec["pitch"]),
        "yaw": float(sec["yaw"]),
    }


def pair_window_sec(
    app_config: dict | None = None,
    *,
    frame_rate: float | None = None,
    pose_frame_interval: int | None = None,
) -> float:
    """按姿态采样周期动态计算 L/R 配对窗，不再写死 0.12s。"""
    from services.runtime_config_service import get_merged_inference_section

    sec = get_dualcam_section(app_config)
    infer = get_merged_inference_section(app_config) if (frame_rate is None or pose_frame_interval is None) else {}
    fps = _as_float(frame_rate if frame_rate is not None else infer.get("frame_rate"), 15.0)
    interval = _as_int(
        pose_frame_interval if pose_frame_interval is not None else infer.get("pose_frame_interval"),
        1,
    )
    fps = max(1.0, fps)
    period = float(interval) / fps
    raw = period * max(0.5, float(sec["pair_window_periods"]))
    lo = max(0.02, float(sec["pair_window_min_sec"]))
    hi = max(lo, float(sec["pair_window_max_sec"]))
    return min(hi, max(lo, raw))


def letterbox_params(
    src_w: int | float,
    src_h: int | float,
    dst_w: int | float,
    dst_h: int | float,
) -> tuple[float, float, float]:
    """把 src 画布均匀缩放到刚好放进 dst，不足的边居中留黑边。

    返回 (scale, pad_x, pad_y)，映射：x' = x * scale + pad_x。
    dst 更宽时左右 pillarbox；dst 更高时上下 letterbox。
    """
    sw = max(1.0, float(src_w))
    sh = max(1.0, float(src_h))
    dw = max(1.0, float(dst_w))
    dh = max(1.0, float(dst_h))
    scale = min(dw / sw, dh / sh)
    pad_x = (dw - sw * scale) / 2.0
    pad_y = (dh - sh * scale) / 2.0
    return scale, pad_x, pad_y


def letterbox_unmap_point(
    x: float,
    y: float,
    src_w: int | float,
    src_h: int | float,
    dst_w: int | float,
    dst_h: int | float,
) -> tuple[float, float]:
    """letterbox_params 的逆映射：dst 画布坐标 → src 像素。"""
    scale, pad_x, pad_y = letterbox_params(src_w, src_h, dst_w, dst_h)
    if scale <= 1e-12:
        return float(x), float(y)
    return (float(x) - pad_x) / scale, (float(y) - pad_y) / scale


def _remap_xy_list(points: list | None, old_w: int, old_h: int, new_w: int, new_h: int) -> list:
    out = []
    for pt in points or []:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            out.append(pt)
            continue
        x, y = letterbox_unmap_point(pt[0], pt[1], new_w, new_h, old_w, old_h)
        extra = list(pt[2:])
        out.append([x, y, *extra])
    return out


def remap_view_image_size(view: dict | None, new_w: int, new_h: int) -> bool:
    """把 view.image_size 写成抽帧真实宽高，并把已标四角从旧画布 contain 回真实像素。

    标注页曾把画面 contain 进 image_size（默认 1280×720 / 16:9）再记点。
    抽帧后 image_size 必须等于静止图像素，否则 4:3 源会把黑边算进墙角。
    尺寸变化时返回 True（调用方应作废已反解相机）。
    """
    if not isinstance(view, dict):
        return False
    nw = _as_int(new_w, 0)
    nh = _as_int(new_h, 0)
    if nw < 2 or nh < 2:
        return False
    old_w, old_h = calib_size_from_view(view)
    if old_w == nw and old_h == nh:
        view["image_size"] = [nw, nh]
        return False
    walls = view.get("walls")
    if isinstance(walls, list):
        remapped = []
        for wall in walls:
            if not isinstance(wall, dict):
                remapped.append(wall)
                continue
            cloned = dict(wall)
            cloned["quad"] = _remap_xy_list(wall.get("quad"), old_w, old_h, nw, nh)
            remapped.append(cloned)
        view["walls"] = remapped
    lines = view.get("layer_lines")
    if isinstance(lines, list):
        new_lines = []
        for line in lines:
            if not isinstance(line, dict):
                new_lines.append(line)
                continue
            cloned = dict(line)
            cloned["uv"] = _remap_xy_list(line.get("uv"), old_w, old_h, nw, nh)
            new_lines.append(cloned)
        view["layer_lines"] = new_lines
    view["image_size"] = [nw, nh]
    return True


def scale_keypoints_to_calib(
    persons: list | None,
    infer_w: int,
    infer_h: int,
    calib_w: int,
    calib_h: int,
) -> tuple[list, dict[str, Any]]:
    """把推理像素映射到标定像素（针孔内参所在坐标系）。

    推理按源图等比缩放（只改 height，宽随比例），纵横比不变。标定 (f, cx, cy)
    定义在 calib 像素系。映射必须是相似变换（均匀缩放 + 平移）：

    - 各轴独立 sx、sy 会改变 fx/fy 比，等价于换了一台相机，三角化射线会偏。
    - letterbox/pillarbox 只做均匀缩放并居中垫边，像素仍是正方形，射线方向与标定一致。
    - 两边尺寸一致：原样（identity）
    - 推理尺寸缺失且坐标像 0–1：按标定宽高展开（normalize，仅保底）
    """
    src = list(persons or [])
    cw = max(1.0, float(calib_w))
    ch = max(1.0, float(calib_h))
    iw = int(infer_w or 0)
    ih = int(infer_h or 0)

    def _apply(sx: float, sy: float, ox: float = 0.0, oy: float = 0.0) -> list:
        out = []
        for person in src:
            if not isinstance(person, dict):
                out.append(person)
                continue
            kpts = []
            for kp in person.get("keypoints") or []:
                if not isinstance(kp, (list, tuple)) or len(kp) < 2:
                    kpts.append(kp)
                    continue
                x = float(kp[0]) * sx + ox
                y = float(kp[1]) * sy + oy
                extra = [float(v) for v in kp[2:]]
                kpts.append([x, y, *extra])
            cloned = dict(person)
            cloned["keypoints"] = kpts
            out.append(cloned)
        return out

    meta = {
        "infer": [iw, ih],
        "calib": [int(cw), int(ch)],
        "sx": 1.0,
        "sy": 1.0,
        "pad_x": 0.0,
        "pad_y": 0.0,
        "mode": "identity",
    }
    if iw > 0 and ih > 0:
        if iw == int(cw) and ih == int(ch):
            return src, meta
        scale, pad_x, pad_y = letterbox_params(iw, ih, cw, ch)
        meta.update({
            "mode": "letterbox",
            "sx": round(scale, 6),
            "sy": round(scale, 6),
            "pad_x": round(pad_x, 4),
            "pad_y": round(pad_y, 4),
        })
        return _apply(scale, scale, pad_x, pad_y), meta

    max_x = 0.0
    max_y = 0.0
    n = 0
    for person in src:
        if not isinstance(person, dict):
            continue
        for kp in person.get("keypoints") or []:
            if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                max_x = max(max_x, abs(float(kp[0])))
                max_y = max(max_y, abs(float(kp[1])))
                n += 1
    if n and max_x <= 1.5 and max_y <= 1.5:
        meta.update({"mode": "normalize", "sx": cw, "sy": ch})
        return _apply(cw, ch), meta
    meta["mode"] = "unknown"
    return src, meta
