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


def scale_keypoints_to_calib(
    persons: list | None,
    infer_w: int,
    infer_h: int,
    calib_w: int,
    calib_h: int,
) -> tuple[list, dict[str, Any]]:
    """把推理像素映射到标定像素。

    - 两边尺寸一致：原样（identity）
    - 两边都有效且不一致：按比例 sx=calib_w/infer_w（ratio），这是主路径
    - 推理尺寸缺失且坐标像 0–1：按标定宽高展开（normalize，仅保底）
    """
    src = list(persons or [])
    cw = max(1.0, float(calib_w))
    ch = max(1.0, float(calib_h))
    iw = int(infer_w or 0)
    ih = int(infer_h or 0)

    def _apply(sx: float, sy: float) -> list:
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
                x = float(kp[0]) * sx
                y = float(kp[1]) * sy
                extra = [float(v) for v in kp[2:]]
                kpts.append([x, y, *extra])
            cloned = dict(person)
            cloned["keypoints"] = kpts
            out.append(cloned)
        return out

    meta = {"infer": [iw, ih], "calib": [int(cw), int(ch)], "sx": 1.0, "sy": 1.0, "mode": "identity"}
    if iw > 0 and ih > 0:
        if iw == int(cw) and ih == int(ch):
            return src, meta
        sx, sy = cw / float(iw), ch / float(ih)
        meta.update({"mode": "ratio", "sx": round(sx, 6), "sy": round(sy, 6)})
        return _apply(sx, sy), meta

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
