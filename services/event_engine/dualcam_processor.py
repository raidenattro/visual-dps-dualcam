"""双路 Pose 对齐后在 3D 做贴墙碰撞（contact_slots），贴墙即报。"""

from __future__ import annotations

from typing import Any

import numpy as np

from dualcam.geom import DEFAULT_CONTACT_M, contact_slots
from dualcam.lift import (
    CONTACT_SRC,
    LWRIST,
    RWRIST,
    keypoints_to_ks,
    lift_point,
    pick_pairs,
    wall_plane_from_solved,
)
from services.dualcam_config import (
    aabb_from_section,
    calib_size_from_view,
    get_dualcam_section,
    scale_keypoints_to_calib,
)
from services.dualcam_overlay import collision_token


def _wrist_uv_score(k: np.ndarray, s: np.ndarray, idx: int) -> tuple[np.ndarray, float]:
    return k[idx], float(s[idx])


def _scale_pose(pose: dict, calib_w: int, calib_h: int) -> tuple[dict, dict]:
    scaled, meta = scale_keypoints_to_calib(
        pose.get("persons") or [],
        int(pose.get("infer_width") or 0),
        int(pose.get("infer_height") or 0),
        calib_w,
        calib_h,
    )
    out = dict(pose)
    out["persons"] = scaled
    return out, meta


class DualcamProcessor:
    """同一巷道 L/R 两帧 → collisions / alarm_collisions。"""

    def __init__(self, aisle: dict[str, Any], dualcam_section: dict | None = None):
        self.aisle = aisle
        self.aisle_id = str(aisle.get("aisle_id") or "")
        self.section = dualcam_section or get_dualcam_section()
        solved = aisle.get("solved") or {}
        self.solved = solved if solved.get("ok") else {}
        self.cams = (self.solved.get("cameras") or {}) if self.solved else {}
        self.meshes = list(aisle.get("slot_meshes") or [])
        self.contact_m = float(
            aisle.get("contact_m", self.section.get("contact_m", DEFAULT_CONTACT_M)) or 0.0
        )
        self.plane = wall_plane_from_solved(self.solved, 1) if self.solved else None
        self._prev_wrists: dict[tuple[int, int], np.ndarray] = {}
        self._prefer: list[tuple[np.ndarray, np.ndarray]] = []
        cams = aisle.get("cameras") or {}
        self.cam_l = str((cams.get("L") or {}).get("camera_id") or "")
        self.cam_r = str((cams.get("R") or {}).get("camera_id") or "")
        self.shelf_code = self.aisle_id
        views = aisle.get("views") or {}
        self.calib_l = calib_size_from_view(views.get("L") if isinstance(views, dict) else None, self.section)
        self.calib_r = calib_size_from_view(views.get("R") if isinstance(views, dict) else None, self.section)
        self.aabb = aabb_from_section(self.section, aisle)

    def ready(self) -> bool:
        return bool(
            self.solved.get("ok")
            and "L" in self.cams
            and "R" in self.cams
            and self.meshes
        )

    def not_ready_reason(self) -> str:
        if not self.solved.get("ok"):
            return f"巷道 {self.aisle_id} 尚未反解"
        if "L" not in self.cams or "R" not in self.cams:
            return f"巷道 {self.aisle_id} 反解结果缺少 L/R 相机"
        if not self.meshes:
            return f"巷道 {self.aisle_id} 尚未生成货格层线"
        return ""

    def process_pair(self, pose_l: dict, pose_r: dict) -> dict[str, Any]:
        """两路 PoseFrame → 贴墙 token。无立体则空报警。"""
        frame_idx = int(pose_l.get("frame_idx") or pose_r.get("frame_idx") or 0)
        pose_ls, meta_l = _scale_pose(pose_l, *self.calib_l)
        pose_rs, meta_r = _scale_pose(pose_r, *self.calib_r)
        empty = {
            "frame_idx": frame_idx,
            "collisions": [],
            "alarm_collisions": [],
            "skeletons_l": pose_l.get("persons") or [],
            "skeletons_r": pose_r.get("persons") or [],
            "scale_l": meta_l,
            "scale_r": meta_r,
        }
        if not self.ready():
            return empty

        fl = keypoints_to_ks(pose_ls.get("persons") or [])
        fr = keypoints_to_ks(pose_rs.get("persons") or [])
        pairs = pick_pairs(fl, fr, self.cams, prefer=self._prefer, aabb=self.aabb)
        tokens: list[str] = []
        new_prefer: list[tuple[np.ndarray, np.ndarray]] = []
        new_prev: dict[tuple[int, int], np.ndarray] = {}

        for i, j, _gap in pairs:
            tl = fl["k"][i][[5, 6]].mean(axis=0)
            tr = fr["k"][j][[5, 6]].mean(axis=0)
            new_prefer.append((tl, tr))
            for wi, idx in (("lw", LWRIST), ("rw", RWRIST)):
                uv_l, sl = _wrist_uv_score(fl["k"][i], fl["s"][i], idx)
                uv_r, sr = _wrist_uv_score(fr["k"][j], fr["s"][j], idx)
                prev = self._prev_wrists.get((i, idx))
                p, _g, src = lift_point(uv_l, sl, uv_r, sr, self.cams, self.plane, prev)
                if p is None or src not in CONTACT_SRC:
                    continue
                new_prev[(i, idx)] = p
                hits = contact_slots(p, self.meshes, self.solved, self.contact_m)
                for hit in hits:
                    tok = collision_token(self.aisle_id, hit.get("wall_id"), hit.get("box_id"))
                    if tok and tok not in tokens:
                        tokens.append(tok)

        self._prefer = new_prefer
        self._prev_wrists = new_prev
        # 贴墙即报：collisions 与 alarm_collisions 相同
        return {
            "frame_idx": frame_idx,
            "collisions": list(tokens),
            "alarm_collisions": list(tokens),
            "skeletons_l": pose_l.get("persons") or [],
            "skeletons_r": pose_r.get("persons") or [],
            "scale_l": meta_l,
            "scale_r": meta_r,
        }
