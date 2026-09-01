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
    nms_indices,
    pick_pairs,
    wall_plane_from_solved,
)
from services.aisle_store import required_wall_ids, wall_shelf_code
from services.box_identity import box_collision_token
from services.dualcam_config import (
    aabb_from_section,
    calib_size_from_view,
    get_dualcam_section,
    scale_keypoints_to_calib,
)


def _kpt_uv_score(k: np.ndarray, s: np.ndarray, idx: int) -> tuple[np.ndarray, float]:
    return k[idx], float(s[idx])


# 单路预览不抬五官：贴墙射线会把鼻子/眼睛拉到货架平面，骨线变成「长射线」
FACE_JOINTS = frozenset({0, 1, 2, 3, 4})


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
        self._prev_xyz: dict[tuple, np.ndarray] = {}
        self._new_prev: dict[tuple, np.ndarray] = {}
        self._prefer: list[tuple[np.ndarray, np.ndarray]] = []
        cams = aisle.get("cameras") or {}
        self.cam_l = str((cams.get("L") or {}).get("camera_id") or "")
        self.cam_r = str((cams.get("R") or {}).get("camera_id") or "")
        self.required_walls = required_wall_ids(aisle)
        views = aisle.get("views") or {}
        self.calib_l = calib_size_from_view(views.get("L") if isinstance(views, dict) else None, self.section)
        self.calib_r = calib_size_from_view(views.get("R") if isinstance(views, dict) else None, self.section)
        self.aabb = aabb_from_section(self.section, aisle)

    def _lift_joints(
        self,
        kl: np.ndarray,
        sl: np.ndarray,
        kr: np.ndarray | None,
        sr: np.ndarray | None,
        prev_key: tuple,
    ) -> tuple[list, list, dict[int, list[str]]]:
        """抬 17 关节。对侧缺失时走单路（只显示，不报贴墙）。"""
        xyz: list[list[float] | None] = [None] * 17
        srcs: list[str | None] = [None] * 17
        wrist_tokens: dict[int, list[str]] = {LWRIST: [], RWRIST: []}
        has_r = kr is not None and sr is not None
        for ji in range(17):
            uv_l, sc_l = _kpt_uv_score(kl, sl, ji)
            if has_r:
                uv_r, sc_r = _kpt_uv_score(kr, sr, ji)
            else:
                uv_r, sc_r = uv_l, 0.0
            # 单路（对侧分数为 0）不抬五官，避免贴墙射线拉成「射向货架」的长骨线
            if ji in FACE_JOINTS and (not has_r or sc_l <= 0.0 or sc_r <= 0.0):
                continue
            prev = self._prev_xyz.get((*prev_key, ji))
            p, _g, src = lift_point(uv_l, sc_l, uv_r, sc_r, self.cams, self.plane, prev)
            srcs[ji] = src
            if p is None:
                continue
            xyz[ji] = [float(p[0]), float(p[1]), float(p[2])]
            self._new_prev[(*prev_key, ji)] = p
            if ji not in (LWRIST, RWRIST) or src not in CONTACT_SRC:
                continue
            hits = contact_slots(p, self.meshes, self.solved, self.contact_m)
            toks: list[str] = []
            for hit in hits:
                shelf = str(hit.get("shelf_code") or "").strip() or wall_shelf_code(
                    self.aisle, int(hit.get("wall_id") or 0)
                )
                box_id = str(hit.get("box_id") or "").strip()
                tok = box_collision_token({"shelf_code": shelf, "box_id": box_id})
                if tok and ":" in tok:
                    toks.append(tok)
            wrist_tokens[ji] = toks
        return xyz, srcs, wrist_tokens

    def process_single(self, role: str, pose: dict) -> dict[str, Any]:
        """对侧未到时的 3D 预览：单路抬到墙平面，不报警。"""
        role = str(role or "").strip().upper() or "L"
        calib = self.calib_l if role == "L" else self.calib_r
        scaled, meta = _scale_pose(pose, *calib)
        frame_idx = int(pose.get("frame_idx") or 0)
        out = {
            "frame_idx": frame_idx,
            "collisions": [],
            "alarm_collisions": [],
            "skeletons_l": pose.get("persons") or [] if role == "L" else [],
            "skeletons_r": pose.get("persons") or [] if role == "R" else [],
            "persons_3d": [],
            "preview": True,
            "scale_l": meta if role == "L" else {},
            "scale_r": meta if role == "R" else {},
        }
        if not self.solved.get("ok") or role not in self.cams:
            return out
        ks = keypoints_to_ks(scaled.get("persons") or [])
        self._new_prev = {}
        people = []
        for i in nms_indices(ks["k"], ks["s"]):
            if role == "L":
                xyz, srcs, _hits = self._lift_joints(ks["k"][i], ks["s"][i], None, None, ("L", i))
            else:
                # 对侧分数为 0，走 R 单路
                xyz, srcs, _hits = self._lift_joints(
                    ks["k"][i], np.zeros(17, np.float32), ks["k"][i], ks["s"][i], ("R", i)
                )
            people.append({"xyz": xyz, "src": srcs, "preview": True, "wrist_alarm": {9: False, 10: False}})
        self._prev_xyz = {**self._prev_xyz, **self._new_prev}
        out["persons_3d"] = people
        return out

    def ready(self) -> bool:
        return not bool(self.not_ready_reason())

    def not_ready_reason(self) -> str:
        if not self.solved.get("ok"):
            return f"巷道 {self.aisle_id} 尚未反解"
        if "L" not in self.cams or "R" not in self.cams:
            return f"巷道 {self.aisle_id} 反解结果缺少 L/R 相机"
        by_wall: dict[int, dict] = {}
        for mesh in self.meshes:
            if not isinstance(mesh, dict):
                continue
            try:
                wid = int(mesh.get("wall_id"))
            except (TypeError, ValueError):
                continue
            by_wall[wid] = mesh
        missing = [w for w in self.required_walls if w not in by_wall]
        if missing:
            walls = "、".join(f"墙{w}" for w in missing)
            return f"巷道 {self.aisle_id} 拣货墙缺少层线：{walls}"
        no_shelf = []
        for wid in self.required_walls:
            mesh = by_wall.get(wid) or {}
            code = str(mesh.get("shelf_code") or "").strip() or wall_shelf_code(self.aisle, wid)
            if not code:
                no_shelf.append(wid)
        if no_shelf:
            walls = "、".join(f"墙{w}" for w in no_shelf)
            return f"巷道 {self.aisle_id} 的{walls}未填写货架号"
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
            "persons_3d": [],
            "scale_l": meta_l,
            "scale_r": meta_r,
        }
        if not self.ready():
            return empty

        fl = keypoints_to_ks(pose_ls.get("persons") or [])
        fr = keypoints_to_ks(pose_rs.get("persons") or [])
        pairs = pick_pairs(fl, fr, self.cams, prefer=self._prefer, aabb=self.aabb)
        if not pairs:
            people = []
            people.extend(self.process_single("L", pose_l).get("persons_3d") or [])
            people.extend(self.process_single("R", pose_r).get("persons_3d") or [])
            empty["persons_3d"] = people
            empty["preview"] = True
            return empty
        tokens: list[str] = []
        new_prefer: list[tuple[np.ndarray, np.ndarray]] = []
        persons_3d: list[dict[str, Any]] = []
        self._new_prev = {}

        for i, j, _gap in pairs:
            tl = fl["k"][i][[5, 6]].mean(axis=0)
            tr = fr["k"][j][[5, 6]].mean(axis=0)
            new_prefer.append((tl, tr))
            xyz, srcs, wrist_tok = self._lift_joints(
                fl["k"][i], fl["s"][i], fr["k"][j], fr["s"][j], ("P", i, j)
            )
            alarm9 = bool(wrist_tok.get(LWRIST))
            alarm10 = bool(wrist_tok.get(RWRIST))
            for tok in (wrist_tok.get(LWRIST) or []) + (wrist_tok.get(RWRIST) or []):
                if tok not in tokens:
                    tokens.append(tok)
            persons_3d.append({
                "xyz": xyz,
                "src": srcs,
                "preview": False,
                "wrist_alarm": {9: alarm9, 10: alarm10},
            })

        self._prefer = new_prefer
        self._prev_xyz = self._new_prev
        return {
            "frame_idx": frame_idx,
            "collisions": list(tokens),
            "alarm_collisions": list(tokens),
            "skeletons_l": pose_l.get("persons") or [],
            "skeletons_r": pose_r.get("persons") or [],
            "persons_3d": persons_3d,
            "preview": False,
            "scale_l": meta_l,
            "scale_r": meta_r,
        }
