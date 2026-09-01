"""双路 Pose 对齐后在 3D 做贴墙碰撞（contact_slots），贴墙即报。"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from dualcam.geom import DEFAULT_CONTACT_M, contact_slots
from dualcam.lift import (
    CONTACT_SRC,
    LWRIST,
    PREFER_PX,
    RWRIST,
    _torso_xy,
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
from dualcam.skel3d_smooth import LivePose2DSmoother, LivePose3DSmoother, pose_time


def _kpt_uv_score(k: np.ndarray, s: np.ndarray, idx: int) -> tuple[np.ndarray, float]:
    return k[idx], float(s[idx])


# 单路预览不抬五官：贴墙射线会把鼻子/眼睛拉到货架平面，骨线变成「长射线」
FACE_JOINTS = frozenset({0, 1, 2, 3, 4})
# 与 pick-state dump_skel3d 一致：单路/检测闪断沿用上一帧 3D，满 8 帧再丢
HOLD_FRAMES = 8
HOLD_MATCH_M = 0.55
TORSO_JOINTS = (5, 6, 11, 12)
LSHO, RSHO = 5, 6
# 超过则腕点几何不可信，不报贴墙（dump_skel3d.SHOULDER_WRIST_MAX）
SHOULDER_WRIST_MAX = 0.85


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


def _subset_persons(pose: dict | None, indices: list[int]) -> list[dict[str, Any]]:
    people = list((pose or {}).get("persons") or [])
    out: list[dict[str, Any]] = []
    for i in indices:
        if 0 <= i < len(people) and isinstance(people[i], dict):
            out.append(people[i])
    return out


def _prefer_xy(k: np.ndarray, s: np.ndarray) -> np.ndarray:
    """与 pick_pairs / _match_torso_idx 同一套躯干中心，避免肩点对不上髋点。"""
    t = _torso_xy(k, s)
    if t is not None:
        return t
    return np.asarray(k[[5, 6]].mean(axis=0), float)


def _nms_idx(pose: dict | None) -> list[int]:
    if not pose:
        return []
    pack = keypoints_to_ks(pose.get("persons") or [])
    return nms_indices(pack["k"], pack["s"])


def _torso_centroid(xyz: list) -> np.ndarray | None:
    pts = []
    for i in TORSO_JOINTS:
        if xyz and i < len(xyz) and xyz[i] and len(xyz[i]) >= 3:
            pts.append(np.asarray(xyz[i][:3], float))
    if len(pts) < 2:
        pts = [np.asarray(p[:3], float) for p in (xyz or []) if p and len(p) >= 3]
    if not pts:
        return None
    return np.mean(pts, axis=0)


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
        self._holds: list[tuple[dict, int, np.ndarray]] = []
        self._last_skel_l: list[dict[str, Any]] = []
        self._last_skel_r: list[dict[str, Any]] = []
        self._sm2d = {"L": LivePose2DSmoother(), "R": LivePose2DSmoother()}
        self._sm3d = LivePose3DSmoother()
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
        prev_xyz: list | None = None,
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
            prev = None
            if prev_xyz is not None and ji < len(prev_xyz) and prev_xyz[ji]:
                prev = np.asarray(prev_xyz[ji][:3], float)
            if prev is None:
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
        self._clamp_flying_wrists(xyz, srcs, wrist_tokens)
        return xyz, srcs, wrist_tokens

    def _clamp_flying_wrists(
        self,
        xyz: list,
        srcs: list,
        wrist_tokens: dict[int, list[str]],
    ) -> None:
        """肩-腕过长则收到上限上，不删点（删了会闪断、骨线抽搐）。"""
        for wi, shi in ((LWRIST, LSHO), (RWRIST, RSHO)):
            w, sh = xyz[wi] if wi < len(xyz) else None, xyz[shi] if shi < len(xyz) else None
            if not w or not sh:
                continue
            wv = np.asarray(w[:3], float)
            sv = np.asarray(sh[:3], float)
            d = float(np.linalg.norm(wv - sv))
            if d <= SHOULDER_WRIST_MAX or d < 1e-6:
                continue
            xyz[wi] = (sv + (wv - sv) * (SHOULDER_WRIST_MAX / d)).tolist()
            wrist_tokens[wi] = []

    def _match_torso_idx(self, pack: dict, xy, used: set[int]) -> int | None:
        if xy is None:
            return None
        best, best_d = None, float(PREFER_PX)
        ks, ss = pack.get("k"), pack.get("s")
        n = 0 if ks is None else len(ks)
        if n == 0 or ss is None:
            return None
        for i in range(n):
            if i in used:
                continue
            p = _torso_xy(ks[i], ss[i])
            if p is None:
                continue
            d = float(np.linalg.norm(p - xy))
            if d < best_d:
                best, best_d = i, d
        return best

    def _overlay_follow(
        self,
        pose_l: dict | None,
        pose_r: dict | None,
        idx_l: list[int],
        idx_r: list[int],
        *,
        have_l: bool,
        have_r: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        # 配上了用配对下标；配不上用当前 NMS 人。空 idx 不再清成 []，否则画面冻在上一帧或空白。
        if have_l and pose_l is not None:
            idx = list(idx_l) if idx_l else _nms_idx(pose_l)
            self._last_skel_l = _subset_persons(pose_l, idx)
        if have_r and pose_r is not None:
            idx = list(idx_r) if idx_r else _nms_idx(pose_r)
            self._last_skel_r = _subset_persons(pose_r, idx)
        return list(self._last_skel_l), list(self._last_skel_r)

    def _hold_xyz_near(self, idx: int) -> list | None:
        if 0 <= idx < len(self._holds):
            return (self._holds[idx][0] or {}).get("xyz")
        if self._holds:
            return (self._holds[0][0] or {}).get("xyz")
        return None

    def _lift_follow(self, fl: dict, fr: dict) -> tuple[list[dict[str, Any]], list[int], list[int]]:
        """配不上立体时：当前 2D + 上一帧深度沿射线跟上。"""
        prefer = list(self._prefer)
        if not prefer and self._holds:
            # 无续帧标定点时，用当前 NMS 人沿 hold 深度跟上，不要把 3D 冻死
            k_l, k_r = fl.get("k"), fr.get("k")
            li = nms_indices(fl["k"], fl["s"]) if k_l is not None and len(k_l) else []
            ri = nms_indices(fr["k"], fr["s"]) if k_r is not None and len(k_r) else []
            i = li[0] if li else None
            j = ri[0] if ri else None
            if i is None and j is None:
                return [], [], []
            people: list[dict[str, Any]] = []
            idx_l: list[int] = []
            idx_r: list[int] = []
            self._new_prev = dict(self._prev_xyz)
            prev_xyz = self._hold_xyz_near(0)
            if i is not None and j is not None:
                xyz, srcs, _tok = self._lift_joints(
                    fl["k"][i], fl["s"][i], fr["k"][j], fr["s"][j], ("F", 0), prev_xyz=prev_xyz,
                )
                idx_l.append(i)
                idx_r.append(j)
            elif i is not None:
                xyz, srcs, _tok = self._lift_joints(
                    fl["k"][i], fl["s"][i], None, None, ("F", 0), prev_xyz=prev_xyz,
                )
                idx_l.append(i)
            else:
                z = np.zeros_like(fr["k"][j])
                zs = np.zeros_like(fr["s"][j])
                xyz, srcs, _tok = self._lift_joints(
                    z, zs, fr["k"][j], fr["s"][j], ("F", 0), prev_xyz=prev_xyz,
                )
                idx_r.append(j)
            people.append({
                "xyz": xyz,
                "src": srcs,
                "preview": True,
                "wrist_alarm": {9: False, 10: False},
            })
            return people, idx_l, idx_r
        if not prefer:
            return [], [], []
        people: list[dict[str, Any]] = []
        idx_l: list[int] = []
        idx_r: list[int] = []
        used_l: set[int] = set()
        used_r: set[int] = set()
        self._new_prev = dict(self._prev_xyz)
        for pi, (pl, pr) in enumerate(prefer):
            i = self._match_torso_idx(fl, pl, used_l)
            j = self._match_torso_idx(fr, pr, used_r)
            if i is not None:
                used_l.add(i)
                idx_l.append(i)
            if j is not None:
                used_r.add(j)
                idx_r.append(j)
            if i is None and j is None:
                continue
            prev_xyz = self._hold_xyz_near(pi)
            if i is not None and j is not None:
                xyz, srcs, _tok = self._lift_joints(
                    fl["k"][i], fl["s"][i], fr["k"][j], fr["s"][j], ("F", pi), prev_xyz=prev_xyz,
                )
            elif i is not None:
                xyz, srcs, _tok = self._lift_joints(
                    fl["k"][i], fl["s"][i], None, None, ("F", pi), prev_xyz=prev_xyz,
                )
            else:
                z = np.zeros_like(fr["k"][j])
                zs = np.zeros_like(fr["s"][j])
                xyz, srcs, _tok = self._lift_joints(
                    z, zs, fr["k"][j], fr["s"][j], ("F", pi), prev_xyz=prev_xyz,
                )
            people.append({
                "xyz": xyz,
                "src": srcs,
                "preview": True,
                "wrist_alarm": {9: False, 10: False},
            })
        return people, idx_l, idx_r

    def _clear_hold(self) -> None:
        self._holds = []
        self._prev_xyz = {}
        self._prefer = []
        self._last_skel_l = []
        self._last_skel_r = []
        self._sm2d["L"].reset()
        self._sm2d["R"].reset()
        self._sm3d.reset()

    def _apply_hold(self, people: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """续帧匹配：对不上时沿用上一帧 3D，最多 HOLD_FRAMES 帧（与 dump_skel3d 相同）。"""
        out = list(people or [])
        cents = [_torso_centroid(p.get("xyz") or []) for p in out]
        used: set[int] = set()
        new_holds: list[tuple[dict, int, np.ndarray]] = []
        for prev, miss, pt in self._holds:
            best, best_d = None, HOLD_MATCH_M
            for ci, c in enumerate(cents):
                if ci in used or c is None:
                    continue
                d = float(np.linalg.norm(pt - c))
                if d < best_d:
                    best, best_d = ci, d
            if best is not None:
                used.add(best)
                new_holds.append((out[best], 0, cents[best]))
            elif (
                len(out) == 1
                and len(cents) == 1
                and 0 not in used
                and cents[0] is not None
            ):
                # 单人深度跟上后躯干可能跳过 HOLD_MATCH_M，仍视为同一人，避免冻帧+复制
                used.add(0)
                new_holds.append((out[0], 0, cents[0]))
            elif miss + 1 <= HOLD_FRAMES:
                held = copy.deepcopy(prev)
                held["held"] = True
                held["preview"] = True
                held["wrist_alarm"] = {9: False, 10: False}
                out.append(held)
                new_holds.append((held, miss + 1, pt))
        for ci, p in enumerate(out):
            if ci in used or p.get("held"):
                continue
            c = cents[ci] if ci < len(cents) else _torso_centroid(p.get("xyz") or [])
            if c is None:
                continue
            new_holds.append((p, 0, c))
        self._holds = new_holds
        if not new_holds:
            self._prev_xyz = {}
            self._prefer = []
        return out

    def _smooth_pose(self, role: str, pose: dict) -> dict:
        """推理像素上做因果 2D 短窗，分数不动。"""
        t = pose_time(pose, int(pose.get("frame_idx") or 0))
        out = dict(pose)
        out["persons"] = self._sm2d[role].update(t, pose.get("persons") or [])
        return out

    def process_single(
        self,
        role: str,
        pose: dict,
        *,
        apply_hold: bool = True,
        smooth_2d: bool = True,
        smooth_3d: bool = True,
    ) -> dict[str, Any]:
        """对侧未到时：2D 跟着当前检测走；3D 用上一帧深度沿射线跟上，不冻帧。"""
        role = str(role or "").strip().upper() or "L"
        if smooth_2d:
            pose = self._smooth_pose(role, pose)
        calib = self.calib_l if role == "L" else self.calib_r
        scaled, meta = _scale_pose(pose, *calib)
        frame_idx = int(pose.get("frame_idx") or 0)
        empty = {"k": [], "s": []}
        pack = keypoints_to_ks(scaled.get("persons") or [])
        fl, fr = (pack, empty) if role == "L" else (empty, pack)
        people: list[dict[str, Any]] = []
        idx_l: list[int] = []
        idx_r: list[int] = []
        if self.solved.get("ok") and role in self.cams:
            people, idx_l, idx_r = self._lift_follow(fl, fr)
            if apply_hold:
                people = self._apply_hold(people)
            if smooth_3d:
                people = self._sm3d.update(pose_time(pose, frame_idx), people, self.plane)
            self._prev_xyz = self._new_prev
        sk_l, sk_r = self._overlay_follow(
            pose if role == "L" else None,
            pose if role == "R" else None,
            idx_l,
            idx_r,
            have_l=(role == "L"),
            have_r=(role == "R"),
        )
        return {
            "frame_idx": frame_idx,
            "collisions": [],
            "alarm_collisions": [],
            "skeletons_l": sk_l,
            "skeletons_r": sk_r,
            "persons_3d": people,
            "preview": True,
            "scale_l": meta if role == "L" else {},
            "scale_r": meta if role == "R" else {},
        }

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
        t = pose_time(pose_l, frame_idx) or pose_time(pose_r, frame_idx)
        pose_l = self._smooth_pose("L", pose_l)
        pose_r = self._smooth_pose("R", pose_r)
        pose_ls, meta_l = _scale_pose(pose_l, *self.calib_l)
        pose_rs, meta_r = _scale_pose(pose_r, *self.calib_r)
        if not self.ready():
            self._clear_hold()
            return {
                "frame_idx": frame_idx,
                "collisions": [],
                "alarm_collisions": [],
                "skeletons_l": [],
                "skeletons_r": [],
                "persons_3d": [],
                "scale_l": meta_l,
                "scale_r": meta_r,
            }

        fl = keypoints_to_ks(pose_ls.get("persons") or [])
        fr = keypoints_to_ks(pose_rs.get("persons") or [])
        pairs = pick_pairs(fl, fr, self.cams, prefer=self._prefer, aabb=self.aabb)
        if not pairs:
            followed, idx_l, idx_r = self._lift_follow(fl, fr)
            held = self._apply_hold(followed)
            held = self._sm3d.update(t, held, self.plane)
            self._prev_xyz = self._new_prev
            sk_l, sk_r = self._overlay_follow(
                pose_l, pose_r, idx_l, idx_r, have_l=True, have_r=True,
            )
            return {
                "frame_idx": frame_idx,
                "collisions": [],
                "alarm_collisions": [],
                "skeletons_l": sk_l,
                "skeletons_r": sk_r,
                "persons_3d": held,
                "preview": True,
                "scale_l": meta_l,
                "scale_r": meta_r,
            }
        tokens: list[str] = []
        new_prefer: list[tuple[np.ndarray, np.ndarray]] = []
        persons_3d: list[dict[str, Any]] = []
        self._new_prev = {}

        for i, j, _gap in pairs:
            new_prefer.append((_prefer_xy(fl["k"][i], fl["s"][i]), _prefer_xy(fr["k"][j], fr["s"][j])))
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
        persons_3d = self._sm3d.update(t, self._apply_hold(persons_3d), self.plane)
        sk_l, sk_r = self._overlay_follow(
            pose_l, pose_r, [i for i, _, _ in pairs], [j for _, j, _ in pairs],
            have_l=True, have_r=True,
        )
        return {
            "frame_idx": frame_idx,
            "collisions": list(tokens),
            "alarm_collisions": list(tokens),
            "skeletons_l": sk_l,
            "skeletons_r": sk_r,
            "persons_3d": persons_3d,
            "preview": False,
            "scale_l": meta_l,
            "scale_r": meta_r,
        }
