"""手腕 vs 货框碰撞检测与连续帧报警门控。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import cv2

from services.box_identity import box_collision_token
from services.event_engine.event_log import PrefilterLogEntry


class CollisionPrefilterProtocol(Protocol):
    def evaluate(self, pose_frame: dict, track_id: int, person: dict): ...

    def should_block(self, pose_frame: dict, track_id: int, person: dict) -> bool: ...


# 与实验仓 dwell_single.py 一致：15fps×0.24s → 4 帧。无四角时比像素。
DEFAULT_DWELL_DELTA_SEC = 0.24
DEFAULT_DWELL_TAU_PX = 20.0
WRIST_SCORE_MIN = 0.3
WRIST_JOINTS = (9, 10)


def lag_frames(fps: float, delta_s: float) -> int:
    """Δ 秒对应的视频帧数。按 frame_idx 换算，不把 6 帧写死。"""
    return max(1, int(round(max(1.0, float(fps)) * float(delta_s))))


@dataclass
class _WristObs:
    frame_idx: int
    x: float
    y: float
    conf: float


@dataclass
class TrackState:
    abs_x: float
    abs_y: float
    ts_sec: float


class PersonTrackAssigner:
    def __init__(self, max_match_dist=220.0, stale_sec=1.2):
        self.max_match_dist = max_match_dist
        self.stale_sec = stale_sec
        self.next_id = 1
        self.tracks: dict[int, TrackState] = {}

    def _cleanup(self, now_ts: float):
        dead_keys = [k for k, st in self.tracks.items() if now_ts - st.ts_sec > self.stale_sec]
        for k in dead_keys:
            self.tracks.pop(k, None)

    def assign(self, abs_x: float, abs_y: float, now_ts: float, occupied_track_ids=None) -> int:
        self._cleanup(now_ts)
        occupied = occupied_track_ids if occupied_track_ids is not None else set()

        best_tid = None
        best_dist = 1e9
        for tid, st in self.tracks.items():
            if tid in occupied:
                continue
            dist = math.hypot(abs_x - st.abs_x, abs_y - st.abs_y)
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        if best_tid is None or best_dist > self.max_match_dist:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = TrackState(abs_x=abs_x, abs_y=abs_y, ts_sec=now_ts)
            occupied.add(tid)
            return tid

        self.tracks[best_tid] = TrackState(abs_x=abs_x, abs_y=abs_y, ts_sec=now_ts)
        occupied.add(best_tid)
        return best_tid


class CollisionProcessor:
    """消费 PoseFrame，输出碰撞 token 与报警 token。"""

    def __init__(
        self,
        boxes: list,
        *,
        alarm_min_consecutive_frames: int = 3,
        alarm_cooldown_frames: int = 12,
        video_fps: float = 25.0,
        dwell_enabled: bool = False,
        dwell_delta_sec: float = DEFAULT_DWELL_DELTA_SEC,
        dwell_tau_px: float = DEFAULT_DWELL_TAU_PX,
        pose_frame_interval: int = 1,
    ):
        self.boxes = boxes
        self.alarm_min_consecutive_frames = max(1, int(alarm_min_consecutive_frames))
        self.alarm_cooldown_frames = max(0, int(alarm_cooldown_frames))
        self.video_fps = max(1.0, float(video_fps))
        self.dwell_enabled = bool(dwell_enabled)
        self.dwell_delta_sec = float(dwell_delta_sec)
        self.dwell_tau_px = float(dwell_tau_px)
        self.pose_frame_interval = max(1, int(pose_frame_interval))
        self.person_assigner = PersonTrackAssigner(max_match_dist=220.0, stale_sec=1.2)
        self._box_consecutive_hits: dict[str, int] = {}
        self._box_last_alarm_frame: dict[str, int] = {}
        # track_id → 腕点下标 → 近期观测。只在 Δt 开启时用来取 t−Δ。
        self._wrist_hist: dict[int, dict[int, list[_WristObs]]] = {}

    def reset_infer_session(self) -> None:
        """infer 重启导致 frame_idx 回绕时清空会话内状态。"""
        self._box_consecutive_hits.clear()
        self._box_last_alarm_frame.clear()
        self._wrist_hist.clear()
        self.person_assigner.tracks.clear()
        self.person_assigner.next_id = 1

    def _lag_frames(self) -> int:
        delta = float(self.dwell_delta_sec)
        if not (delta > 0):
            delta = DEFAULT_DWELL_DELTA_SEC
        return lag_frames(self.video_fps, delta)

    def _push_wrist(self, track_id: int, joint: int, obs: _WristObs) -> None:
        by_joint = self._wrist_hist.setdefault(track_id, {})
        hist = by_joint.setdefault(joint, [])
        if hist and hist[-1].frame_idx == obs.frame_idx:
            hist[-1] = obs
        else:
            hist.append(obs)
        keep_from = obs.frame_idx - int(self.video_fps * 3) - 1
        if hist[0].frame_idx < keep_from:
            by_joint[joint] = [item for item in hist if item.frame_idx >= keep_from]

    def _observe_wrists(self, track_id: int, keypoints: list, frame_idx: int) -> None:
        """记下当前腕点，供之后的帧拿来做 Δ 对比。门控关闭时不记。"""
        if not self.dwell_enabled:
            return
        for joint in WRIST_JOINTS:
            if len(keypoints) <= joint:
                continue
            kp = keypoints[joint]
            if not isinstance(kp, (list, tuple)) or len(kp) < 2:
                continue
            conf = float(kp[2]) if len(kp) > 2 else 0.0
            self._push_wrist(track_id, joint, _WristObs(frame_idx, float(kp[0]), float(kp[1]), conf))

    def _past_wrist(self, track_id: int, joint: int, frame_idx: int) -> _WristObs | None:
        """取不晚于 t−Δ 的样本。比目标早过一个姿态间隔，视为缺对照帧。"""
        hist = self._wrist_hist.get(track_id, {}).get(joint) or []
        target = frame_idx - self._lag_frames()
        slack = max(1, int(self.pose_frame_interval))
        best: _WristObs | None = None
        best_err = 10**9
        for obs in hist:
            if obs.frame_idx > target:
                continue
            err = target - obs.frame_idx
            if err < best_err:
                best_err = err
                best = obs
        if best is None or best_err > slack:
            return None
        return best

    def _wrist_is_still(
        self,
        track_id: int,
        joint: int,
        x: float,
        y: float,
        conf: float,
        frame_idx: int,
    ) -> bool:
        """同一只手在 Δ 秒内位移 ≤ τ 才算停住。缺对照或低置信度不算。"""
        if not self.dwell_enabled:
            return True
        if conf <= WRIST_SCORE_MIN:
            return False
        past = self._past_wrist(track_id, joint, frame_idx)
        if past is None or past.conf <= WRIST_SCORE_MIN:
            return False
        tau = float(self.dwell_tau_px)
        if not (tau > 0):
            tau = DEFAULT_DWELL_TAU_PX
        return math.hypot(x - past.x, y - past.y) <= tau

    def _wrist_collision_tokens(self, keypoints: list, track_id: int, frame_idx: int) -> list[str]:
        """Δt 开启时：停住且落在货框内。关闭时：只要腕点在框内。"""
        tokens: list[str] = []
        for kpt_idx in WRIST_JOINTS:
            if len(keypoints) <= kpt_idx:
                continue
            kp = keypoints[kpt_idx]
            if len(kp) < 3 or float(kp[2]) <= WRIST_SCORE_MIN:
                continue
            wx, wy = float(kp[0]), float(kp[1])
            if not self._wrist_is_still(track_id, kpt_idx, wx, wy, float(kp[2]), frame_idx):
                continue
            for box in self.boxes:
                contour = box.get("orig_contour")
                if contour is None:
                    continue
                if cv2.pointPolygonTest(contour, (wx, wy), False) >= 0:
                    token = box_collision_token(box)
                    if token:
                        tokens.append(token)
                    break
        return list(set(tokens))

    def _drop_stale_wrist_hist(self) -> None:
        alive = set(self.person_assigner.tracks)
        for track_id in list(self._wrist_hist):
            if track_id not in alive:
                self._wrist_hist.pop(track_id, None)

    def process(
        self,
        pose_frame: dict,
        prefilter: CollisionPrefilterProtocol | None = None,
    ) -> dict:
        """返回 collisions、alarm_collisions、带 track 的 skeletons（供 SSE 合并）。"""
        frame_idx = int(pose_frame.get("frame_idx") or 0)
        now_ts = frame_idx / self.video_fps if self.video_fps > 0 else 0.0
        persons = pose_frame.get("persons") or pose_frame.get("skeletons") or []

        active_collisions: list[str] = []
        skeletons_data = []
        used_track_ids: set[int] = set()
        prefilter_logs: list[PrefilterLogEntry] = []

        for person in persons:
            if not isinstance(person, dict):
                continue
            p_idx = int(person.get("person_id", len(skeletons_data)))
            keypoints = person.get("keypoints") or []
            if len(keypoints) < 11:
                skeletons_data.append(person)
                continue

            def _pt(i):
                kp = keypoints[i]
                return float(kp[0]), float(kp[1]), float(kp[2]) if len(kp) > 2 else 0.0

            lx, ly, ls = _pt(5)
            rx, ry, rs = _pt(6)
            if ls > 0.2 and rs > 0.2:
                anchor_x = (lx + rx) / 2.0
                anchor_y = (ly + ry) / 2.0
            else:
                xs = [float(k[0]) for k in keypoints if len(k) >= 2]
                ys = [float(k[1]) for k in keypoints if len(k) >= 2]
                anchor_x = sum(xs) / len(xs) if xs else 0.0
                anchor_y = sum(ys) / len(ys) if ys else 0.0

            person_track_id = self.person_assigner.assign(
                anchor_x, anchor_y, now_ts=now_ts, occupied_track_ids=used_track_ids
            )
            skel = dict(person)
            skel["person_track_id"] = person_track_id
            skeletons_data.append(skel)
            self._observe_wrists(person_track_id, keypoints, frame_idx)

            decision = None
            if prefilter is not None:
                decision = prefilter.evaluate(pose_frame, person_track_id, person)

            collision_tokens = self._wrist_collision_tokens(keypoints, person_track_id, frame_idx)
            if prefilter is not None and decision is not None and collision_tokens:
                prefilter_logs.append(PrefilterLogEntry(decision=decision, hits=collision_tokens))

            if decision is not None and decision.blocked:
                continue

            for token in collision_tokens:
                active_collisions.append(token)

        active_collisions = list(set(active_collisions))
        current_tokens = set(active_collisions)

        alarm_collisions: list[str] = []
        if self.dwell_enabled:
            # Δt 已经是「停住」判定，不再要求连续 n 帧进框。冷却仍避免同一货框每帧重复报。
            for token in current_tokens:
                last_alarm = self._box_last_alarm_frame.get(token, -10**9)
                if frame_idx - last_alarm >= self.alarm_cooldown_frames:
                    alarm_collisions.append(token)
                    self._box_last_alarm_frame[token] = frame_idx
        else:
            for token in list(self._box_consecutive_hits.keys()):
                if token not in current_tokens:
                    self._box_consecutive_hits[token] = 0
            for token in current_tokens:
                self._box_consecutive_hits[token] = self._box_consecutive_hits.get(token, 0) + 1
                last_alarm = self._box_last_alarm_frame.get(token, -10**9)
                if (
                    self._box_consecutive_hits[token] >= self.alarm_min_consecutive_frames
                    and frame_idx - last_alarm >= self.alarm_cooldown_frames
                ):
                    alarm_collisions.append(token)
                    self._box_last_alarm_frame[token] = frame_idx

        self._drop_stale_wrist_hist()
        return {
            "collisions": active_collisions,
            "alarm_collisions": alarm_collisions,
            "skeletons": skeletons_data,
            "frame_idx": frame_idx,
            "prefilter_logs": prefilter_logs,
        }
