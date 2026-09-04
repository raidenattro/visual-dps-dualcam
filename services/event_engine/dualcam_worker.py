"""消费 pose stream：按巷道对齐 L/R，再走 DualcamProcessor。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Any

from services.aisle_store import grouped_cameras, load_aisle
from services.box_identity import parse_collision_token
from services.dualcam_config import pair_window_sec
from services.event_bus import publish_event_frame
from services.event_engine.dualcam_processor import DualcamProcessor
from services.event_engine.sharding import owns_aisle
from services.event_engine.worker import EventRedisWorker
from services.pipeline_log import log_pipeline_stage

logger = logging.getLogger(__name__)


def _compact_contact_probe(probes: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for p in probes or []:
        if not isinstance(p, dict):
            continue
        w = str(p.get("wrist") or "?")
        src = str(p.get("src") or "-")
        d = p.get("d")
        d_s = "—" if d is None else f"{float(d):.3f}"
        cell = str(p.get("cell") or "-")
        alarm = "1" if p.get("wrist_alarm") else "0"
        extra = ""
        if p.get("reason"):
            extra = f" reason={p['reason']}"
        if p.get("held"):
            extra += " held=1"
        if p.get("preview"):
            extra += " preview=1"
        parts.append(f"{w}[src={src} d={d_s} cell={cell} alarm={alarm}{extra}]")
    return " ".join(parts) if parts else "—"


class DualcamRedisWorker(EventRedisWorker):
    """同一 aisle_id 的 L/R 必须落在本 worker 的 shard 上。"""

    def __init__(self, app_config: dict, callback_reporter=None):
        # 须在 super().__init__ 之前：父类 init 会调用 _apply_runtime_settings
        self._aisle_proc: dict[str, DualcamProcessor] = {}
        self._aisle_mtime: dict[str, float] = {}
        self._aisle_last_frame: dict[str, int] = {}
        self._pending: dict[str, dict[str, tuple[float, dict]]] = {}
        self._last_role_mono: dict[str, dict[str, float]] = {}
        self._pair_window = pair_window_sec(app_config)
        self._pair_window_mono = 0.0
        self._last_drop_log: dict[str, float] = {}
        super().__init__(app_config, callback_reporter=callback_reporter)
        logger.info("dualcam pair_window=%.3fs", self._pair_window)

    def _apply_runtime_settings(self, infer_cfg: dict) -> None:
        super()._apply_runtime_settings(infer_cfg)
        procs = getattr(self, "_aisle_proc", None)
        if not procs:
            return
        for proc in procs.values():
            proc.alarm_min_consecutive_frames = self._alarm_min

    def _maybe_reset_aisle_on_frame_regression(
        self,
        aisle_id: str,
        proc: DualcamProcessor,
        frame_idx: int,
    ) -> None:
        """frame_idx 回退视为 infer 重启，清空连续命中计数（避免旧状态导致永不告警）。"""
        last = self._aisle_last_frame.get(aisle_id, -1)
        if last >= 0 and frame_idx < last:
            proc.reset_infer_session()
            logger.info(
                "dualcam worker: infer frame_idx regression aisle=%s frame=%s last=%s; reset collision session",
                aisle_id,
                frame_idx,
                last,
            )
        self._aisle_last_frame[aisle_id] = frame_idx

    def _json_root(self) -> str:
        return self._json_dir

    def _get_processor(self, camera_id: str, infer_w: int, infer_h: int):
        """单路不再建 CollisionProcessor；成组后走 aisle processor。"""
        return None

    def _refresh_pair_window(self) -> float:
        now = time.monotonic()
        if now - self._pair_window_mono < 2.0:
            return self._pair_window
        self._pair_window_mono = now
        try:
            self._pair_window = pair_window_sec(self.app_config)
        except Exception:
            pass
        return self._pair_window

    def _aisle_processor(self, aisle_id: str) -> DualcamProcessor | None:
        import os

        path = os.path.join(self._json_root(), "aisles", f"{aisle_id}.json")
        mtime = os.path.getmtime(path) if os.path.isfile(path) else 0.0
        proc = self._aisle_proc.get(aisle_id)
        if proc is not None and self._aisle_mtime.get(aisle_id) == mtime:
            return proc
        data = load_aisle(aisle_id, self._json_root())
        if not data:
            return None
        proc = DualcamProcessor(data)
        proc.alarm_min_consecutive_frames = self._alarm_min
        self._aisle_proc[aisle_id] = proc
        self._aisle_mtime[aisle_id] = mtime
        return proc

    def _owns_aisle(self, aisle_id: str) -> bool:
        return owns_aisle(aisle_id)

    def _note(self, key: str, msg: str, *args) -> None:
        now = time.monotonic()
        if now - self._last_drop_log.get(key, 0.0) < 5.0:
            return
        self._last_drop_log[key] = now
        logger.warning(msg, *args)

    def _ingest_pose(self, pose: dict) -> dict[str, Any] | None:
        """更新配对桶。返回待处理 job，或 None（等待对侧 / 丢弃）。"""
        camera_id = str(pose.get("camera_id") or "").strip()
        if not camera_id:
            return None
        groups = grouped_cameras(self._json_root())
        g = groups.get(camera_id)
        if not g:
            self._note(f"ungrouped:{camera_id}", "dualcam worker: 未成组相机丢弃 pose camera=%s", camera_id)
            return None
        aisle_id = g["aisle_id"]
        role = g["role"]
        if not self._owns_aisle(aisle_id):
            self._note(
                f"shard:{camera_id}",
                "dualcam worker: camera=%s 巷道 %s 不在本实例分片上（pose 仍按 camera_id 分片时会对不上），丢弃",
                camera_id,
                aisle_id,
            )
            return None

        window = self._refresh_pair_window()
        now = float(pose.get("ts") or time.time())
        now_mono = time.monotonic()
        bucket = self._pending.setdefault(aisle_id, {})
        seen = self._last_role_mono.setdefault(aisle_id, {})
        seen[role] = now_mono
        bucket[role] = (now, pose)

        stale = [k for k, (t, _) in bucket.items() if now - t > window * 3]
        for k in stale:
            bucket.pop(k, None)

        proc = self._aisle_processor(aisle_id)
        if "L" not in bucket or "R" not in bucket:
            missing = "R" if "L" in bucket else "L"
            other_last = seen.get(missing)
            wait_s = max(window * 2.0, 0.25)
            if other_last is not None and (now_mono - other_last) <= wait_s:
                return None
            self._note(
                f"wait:{aisle_id}:{missing}",
                "dualcam worker: 巷道 %s 只收到 %s，等待对侧 %s 已 %.2fs（窗=%.3fs）。单路 3D 预览仍推送。",
                aisle_id,
                role,
                missing,
                now_mono - (other_last or now_mono),
                window,
            )
            if proc is None or not proc.ready():
                return None
            return {"kind": "single", "aisle_id": aisle_id, "role": role, "pose": pose, "proc": proc}

        t_l, pose_l = bucket["L"]
        t_r, pose_r = bucket["R"]
        if abs(t_l - t_r) > window:
            self._note(
                f"skew:{aisle_id}",
                "dualcam worker: 巷道 %s L/R 时差 %.3fs 超过配对窗 %.3fs，本帧改单路预览。",
                aisle_id,
                abs(t_l - t_r),
                window,
            )
            if t_l <= t_r:
                bucket.pop("L", None)
            else:
                bucket.pop("R", None)
            if proc is None or not proc.ready():
                return None
            return {"kind": "single", "aisle_id": aisle_id, "role": role, "pose": pose, "proc": proc}

        bucket.pop("L", None)
        bucket.pop("R", None)
        if proc is None or not proc.ready():
            reason = proc.not_ready_reason() if proc else f"巷道 {aisle_id} 标定文件读失败"
            self._note(f"ready:{aisle_id}", "dualcam worker: %s，丢弃配对帧", reason)
            return None
        return {
            "kind": "pair",
            "aisle_id": aisle_id,
            "pose_l": pose_l,
            "pose_r": pose_r,
            "proc": proc,
        }

    def _publish_overlay(self, proc: DualcamProcessor, result: dict) -> None:
        frame_idx = int(result.get("frame_idx") or 0)
        collisions = result.get("collisions") or []
        alarm_collisions = result.get("alarm_collisions") or []
        skel_map = {
            proc.cam_l: result.get("skeletons_l"),
            proc.cam_r: result.get("skeletons_r"),
        }
        people = result.get("persons_3d") or []
        for cid in (proc.cam_l, proc.cam_r):
            if not cid:
                continue
            publish_event_frame(
                cid,
                frame_idx=frame_idx,
                collisions=collisions,
                alarm_collisions=alarm_collisions,
                skeletons=skel_map.get(cid),
                persons_3d=people,
            )

    def _log_worker_done(
        self,
        job: dict[str, Any],
        proc: DualcamProcessor,
        result: dict,
        worker_ms: float,
    ) -> None:
        """有命中/告警时强制落盘并带上货位 token，便于按 frame 统计误报。"""
        collisions = list(result.get("collisions") or [])
        alarm_collisions = list(result.get("alarm_collisions") or [])
        contact_probe = list(result.get("contact_probe") or [])
        fields: dict[str, Any] = {
            "worker_ms": worker_ms,
            "hits": len(collisions),
            "alarms": len(alarm_collisions),
            "kind": job.get("kind"),
            "aisle_id": job.get("aisle_id"),
        }
        if collisions:
            fields["hit_tokens"] = collisions
        if alarm_collisions:
            fields["alarm_tokens"] = alarm_collisions
        if contact_probe:
            fields["contact_probe"] = contact_probe
            fields["contact"] = _compact_contact_probe(contact_probe)
        near_wall = any(
            isinstance(p.get("d"), (int, float)) and float(p["d"]) < float(proc.contact_m) * 2.0
            for p in contact_probe
        )
        hold_src = any(str(p.get("src") or "") in ("Lhold", "Rhold", "Lmono", "Rmono") for p in contact_probe)
        force_log = bool(collisions or alarm_collisions or near_wall or hold_src)
        # pipeline 按 infer 相机过滤；aisle_id 写入 fields 便于 rg aisle11
        log_camera = proc.cam_l or proc.cam_r or ""
        log_pipeline_stage(
            "worker_done",
            camera_id=log_camera,
            frame_idx=int(result.get("frame_idx") or 0),
            sample=not force_log,
            **fields,
        )

    def _execute_job(self, job: dict[str, Any]) -> tuple[DualcamProcessor, dict] | None:
        proc: DualcamProcessor = job["proc"]
        if job["kind"] == "single":
            frame_idx = int(job["pose"].get("frame_idx") or 0)
        else:
            frame_idx = int(
                job["pose_l"].get("frame_idx")
                or job["pose_r"].get("frame_idx")
                or 0
            )
        self._maybe_reset_aisle_on_frame_regression(str(job["aisle_id"]), proc, frame_idx)
        started = time.monotonic()
        if job["kind"] == "single":
            result = proc.process_single(job["role"], job["pose"])
        else:
            result = proc.process_pair(job["pose_l"], job["pose_r"])
        worker_ms = round((time.monotonic() - started) * 1000.0, 1)
        self._log_worker_done(job, proc, result, worker_ms)
        alarm_collisions = result.get("alarm_collisions") or []
        if self.callback_reporter and alarm_collisions and proc.cam_l:
            upload_tag = f"infer_{proc.cam_l}"
            video_time_sec = int(result.get("frame_idx") or 0) / max(self._video_fps, 1.0)
            for collision in alarm_collisions:
                shelf_code, box_id = parse_collision_token(collision)
                if not box_id:
                    continue
                self.callback_reporter.enqueue_pick_finished(
                    box_id=box_id,
                    frame_idx=int(result.get("frame_idx") or 0),
                    video_time_sec=video_time_sec,
                    upload_tag=upload_tag,
                    shelf_code=shelf_code or None,
                )
        return proc, result

    def _run_aisle_jobs_sync(self, jobs: list[dict[str, Any]]) -> None:
        """同一巷道串行（processor 有状态）；不同巷道由 gather 进不同线程。"""
        for job in jobs:
            out = self._execute_job(job)
            if out is None:
                continue
            proc, result = out
            self._publish_overlay(proc, result)

    async def _handle_pose_batch(self, batch: list[tuple[str, str, str | None]]) -> None:
        self._refresh_runtime_settings_if_needed()
        by_aisle: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for _stream, _msg_id, payload in batch:
            if not payload:
                continue
            try:
                pose = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(pose, dict) or pose.get("kind") != "pose":
                continue
            job = self._ingest_pose(pose)
            if job is not None:
                by_aisle[job["aisle_id"]].append(job)
        if not by_aisle:
            return
        await asyncio.gather(
            *[asyncio.to_thread(self._run_aisle_jobs_sync, jobs) for jobs in by_aisle.values()]
        )

    async def _handle_pose_payload(self, payload: str) -> None:
        await self._handle_pose_batch([("", "", payload)])
