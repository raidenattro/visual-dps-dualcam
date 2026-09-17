"""Dualcam / legacy 摄像头分区一致性巡检。"""

from __future__ import annotations

import json
import os
from typing import Any

from services.aisle_store import camera_collision_mode, grouped_cameras, list_aisles, load_aisle


def _norm_path(path: str) -> str:
    return str(path or "").strip().lstrip("/")


def legacy_path_aisle_conflict(path: str, json_dir: str | None) -> str | None:
    """单路（legacy）通道与已有巷道编号 / L-R 专用通道冲突。"""
    p = _norm_path(path)
    if not p:
        return None
    for item in list_aisles(json_dir, bound_only=False):
        aid = str(item.get("aisle_id") or "").strip()
        if not aid:
            continue
        if p == aid:
            return (
                f"通道「{p}」与现有巷道编号冲突，单路请改用其他通道号。"
            )
        if p in (f"{aid}-L", f"{aid}-R"):
            return (
                f"通道「{p}」为巷道 {aid} 的 L/R 专用通道，不能作为单路添加。"
            )
    return None


def aisle_id_legacy_path_conflict(
    aisle_id: str, camera_ips_file: str, json_dir: str | None
) -> str | None:
    """创建巷道时，编号不得与已有未成组单路的通道号相同。"""
    aid = str(aisle_id or "").strip()
    if not aid:
        return None
    grouped = set(grouped_cameras(json_dir).keys())
    for cam in load_cameras(camera_ips_file):
        cid = str(cam.get("id") or "").strip()
        if cid in grouped:
            continue
        cp = _norm_path(cam.get("path") or cam.get("id"))
        if cp == aid:
            return (
                f"巷道编号「{aid}」与已有单路通道「{cp}」冲突，"
                "请先删除或修改该单路。"
            )
    return None
from services.annotation_service import existing_camera_annotation_path, flatten_annotation_boxes
from services.camera_store import load_cameras


def _json_dir_from_app(app_config: dict) -> str:
    return str(app_config.get("paths", {}).get("json_dir", "localdata/json") or "localdata/json")


def run_partition_audit(app_config: dict, camera_ips_file: str) -> dict[str, Any]:
    """只读：检查 camera_id 分区、巷道引用、legacy 标定就绪。"""
    json_dir = _json_dir_from_app(app_config)
    cameras = load_cameras(camera_ips_file)
    ids_in_list = []
    duplicate_ids: list[str] = []
    seen: set[str] = set()
    for cam in cameras:
        cid = str(cam.get("id") or cam.get("path") or "").strip()
        if not cid:
            continue
        ids_in_list.append(cid)
        if cid in seen:
            duplicate_ids.append(cid)
        seen.add(cid)

    grouped = grouped_cameras(json_dir)
    aisle_orphans: list[dict[str, str]] = []
    aisle_incomplete: list[str] = []
    grouped_overlap: list[dict[str, str]] = []
    camera_to_aisle: dict[str, str] = {}

    for item in list_aisles(json_dir, bound_only=False):
        aid = item["aisle_id"]
        data = load_aisle(aid, json_dir) or {}
        cams = data.get("cameras") or {}
        left = str((cams.get("L") or {}).get("camera_id") or "").strip()
        right = str((cams.get("R") or {}).get("camera_id") or "").strip()
        if left and left not in seen:
            aisle_orphans.append({"aisle_id": aid, "camera_id": left, "role": "L"})
        if right and right not in seen:
            aisle_orphans.append({"aisle_id": aid, "camera_id": right, "role": "R"})
        if bool(left) ^ bool(right):
            aisle_incomplete.append(aid)
        for cid, role in ((left, "L"), (right, "R")):
            if not cid:
                continue
            prev = camera_to_aisle.get(cid)
            if prev and prev != aid:
                grouped_overlap.append({"camera_id": cid, "aisle_a": prev, "aisle_b": aid})
            camera_to_aisle[cid] = aid

    legacy_not_ready: list[str] = []
    legacy_ready: list[str] = []
    dualcam_ids: list[str] = []
    legacy_aisle_path_conflicts: list[dict[str, str]] = []

    for cid in ids_in_list:
        mode = camera_collision_mode(cid, json_dir)
        if mode == "dualcam":
            dualcam_ids.append(cid)
            continue
        cam_rec = next((c for c in cameras if str(c.get("id") or "") == cid), None)
        if cam_rec:
            cp = _norm_path(cam_rec.get("path") or cam_rec.get("id"))
            msg = legacy_path_aisle_conflict(cp, json_dir)
            if msg:
                legacy_aisle_path_conflicts.append(
                    {"camera_id": cid, "path": cp, "message": msg}
                )
        apath = existing_camera_annotation_path(json_dir, cid, camera=cam_rec, camera_ips_file=camera_ips_file)
        ready = False
        if os.path.isfile(apath):
            try:
                with open(apath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and flatten_annotation_boxes(data):
                    ready = True
            except (OSError, json.JSONDecodeError):
                pass
        if ready:
            legacy_ready.append(cid)
        else:
            legacy_not_ready.append(cid)

    warnings: list[str] = []
    if duplicate_ids:
        warnings.append(f"camera_ips 重复 id: {', '.join(sorted(set(duplicate_ids)))}")
    if aisle_orphans:
        warnings.append(f"巷道引用了不在 camera_ips 的摄像头（{len(aisle_orphans)} 处）")
    if grouped_overlap:
        warnings.append(f"同一 camera 出现在多个巷道（{len(grouped_overlap)} 处）")
    if aisle_incomplete:
        warnings.append(f"巷道仅绑定了单侧 L/R: {', '.join(aisle_incomplete)}")
    if legacy_aisle_path_conflicts:
        warnings.append(
            f"单路通道与巷道命名冲突（{len(legacy_aisle_path_conflicts)} 处）"
        )

    return {
        "status": "ok",
        "camera_count": len(ids_in_list),
        "dualcam_camera_ids": dualcam_ids,
        "legacy_ready_ids": legacy_ready,
        "legacy_not_ready_ids": legacy_not_ready,
        "duplicate_ids": sorted(set(duplicate_ids)),
        "aisle_orphans": aisle_orphans,
        "aisle_incomplete": aisle_incomplete,
        "grouped_overlap": grouped_overlap,
        "legacy_aisle_path_conflicts": legacy_aisle_path_conflicts,
        "warnings": warnings,
        "ok": not warnings,
    }
