"""视频上传、转码和首帧提取相关函数。"""

import base64
import os
import shutil
from typing import Tuple

import cv2
from fastapi import UploadFile

from core.state import STATE


def resize_frame_to_height(frame, target_height: int):
    """按目标高度等比缩放帧（不放大）。"""
    src_h, src_w = frame.shape[:2]
    target_h = min(int(target_height), src_h)
    target_w = int(round(src_w * (target_h / src_h)))
    target_w = max(2, target_w - (target_w % 2))
    target_h = max(2, target_h - (target_h % 2))

    if target_w == src_w and target_h == src_h:
        return frame

    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def read_non_black_frame(cap, max_reads: int = 30):
    best_frame = None
    best_score = -1.0
    for _ in range(max_reads):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        score = float(frame.mean())
        if score > best_score:
            best_score = score
            best_frame = frame
        if score >= 10.0:
            break
    return best_frame


def reserve_next_upload_id(counter_file: str, base_localdata_dir: str) -> int:
    os.makedirs(base_localdata_dir, exist_ok=True)
    current = 0
    if os.path.exists(counter_file):
        try:
            with open(counter_file, "r", encoding="utf-8") as f:
                current = int((f.read() or "0").strip())
        except Exception:
            current = 0

    next_id = current + 1
    with open(counter_file, "w", encoding="utf-8") as f:
        f.write(str(next_id))
    return next_id


def transcode_video_to_height(src_path: str, dst_path: str, target_height: int) -> Tuple[int, int]:
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开上传视频进行转码")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    target_h = min(int(target_height), src_h)
    target_w = int(round(src_w * (target_h / src_h)))
    target_w = max(2, target_w - (target_w % 2))
    target_h = max(2, target_h - (target_h % 2))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst_path, fourcc, fps, (target_w, target_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("无法创建转码输出文件")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            writer.write(resized)
    finally:
        cap.release()
        writer.release()

    return target_w, target_h


async def handle_video_upload(
    file: UploadFile,
    upload_dir: str,
    upload_480p_dir: str,
    json_dir: str,
    counter_file: str,
    base_localdata_dir: str,
    transcode_height: int,
):
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(upload_480p_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    upload_id = reserve_next_upload_id(counter_file, base_localdata_dir)
    upload_tag = f"u{upload_id:06d}"

    raw_video_path = os.path.join(upload_dir, f"{upload_tag}_src.mp4")
    compressed_video_path = os.path.join(upload_480p_dir, f"{upload_tag}_480p.mp4")
    upload_json_path = os.path.join(json_dir, f"{upload_tag}_boxes.json")

    with open(raw_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        out_w, out_h = transcode_video_to_height(raw_video_path, compressed_video_path, transcode_height)
        STATE.video_path = compressed_video_path
        print(f"✅ 上传视频已转码: {compressed_video_path} ({out_w}x{out_h})")
    except Exception as e:
        STATE.video_path = raw_video_path
        print(f"⚠️ 转码失败，回退原视频: {e}")

    STATE.is_inferencing = False
    STATE.upload_id = upload_id
    STATE.upload_tag = upload_tag
    STATE.json_path = upload_json_path
    STATE.source_type = "file"
    STATE.source_url = ""

    return {"status": "success", "upload_id": upload_id, "upload_tag": upload_tag}


def get_first_frame_b64(video_path: str, capture_height: int | None = None):
    if not video_path:
        return {"error": "No video"}

    is_url = video_path.startswith("rtsp://") or video_path.startswith("http://") or video_path.startswith("https://")
    if not is_url and not os.path.exists(video_path):
        return {"error": "No video"}

    cap = cv2.VideoCapture(video_path)
    frame = read_non_black_frame(cap) if is_url else None
    if frame is None:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {"error": "Read failed"}
    cap.release()

    if capture_height:
        frame = resize_frame_to_height(frame, capture_height)

    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return {"status": "success", "image": base64.b64encode(buffer).decode("utf-8")}


def initialize_source_from_config(app_config: dict, json_dir: str, default_json_file: str):
    source_cfg = app_config.get("source", {})
    if not isinstance(source_cfg, dict):
        source_cfg = {}

    stream_enabled = bool(source_cfg.get("enabled", False))
    stream_url = str(source_cfg.get("stream_url", "")).strip()

    if not stream_enabled or not stream_url:
        return

    os.makedirs(json_dir, exist_ok=True)

    upload_tag = str(source_cfg.get("upload_tag", "stream_config")).strip() or "stream_config"
    annotation_json = str(source_cfg.get("annotation_json", "")).strip()
    if annotation_json:
        if os.path.isabs(annotation_json):
            upload_json_path = annotation_json
        else:
            upload_json_path = os.path.join(json_dir, annotation_json)
    else:
        upload_json_path = default_json_file

    STATE.video_path = stream_url
    STATE.is_inferencing = False
    STATE.upload_id = 0
    STATE.upload_tag = upload_tag
    STATE.json_path = upload_json_path
    STATE.source_type = "stream"
    STATE.source_url = stream_url

    print(f"✅ 已从配置文件初始化网络流: {stream_url} json_path={upload_json_path}")
