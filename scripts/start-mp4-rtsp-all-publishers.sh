#!/usr/bin/env bash
# 宿主机 MP4 → 本机 MediaMTX 多路推流（153 等，不经 WSL 中继）
# 用法: ./scripts/start-mp4-rtsp-all-publishers.sh [视频文件]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VIDEO="${1:-${ROOT}/test/c7e2fb9b6551c2fc6a4c93ee65ac6c23_raw.mp4}"

# 与 camera_ips / mediamtx publisher path 对齐（不含 cam13：由 MediaMTX 拉 149）
# 全量: cam1 cam2 cam3 cam4 cam5 cam6 cam7 cam8 cam9 cam10 cam11 12 ces
# 默认仅 cam1 cam2（减轻 CPU）；一半: MP4_PUBLISH_PATHS=half  全量: MP4_PUBLISH_PATHS=all
if [[ "${MP4_PUBLISH_PATHS:-}" == "all" ]]; then
  PATHS=(cam1 cam2 cam3 cam4 cam5 cam6 cam7 cam8 cam9 cam10 cam11 12 ces)
elif [[ "${MP4_PUBLISH_PATHS:-}" == "half" ]]; then
  PATHS=(cam1 cam2 cam3 cam6 cam7 cam8)
else
  PATHS=(cam1 cam2)
fi

if [[ ! -f "${VIDEO}" ]]; then
  echo "视频不存在: ${VIDEO}"
  exit 1
fi

for p in "${PATHS[@]}"; do
  echo ">>> ${p}"
  "${SCRIPT_DIR}/start-mp4-rtsp.sh" "${VIDEO}" "${p}"
done

echo ""
echo "已启动 ${#PATHS[@]} 路 MP4 推流 → rtsp://127.0.0.1:8554/<path>"
echo "cam13 请保持 mediamtx 从 192.168.1.149 拉流。"
