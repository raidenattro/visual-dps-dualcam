#!/usr/bin/env bash
# 同一段 MP4 同时推到多路 MediaMTX path（cam2 cam3 … 各起一个 ffmpeg）
# 用法:
#   ./scripts/start-mp4-rtsp-multi.sh cam2 cam3 cam4 cam5
#   ./scripts/start-mp4-rtsp-multi.sh /path/to/video.mp4 cam2 cam3 cam4 cam5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VIDEO="/mnt/c/Users/sugar/Videos/33611ddf17439fe92fa3620b1fe6da92.mp4"

if [[ $# -lt 1 ]]; then
  echo "用法: $0 [视频文件] path1 [path2 ...]"
  echo "示例: $0 cam2 cam3 cam4 cam5"
  echo "示例: $0 /path/to/demo.mp4 cam2 cam3"
  exit 1
fi

if [[ -f "${1}" ]]; then
  VIDEO="${1}"
  shift
else
  VIDEO="${DEFAULT_VIDEO}"
fi

PATHS=("$@")
if [[ ${#PATHS[@]} -eq 0 ]]; then
  echo "请至少指定一个 MediaMTX path（如 cam2 cam3）"
  exit 1
fi

for name in "${PATHS[@]}"; do
  echo ">>> 启动推流 ${name}"
  "${SCRIPT_DIR}/start-mp4-rtsp.sh" "${VIDEO}" "${name}"
done

echo ""
echo "已处理 ${#PATHS[@]} 路。Dashboard 刷新后应显示在线（每路需有独立推流进程）。"
