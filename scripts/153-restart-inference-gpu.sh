#!/usr/bin/env bash
# 在 153 上重启指定摄像头推理容器（需已 load 新 gpu-onnx 镜像）
# 用法（在 153 上）: ./scripts/153-restart-inference-gpu.sh [UI端口] cam2 cam3 ...
# 或本机: ssh hqit@192.168.1.153 'bash -s' < scripts/153-restart-inference-gpu.sh 8055 cam2 cam3 cam4 cam5
set -euo pipefail

PORT="${1:-8055}"
shift || true
CAMS=("${@:-cam2 cam3 cam4 cam5}")
BASE="http://127.0.0.1:${PORT}/api"

for cam in "${CAMS[@]}"; do
  echo "==> ${cam} stop"
  curl -sf -X POST "${BASE}/cameras/${cam}/inference/stop" >/dev/null || echo "  stop: 无或已停"
  sleep 1
  echo "==> ${cam} start"
  curl -sf -X POST "${BASE}/cameras/${cam}/inference/start" | head -c 400
  echo ""
done

echo "==> 容器镜像"
docker ps --format '{{.Names}} {{.Image}}' | grep infer-cam || true
