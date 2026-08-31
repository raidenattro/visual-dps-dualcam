#!/usr/bin/env bash
# 构建 MediaPipe / RTMPose ONNX 轻量推理镜像
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
# shellcheck disable=SC1091
source scripts/lib/docker-build.sh

visual_dps_compose_build visual-dps-inference-lite visual-dps-inference-lite inference-lite
echo "示例: INFERENCE_BACKEND=rtmpose_t docker compose up -d visual-dps-ui"
