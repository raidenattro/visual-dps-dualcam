#!/usr/bin/env bash
# 校验 inference-lite-gpu-onnx 镜像（包装 deploy/verify-gpu-onnx-content.sh）
# 用法: ./scripts/verify-gpu-onnx-image.sh [image:tag]
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${1:-visual-dps-inference-lite-gpu-onnx:latest}"
echo "==> 校验镜像 ${IMAGE}"
"${ROOT}/deploy/verify-gpu-onnx-content.sh" "${IMAGE}"
echo "==> 通过: ${IMAGE}"
