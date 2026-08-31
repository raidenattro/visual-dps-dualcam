#!/usr/bin/env bash
# 在已有 lite-gpu 镜像上增量构建 ONNX 版（国内 apt/pip/conda/docker 镜像见 .env）
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/lib/docker-build.sh

BASE_REPO="visual-dps-inference-lite-gpu"
if ! docker image inspect "${BASE_REPO}:latest" >/dev/null 2>&1; then
  if ! docker image inspect "$(docker images "${BASE_REPO}" -q | head -1)" >/dev/null 2>&1; then
    echo "缺少基底镜像 ${BASE_REPO}，请先: ./scripts/build-inference-lite-gpu-image.sh"
    exit 1
  fi
fi

visual_dps_compose_build visual-dps-inference-lite-gpu-onnx visual-dps-inference-lite-gpu-onnx inference-lite

ref="$(visual_dps_tag_image visual-dps-inference-lite-gpu-onnx "${VISUAL_DPS_IMAGE_TAG}")"
echo "建议校验: ./scripts/verify-gpu-onnx-image.sh ${ref}"
