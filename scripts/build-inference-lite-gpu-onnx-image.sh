#!/usr/bin/env bash
# 在已有 lite-gpu 镜像上增量构建 ONNX 版（国内 apt/pip/conda/docker 镜像见 .env）
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/lib/docker-build.sh

BASE_REPO="visual-dps-inference-lite-gpu"
export VISUAL_DPS_IMAGE_TAG="${VISUAL_DPS_IMAGE_TAG:-$(visual_dps_image_tag)}"
BASE_REF="$(visual_dps_tag_image "${BASE_REPO}" "${VISUAL_DPS_IMAGE_TAG}")"
if ! docker image inspect "${BASE_REF}" >/dev/null 2>&1; then
  echo "缺少基底镜像 ${BASE_REF}，请先: ./scripts/build-inference-lite-gpu-image.sh" >&2
  echo "  （需与 .env VISUAL_DPS_IMAGE_TAG=${VISUAL_DPS_IMAGE_TAG} 一致）" >&2
  exit 1
fi
echo "ONNX 基底: ${BASE_REF}"

visual_dps_compose_build visual-dps-inference-lite-gpu-onnx visual-dps-inference-lite-gpu-onnx inference-lite

ref="$(visual_dps_tag_image visual-dps-inference-lite-gpu-onnx "${VISUAL_DPS_IMAGE_TAG}")"
echo "建议校验: ./scripts/verify-gpu-onnx-image.sh ${ref}"
