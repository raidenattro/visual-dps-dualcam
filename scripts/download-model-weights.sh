#!/usr/bin/env bash
# 下载全部推理权重到 localdata/models/（RTMPose ONNX + YOLO26-pose）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# shellcheck disable=SC1091
source scripts/lib/load-build-env.sh
load_build_env "${ROOT}"

echo "==> RTMPose ONNX（已有则跳过）"
"${ROOT}/scripts/download-rtmpose-onnx-weights.sh"
echo "==> YOLO26-pose（已有且完整则跳过）"
"${ROOT}/scripts/download-yolo-pose-weights.sh"

echo "==> 校验权重完整性（存在 + 体积下限）..."
# shellcheck disable=SC1091
source "${ROOT}/deploy/check-model-weights.sh"
visual_dps_check_model_weights "${ROOT}/localdata"
