#!/usr/bin/env bash
# 延锋等现场：沿用已有 gpu / gpu-onnx 推理镜像，从 20260907-dualcam-3041fb0 retag 为本包 tag。
# Docker 无组权限时自动 sudo（与 retag-infer-images.sh 相同）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIELD_OLD_TAG="${VISUAL_DPS_FIELD_INFER_OLD_TAG:-20260907-dualcam-3041fb0}"

export VISUAL_DPS_DOCKER_SUDO="${VISUAL_DPS_DOCKER_SUDO:-auto}"

echo "==> 现场推理 retag（旧 tag: ${FIELD_OLD_TAG}）"
echo "    处理: visual-dps-inference-lite-gpu、visual-dps-inference-lite-gpu-onnx"
echo "    新 tag: 从 app/.env 的 VISUAL_DPS_IMAGE_TAG 读取"
echo "    强制 sudo: export VISUAL_DPS_DOCKER_SUDO=1"
echo ""

exec "${SCRIPT_DIR}/retag-infer-images.sh" "${FIELD_OLD_TAG}" "$@"
