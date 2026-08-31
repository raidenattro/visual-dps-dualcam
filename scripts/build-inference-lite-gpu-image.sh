#!/usr/bin/env bash
# 构建 GPU 推理镜像（apt/pip/github 国内源 + 可选 BUILD_HTTP_PROXY）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
# shellcheck disable=SC1091
source scripts/lib/docker-build.sh

visual_dps_compose_build visual-dps-inference-lite-gpu visual-dps-inference-lite-gpu inference-lite
echo "推理容器需 --gpus all"
