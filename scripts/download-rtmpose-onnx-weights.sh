#!/usr/bin/env bash
# 宿主机下载 RTMPose ONNX 到 localdata（读 .env 镜像/代理，已有文件跳过）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# shellcheck disable=SC1091
source scripts/lib/load-build-env.sh
load_build_env "${ROOT}"

# 下载脚本走 BUILD_HTTP_PROXY，避免 WSL 无效代理
export DOWNLOAD_HTTP_PROXY="${BUILD_HTTP_PROXY:-${HTTP_PROXY:-}}"

DEST="${1:-${ROOT}/localdata/models/rtmpose_onnx}"
mkdir -p "${DEST}"

echo "DEST=${DEST}"
echo "OPENMMLAB_MIRROR_BASE=${OPENMMLAB_MIRROR_BASE:-<官方 download.openmmlab.com>}"
echo "DOWNLOAD_HTTP_PROXY=${DOWNLOAD_HTTP_PROXY:-<无>}"

GITHUB_PROXY_BASE="${GITHUB_PROXY_BASE:-}" \
OPENMMLAB_MIRROR_BASE="${OPENMMLAB_MIRROR_BASE:-}" \
DOWNLOAD_HTTP_PROXY="${DOWNLOAD_HTTP_PROXY:-}" \
  sh "${ROOT}/docker/download-rtmpose-onnx-weights.sh" "${DEST}"
