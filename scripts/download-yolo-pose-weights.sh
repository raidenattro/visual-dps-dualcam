#!/usr/bin/env bash
# 在宿主机下载 YOLO26-pose 权重到 localdata（compose 卷挂载进推理容器；可走 .env 的 GITHUB_PROXY_BASE）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# shellcheck disable=SC1091
source scripts/lib/load-build-env.sh
load_build_env "${ROOT}"

export DOWNLOAD_HTTP_PROXY="${BUILD_HTTP_PROXY:-${HTTP_PROXY:-}}"

DEST="${1:-${ROOT}/localdata/models/yolo_pose}"
mkdir -p "${DEST}"

echo "DEST=${DEST}"
echo "GITHUB_PROXY_BASE=${GITHUB_PROXY_BASE:-<直连 GitHub>}"
echo "DOWNLOAD_HTTP_PROXY=${DOWNLOAD_HTTP_PROXY:-<无>}"

GITHUB_PROXY_BASE="${GITHUB_PROXY_BASE:-}" \
OPENMMLAB_MIRROR_BASE="${OPENMMLAB_MIRROR_BASE:-}" \
DOWNLOAD_HTTP_PROXY="${DOWNLOAD_HTTP_PROXY:-}" \
  sh "${ROOT}/docker/download-yolo-pose-weights.sh" "${DEST}"
