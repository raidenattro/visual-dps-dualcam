#!/usr/bin/env sh
# 下载 YOLO26-pose 权重（已有且完整则跳过；GITHUB_PROXY_BASE 加速）
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=deploy/model-weights-spec.sh
. "${ROOT}/deploy/model-weights-spec.sh"
# shellcheck source=docker/download-http.sh
. "${SCRIPT_DIR}/download-http.sh"

DEST="${1:-/app/localdata/models/yolo_pose}"
mkdir -p "${DEST}"

BASE="https://github.com/ultralytics/assets/releases/download/v8.4.0"

for f in yolo26n-pose.pt yolo26s-pose.pt yolo26m-pose.pt yolo26l-pose.pt; do
  rel="yolo_pose/${f}"
  out="${DEST}/${f}"
  need="$(vdps_model_min_bytes "${rel}")"
  url="$(vdps_github_url "${BASE}/${f}")"
  vdps_fetch_file "${url}" "${out}" "${need}" "${f}" || exit 1
done

echo "==> YOLO26-pose 权重已就绪: ${DEST}"
