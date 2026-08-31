#!/usr/bin/env bash
# [非主流程] 仅 save|load 单个 gpu-onnx 镜像。整栈恢复请用 export-offline-one-shot.sh
# 用法: ./scripts/sync-gpu-onnx-image-to-153.sh [tag] [ssh目标]
# digest 可能不一致（Docker 存储差异）；不一致时用内容校验判定成功
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

TAG="${1:-}"
REMOTE="${2:-hqit@192.168.1.153}"

if [[ -z "${TAG}" ]]; then
  set -a
  # shellcheck disable=SC1090
  [[ -f .env ]] && source .env
  set +a
  if [[ -n "${INFERENCE_LITE_GPU_ONNX_IMAGE:-}" ]]; then
    TAG="${INFERENCE_LITE_GPU_ONNX_IMAGE##*:}"
  else
    TAG="$(docker images --format '{{.Tag}}' visual-dps-inference-lite-gpu-onnx \
      | grep -v '^<none>$' | grep -v '^latest$' | sort | tail -n 1)"
  fi
fi
[[ -n "${TAG}" ]] || { echo "错误: 请指定 tag 或在本机 .env 设置 INFERENCE_LITE_GPU_ONNX_IMAGE" >&2; exit 1; }

IMAGE="visual-dps-inference-lite-gpu-onnx:${TAG}"

docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
  echo "错误: 本机无镜像 ${IMAGE}" >&2
  exit 1
}

LOCAL_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
echo "本机 ${IMAGE}"
echo "  ${LOCAL_ID}"

echo "==> docker save | ssh load（约 11GB，请耐心等待）"
docker save "${IMAGE}" | ssh "${REMOTE}" docker load

echo "==> 153 digest"
REMOTE_ID="$(ssh "${REMOTE}" docker image inspect "${IMAGE}" --format '{{.Id}}' 2>/dev/null || true)"
echo "  ${REMOTE_ID:-<未找到>}"

if [[ "${LOCAL_ID}" == "${REMOTE_ID}" ]]; then
  echo "OK: digest 一致"
  exit 0
fi

echo "WARN: digest 不一致，做远程内容校验（常见，不代表传错）" >&2
VERIFY_GPU_SKIP="${VERIFY_GPU_SKIP:-0}" ssh "${REMOTE}" "VERIFY_GPU_SKIP=${VERIFY_GPU_SKIP} bash -s" -- "${IMAGE}" \
  < "${ROOT}/deploy/verify-gpu-onnx-content.sh"
echo "OK: 153 镜像内容校验通过（digest 可忽略）"
