#!/usr/bin/env bash
# 统一 docker compose build：注入 VISUAL_DPS_IMAGE_TAG，避免仅使用 :latest
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/lib/load-build-env.sh"
load_build_env "${ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/lib/docker-image-tag.sh"

visual_dps_compose_build() {
  local service="$1"
  local repo="$2"
  local profile="${3:-}"
  export VISUAL_DPS_IMAGE_TAG="${VISUAL_DPS_IMAGE_TAG:-$(visual_dps_image_tag)}"
  export DOCKER_BUILDKIT=1
  export COMPOSE_DOCKER_CLI_BUILD=1
  local ref
  ref="$(visual_dps_tag_image "${repo}" "${VISUAL_DPS_IMAGE_TAG}")"

  echo "构建 ${ref}"
  echo "  APT_MIRROR=${APT_MIRROR}"
  echo "  PIP_INDEX=${PIP_INDEX}"
  echo "  TORCH_INDEX=${TORCH_INDEX:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu121}"
  echo "  GITHUB_PROXY_BASE=${GITHUB_PROXY_BASE:-<直连>}"
  echo "  HTTP_PROXY=${HTTP_PROXY:-<无>}"

  local -a args=(compose)
  if [[ -n "${profile}" ]]; then
    args+=(--profile "${profile}")
  fi
  args+=(build "${service}")

  (cd "${ROOT}" && docker "${args[@]}")

  if [[ "${DOCKER_TAG_ALSO_LATEST:-0}" == "1" ]]; then
    docker tag "${ref}" "${repo}:latest"
    echo "  另打标签: ${repo}:latest"
  fi

  echo "OK: ${ref}"
  echo "  export INFERENCE_LITE_GPU_ONNX_IMAGE=${ref}"
}
