#!/usr/bin/env bash
# 从本机已有离线包/weights 目录补齐 localdata/models（不联网）
set -euo pipefail

ROOT="${1:-}"
LOCALDATA="${2:-}"
if [[ -z "${ROOT}" || -z "${LOCALDATA}" ]]; then
  echo "用法: recover-model-weights.sh <repo-root> <localdata-dir>" >&2
  exit 1
fi

_SPEC="$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model-weights-spec.sh"
# shellcheck source=deploy/model-weights-spec.sh
source "${_SPEC}"
# shellcheck source=deploy/check-model-weights.sh
source "$(dirname "${BASH_SOURCE[0]}")/check-model-weights.sh"

MODELS="${LOCALDATA%/}/models"
mkdir -p "${MODELS}/rtmpose_onnx" "${MODELS}/yolo_pose"

_search_roots() {
  local d
  for d in \
    "${ROOT}/dist"/*/weights \
    "${ROOT}/dist"/*/app/localdata/models \
    "${ROOT}/dist"/*/localdata/models; do
    [[ -d "${d}" ]] && printf '%s\n' "${d}"
  done | sort -u
}

recovered=0
rel path need size src
while IFS= read -r rel; do
  [[ -n "${rel}" ]] || continue
  path="${MODELS}/${rel}"
  need="$(vdps_model_min_bytes "${rel}")"
  if [[ -f "${path}" ]]; then
    size="$(wc -c < "${path}" | tr -d ' ')"
    [[ "${size}" -ge "${need}" ]] && continue
  fi
  while IFS= read -r base; do
    src="${base%/}/${rel}"
    if [[ -f "${src}" ]]; then
      size="$(wc -c < "${src}" | tr -d ' ')"
      if [[ "${size}" -ge "${need}" ]]; then
        mkdir -p "$(dirname "${path}")"
        cp -f "${src}" "${path}"
        echo "  已从 ${src} 恢复 ${rel}"
        recovered=$((recovered + 1))
        break
      fi
    fi
  done < <(_search_roots)
done < <(vdps_each_required_model_weight)

if [[ "${recovered}" -gt 0 ]]; then
  echo "本地恢复 ${recovered} 个权重文件"
fi

visual_dps_check_model_weights "${LOCALDATA}"
