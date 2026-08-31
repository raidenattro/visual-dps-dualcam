#!/usr/bin/env bash
# 检查 localdata/models 下推理权重：必须存在且体积 >= 约定下限（防半截下载）
# 用法: visual_dps_check_model_weights /path/to/localdata

_CHECK_SPEC="$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model-weights-spec.sh"
# shellcheck source=deploy/model-weights-spec.sh
source "${_CHECK_SPEC}"

_visual_dps_check_models_dir_impl() {
  local models_dir="${1:-}"
  local rel path need size missing=0 total=0
  while IFS= read -r rel; do
    [[ -n "${rel}" ]] || continue
    total=$((total + 1))
    path="${models_dir}/${rel}"
    need="$(vdps_model_min_bytes "${rel}")"
    if [[ ! -f "${path}" ]]; then
      echo "  缺失: ${rel}（需要 >= ${need} bytes）" >&2
      missing=$((missing + 1))
      continue
    fi
    size="$(wc -c < "${path}" | tr -d ' ')"
    if [[ "${size}" -lt "${need}" ]]; then
      echo "  不完整: ${rel}（${size} bytes < ${need} bytes，请重新 download）" >&2
      missing=$((missing + 1))
    fi
  done < <(vdps_each_required_model_weight)

  if [[ "${missing}" -ne 0 ]]; then
    echo "权重校验失败: ${missing}/${total} 项缺失或不完整（仅有空文件不够）" >&2
    echo "修复: ./scripts/download-model-weights.sh（会自动跳过完整项、重下不完整项）" >&2
    return 1
  fi
  echo "权重完整: ${total}/${total} 项通过体积校验"
  return 0
}

# 检查 .../localdata（其下须有 models/）
visual_dps_check_model_weights() {
  local base="${1:-}"
  if [[ "${VISUAL_DPS_CHECK_MODELS:-1}" == "0" ]]; then
    return 0
  fi
  if [[ -z "${base}" ]]; then
    echo "错误: visual_dps_check_model_weights 需要 localdata 目录路径" >&2
    return 2
  fi
  local models_dir="${base%/}/models"
  if [[ ! -d "${models_dir}" ]]; then
    echo "错误: 缺少目录 ${models_dir}" >&2
    visual_dps_print_required_model_files
    return 1
  fi
  _visual_dps_check_models_dir_impl "${models_dir}"
}

# 检查离线包 weights/ 或任意 models 树（路径即 models 根）
visual_dps_check_models_dir() {
  local models_dir="${1:-}"
  if [[ "${VISUAL_DPS_CHECK_MODELS:-1}" == "0" ]]; then
    return 0
  fi
  if [[ -z "${models_dir}" ]] || [[ ! -d "${models_dir}" ]]; then
    echo "错误: visual_dps_check_models_dir 需要 models 目录路径" >&2
    visual_dps_print_required_model_files
    return 1
  fi
  _visual_dps_check_models_dir_impl "${models_dir%/}"
}

visual_dps_print_required_model_files() {
  echo "必需文件（相对 localdata/models/，括号内为最小字节）:" >&2
  local rel need
  while IFS= read -r rel; do
    [[ -n "${rel}" ]] || continue
    need="$(vdps_model_min_bytes "${rel}")"
    echo "  ${rel}  (>= ${need})" >&2
  done < <(vdps_each_required_model_weight)
}
