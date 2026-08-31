#!/usr/bin/env bash
# 为 weights/ 目录生成 SHA256SUMS（相对路径与 model-weights-spec 一致）
set -euo pipefail

WEIGHTS_DIR="${1:-}"
if [[ -z "${WEIGHTS_DIR}" ]] || [[ ! -d "${WEIGHTS_DIR}" ]]; then
  echo "用法: $0 /path/to/weights" >&2
  exit 1
fi

_SPEC="$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model-weights-spec.sh"
# shellcheck source=deploy/model-weights-spec.sh
source "${_SPEC}"

MANIFEST="${WEIGHTS_DIR%/}/SHA256SUMS"
: > "${MANIFEST}"
while IFS= read -r rel; do
  [[ -n "${rel}" ]] || continue
  if [[ ! -f "${WEIGHTS_DIR}/${rel}" ]]; then
    echo "错误: 缺少 ${rel}" >&2
    exit 1
  fi
  (cd "${WEIGHTS_DIR}" && sha256sum "${rel}") >> "${MANIFEST}"
done < <(vdps_each_required_model_weight)

echo "已写入 ${MANIFEST} ($(wc -l < "${MANIFEST}") 条)"
