#!/usr/bin/env bash
# 离线包 dry 校验（不 install、不 load）
# 用法: ./deploy/verify-package.sh [包根目录]
set -euo pipefail

PKG_ROOT="$(cd "${1:-.}" && pwd)"
FAIL=0
QUICK="${VERIFY_QUICK:-0}"

note_fail() { echo "FAIL: $*" >&2; FAIL=1; }
note_ok() { echo "OK: $*"; }

echo "==> 校验离线包: ${PKG_ROOT}"

for f in install.sh docker-images/bundle.tar app/docker-compose.deploy.yml app/.env app/app_config.json OFFLINE-QUICKSTART.md; do
  [[ -f "${PKG_ROOT}/${f}" ]] && note_ok "${f}" || note_fail "缺少 ${f}"
done

if [[ -f "${PKG_ROOT}/docker-images/bundle.tar" ]]; then
  if tar -tf "${PKG_ROOT}/docker-images/bundle.tar" repositories >/dev/null 2>&1; then
    note_ok "bundle.tar (docker save)"
    if [[ "${QUICK}" != "1" ]]; then
      repos_json="$(tar -xOf "${PKG_ROOT}/docker-images/bundle.tar" repositories 2>/dev/null || true)"
      if [[ -n "${repos_json}" ]]; then
        echo "--- 镜像列表 ---"
        echo "${repos_json}" | python3 -m json.tool 2>/dev/null || echo "${repos_json}"
        if [[ -f "${PKG_ROOT}/PACKAGE_INFO.txt" ]] && grep -q 'inference_mode: all' "${PKG_ROOT}/PACKAGE_INFO.txt"; then
          for need in visual-dps-inference-lite visual-dps-inference-lite-gpu visual-dps-inference-lite-gpu-onnx; do
            if echo "${repos_json}" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if '${need}' in d else 1)"; then
              note_ok "complete 包含 ${need}"
            else
              note_fail "complete 包缺少 ${need}"
            fi
          done
        fi
      fi
    fi
  else
    note_fail "bundle.tar 无法读取 repositories"
  fi
fi

# shellcheck disable=SC1091
source "${PKG_ROOT}/app/deploy/check-model-weights.sh" 2>/dev/null \
  || source "${PKG_ROOT}/deploy/check-model-weights.sh"

if [[ -d "${PKG_ROOT}/weights" ]]; then
  note_ok "weights/ 目录"
  visual_dps_check_models_dir "${PKG_ROOT}/weights" || note_fail "weights 不完整"
  [[ -f "${PKG_ROOT}/weights/SHA256SUMS" ]] && note_ok "weights/SHA256SUMS" || note_fail "缺少 weights/SHA256SUMS"
elif [[ -d "${PKG_ROOT}/app/localdata/models" ]]; then
  note_ok "旧布局 app/localdata/models"
  visual_dps_check_model_weights "${PKG_ROOT}/app/localdata" || note_fail "models 不完整"
else
  note_fail "缺少 weights/ 或 app/localdata/models"
fi

[[ -f "${PKG_ROOT}/app/deploy/model-weights-spec.sh" ]] && note_ok "model-weights-spec.sh" || note_fail "缺少 model-weights-spec.sh"

if [[ "${FAIL}" -ne 0 ]]; then
  echo "校验未通过" >&2
  exit 1
fi
echo "==> 全部校验通过"
