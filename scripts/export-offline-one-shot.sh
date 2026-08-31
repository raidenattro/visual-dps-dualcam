#!/usr/bin/env bash
# 源机「一把过」：权重 → 构建全部镜像 → 校验 gpu-onnx → 预检 → 打全量离线包
# 用法: ./scripts/export-offline-one-shot.sh [export-offline-package 额外参数...]
# 示例: ./scripts/export-offline-one-shot.sh -o dist/my-pkg
#       ./scripts/export-offline-one-shot.sh --rebuild-ui
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/scripts/lib/load-build-env.sh"
load_build_env "${ROOT}"

echo "========================================"
echo " Visual-DPS 离线包一键流程（源机）"
echo "========================================"

echo ""
echo "==> [1/6] 推理权重"
# shellcheck disable=SC1091
source "${ROOT}/deploy/check-model-weights.sh"
if visual_dps_check_model_weights "${ROOT}/localdata" 2>/dev/null; then
  echo "    已有完整权重，跳过 download"
else
  "${ROOT}/scripts/download-model-weights.sh"
  visual_dps_check_model_weights "${ROOT}/localdata"
fi

echo ""
echo "==> [2/6] UI + Event 镜像"
"${ROOT}/scripts/build-ui-image.sh"

echo ""
echo "==> [3/6] 推理镜像（lite / lite-gpu / lite-gpu-onnx）"
"${ROOT}/scripts/build-inference-lite-image.sh"
"${ROOT}/scripts/build-inference-lite-gpu-image.sh"
"${ROOT}/scripts/build-inference-lite-gpu-onnx-image.sh"

set -a
# shellcheck disable=SC1090
[[ -f "${ROOT}/.env" ]] && source "${ROOT}/.env"
set +a

ONNX_REF="${INFERENCE_LITE_GPU_ONNX_IMAGE:-}"
if [[ -z "${ONNX_REF}" ]] || ! docker image inspect "${ONNX_REF}" >/dev/null 2>&1; then
  ONNX_REF="$(docker images --format '{{.Repository}}:{{.Tag}}' visual-dps-inference-lite-gpu-onnx \
    | grep -v ':<none>' | grep -v ':latest' | sort | tail -n 1)"
fi
[[ -n "${ONNX_REF}" ]] || { echo "错误: 无法解析 gpu-onnx 镜像 tag" >&2; exit 1; }

echo ""
echo "==> [4/6] 校验 gpu-onnx: ${ONNX_REF}"
"${ROOT}/scripts/verify-gpu-onnx-image.sh" "${ONNX_REF}"

echo ""
echo "==> [5/6] 打包预检"
"${ROOT}/scripts/preflight-offline-export.sh" --inference all

echo ""
echo "==> [6/6] 导出全量离线包"
"${ROOT}/scripts/export-offline-package.sh" --inference all "$@"

PKG="$(ls -dt "${ROOT}"/dist/visual-dps-offline-complete-* 2>/dev/null | head -1)"
echo ""
echo "========================================"
echo " 完成"
if [[ -n "${PKG}" ]]; then
  echo " 包目录: ${PKG}"
  echo ""
  echo " 内网分发（推荐，勿默认 tar.gz 整包）:"
  echo "   rsync -av --progress ${PKG}/ hqit@192.168.1.153:~/workspace/visual-dps/"
  echo ""
  echo " 目标机:"
  echo "   cd $(basename "${PKG}") && ./verify-package.sh"
  echo "   编辑 app/.env 的 REDIS_PASSWORD、MEDIAMTX_PUBLIC_HOST、端口"
  echo "   ./install.sh --host <IP> --stop-infer"
fi
echo "========================================"
