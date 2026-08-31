#!/usr/bin/env bash
# 宿主机探测 PyTorch cu121 镜像站（改 TORCH_INDEX 前必跑）
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/lib/load-build-env.sh
load_build_env .

TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
WHEEL_PLAIN="torch-${TORCH_VERSION}+cu121-cp310-cp310-linux_x86_64.whl"

probe() {
  local name="$1" base="$2"
  local url="${base%/}/torch/"
  printf '%-12s ' "$name"
  if ! curl -sfI --connect-timeout 8 "$url" >/dev/null 2>&1; then
    echo "FAIL (无响应)"
    return 1
  fi
  if curl -sf --connect-timeout 15 "$url" 2>/dev/null | grep -qF "$WHEEL_PLAIN"; then
    echo "OK  $base"
    return 0
  fi
  echo "FAIL (无 ${WHEEL_PLAIN})"
  return 1
}

echo "探测 torch ${TORCH_VERSION} cu121 (${WHEEL_PLAIN})"
echo ""

ok=0
probe "SJTU" "${TORCH_INDEX:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu121}" && ok=1
probe "official" "https://download.pytorch.org/whl/cu121" && ok=1
probe "tuna-pw" "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu121" || true
probe "aliyun-pw" "https://mirrors.aliyun.com/pytorch-wheels/cu121" || true

echo ""
if [[ "$ok" -eq 1 ]]; then
  echo "推荐 TORCH_INDEX（与 docker/probe-torch-index.sh 一致）:"
  sh docker/probe-torch-index.sh 2>/dev/null || true
  exit 0
fi
echo "错误: 无可用镜像站" >&2
exit 1
