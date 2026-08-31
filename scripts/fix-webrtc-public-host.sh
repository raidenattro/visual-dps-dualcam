#!/usr/bin/env bash
# 修正 WebRTC：MEDIAMTX_PUBLIC_HOST + mediamtx.yml webrtcAdditionalHosts，并重启 mediamtx / UI
# 用法:
#   ./scripts/fix-webrtc-public-host.sh
#   ./scripts/fix-webrtc-public-host.sh 192.168.0.204
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

HOST="${1:-}"
if [[ -z "${HOST}" ]]; then
  HOST="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
if [[ -z "${HOST}" ]]; then
  echo "请指定宿主机 IP: $0 192.168.0.204"
  exit 1
fi

ENV_FILE="${ROOT}/.env"
MTX_YML="${ROOT}/localdata/mediamtx.yml"

set_env() {
  local key="$1"
  local val="$2"
  if [[ -f "${ENV_FILE}" ]]; then
    if grep -q "^${key}=" "${ENV_FILE}"; then
      sed -i "s|^${key}=.*|${key}=${val}|" "${ENV_FILE}"
    else
      echo "${key}=${val}" >>"${ENV_FILE}"
    fi
  fi
}

echo "==> MEDIAMTX_PUBLIC_HOST=${HOST}"
set_env MEDIAMTX_PUBLIC_HOST "${HOST}"
set_env MEDIAMTX_WEBRTC_ICE_PORT "${MEDIAMTX_WEBRTC_ICE_PORT:-8189}"

if [[ -f "${MTX_YML}" ]]; then
  if grep -q '^webrtcAdditionalHosts:' "${MTX_YML}"; then
    sed -i "s|^webrtcAdditionalHosts:.*|webrtcAdditionalHosts: ['${HOST}']|" "${MTX_YML}"
  else
    echo "警告: ${MTX_YML} 无 webrtcAdditionalHosts 行，请在 UI 中「应用 MediaMTX 配置」"
  fi
  grep -E 'webrtcAdditionalHosts|webrtcLocal' "${MTX_YML}" || true
fi

export VISUAL_DPS_IMAGE_TAG="${VISUAL_DPS_IMAGE_TAG:-latest}"
echo "==> 重启 mediamtx、visual-dps-ui（需已 docker compose up）"
docker compose restart mediamtx visual-dps-ui

echo ""
echo "完成。浏览器请用: http://${HOST}:$(grep -E '^UI_PORT=' "${ENV_FILE}" 2>/dev/null | cut -d= -f2 || echo 8045)/"
echo "防火墙放行: TCP 8045 8889 8189，UDP 8189"
echo "WebRTC 仍失败时可先选 HLS/MJPEG。"
