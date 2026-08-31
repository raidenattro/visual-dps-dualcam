#!/bin/sh
# Ubuntu apt 镜像（GPU 推理镜像等）
set -eu
MIRROR="${1:-}"
if [ -z "${MIRROR}" ] || [ "${MIRROR}" = "archive.ubuntu.com" ]; then
  exit 0
fi
if [ -f /etc/apt/sources.list ]; then
  sed -i "s|archive.ubuntu.com|${MIRROR}|g; s|security.ubuntu.com|${MIRROR}|g" /etc/apt/sources.list
fi
if [ -d /etc/apt/sources.list.d ]; then
  sed -i "s|archive.ubuntu.com|${MIRROR}|g; s|security.ubuntu.com|${MIRROR}|g" /etc/apt/sources.list.d/*.list 2>/dev/null || true
fi
