#!/bin/sh
# 构建期切换 Debian apt 源（由 Dockerfile ARG APT_MIRROR 传入）
set -eu
MIRROR="${1:-}"
if [ -z "${MIRROR}" ] || [ "${MIRROR}" = "deb.debian.org" ]; then
  exit 0
fi
if [ -f /etc/apt/sources.list.d/debian.sources ]; then
  sed -i "s|http://deb.debian.org|https://${MIRROR}|g; s|http://security.debian.org|https://${MIRROR}|g" \
    /etc/apt/sources.list.d/debian.sources
fi
if [ -f /etc/apt/sources.list ]; then
  sed -i "s|deb.debian.org|${MIRROR}|g; s|security.debian.org|${MIRROR}|g" /etc/apt/sources.list
fi
