#!/bin/sh
# 使用国内 PyPI 镜像安装依赖：/docker/pip-install.sh <index-url> pkg1 pkg2 ...
set -eu
INDEX="${1:-}"
shift || true
if [ -z "${INDEX}" ]; then
  pip install "$@"
else
  pip install -i "${INDEX}" "$@"
fi
