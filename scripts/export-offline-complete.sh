#!/usr/bin/env bash
# 全量离线包（镜像已齐时）：7 镜像 + weights/，默认目录输出
# 源机构建+打包请用: ./scripts/export-offline-one-shot.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "${ROOT}/scripts/export-offline-package.sh" --inference all "$@"
