#!/usr/bin/env bash
# 已并入 3D event-worker：请构建 visual-dps-event-worker，不要再打 worker-2 镜像。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "本仓已取消 pick_state worker-2。请改跑: $ROOT/scripts/build-event-worker-image.sh" >&2
exec "$ROOT/scripts/build-event-worker-image.sh" "$@"
