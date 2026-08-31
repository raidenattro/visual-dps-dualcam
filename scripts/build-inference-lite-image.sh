#!/usr/bin/env bash
# 本仓已取消 CPU inference-lite。现场只跑 gpu-onnx。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "本仓不再构建 visual-dps-inference-lite（CPU）。请改跑: $ROOT/scripts/build-inference-lite-gpu-onnx-image.sh" >&2
exec "$ROOT/scripts/build-inference-lite-gpu-onnx-image.sh" "$@"
