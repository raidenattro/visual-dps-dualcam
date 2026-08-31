#!/bin/sh
# 构建前探测：返回第一个含 torch==$TORCH_VERSION+cu121 的 PyTorch wheel 索引 URL
set -eu

TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
WHEEL="torch-${TORCH_VERSION}%2Bcu121-cp310-cp310-linux_x86_64.whl"
WHEEL_PLAIN="torch-${TORCH_VERSION}+cu121-cp310-cp310-linux_x86_64.whl"

try_index() {
    base="$1"
    url="${base%/}/torch/"
    code=$(wget -q -O /dev/null -T 15 --spider "$url" 2>/dev/null && echo 200 || echo fail)
    [ "$code" = "200" ] || return 1
    if wget -q -O - -T 20 "$url" 2>/dev/null | grep -qF "${WHEEL_PLAIN}"; then
        echo "$base"
        return 0
    fi
    return 1
}

for base in \
    "${TORCH_INDEX:-}" \
    "https://mirror.sjtu.edu.cn/pytorch-wheels/cu121" \
    "https://download.pytorch.org/whl/cu121"; do
    [ -n "$base" ] || continue
    if picked=$(try_index "$base"); then
        echo "$picked"
        exit 0
    fi
done

echo "ERROR: 无可用 PyTorch cu121 镜像（已试 SJTU / official）" >&2
exit 1
