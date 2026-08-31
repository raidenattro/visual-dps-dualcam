#!/bin/sh
# 为镜像内 conda（/opt/conda）配置清华源；无 conda 则跳过
set -eu
CONDARC="${1:-/opt/conda/.condarc}"
if [ ! -x /opt/conda/bin/conda ] && [ ! -x /opt/conda/bin/pip ]; then
  echo "conda-mirror: skip (no /opt/conda)"
  exit 0
fi
mkdir -p "$(dirname "${CONDARC}")"
cat > "${CONDARC}" <<'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
echo "conda-mirror: wrote ${CONDARC}"
