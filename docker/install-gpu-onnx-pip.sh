#!/bin/sh
# gpu-onnx：大文件 wget 直拉（有进度）；torch 依赖仍走 PyTorch 镜像站 pip
set -eu

PIP_INDEX="${1:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
ORT_GPU_VERSION="${ORT_GPU_VERSION:-1.20.2}"
ULTRALYTICS_VERSION="${ULTRALYTICS_VERSION:-8.4.0}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

TORCH_INDEX="$(sh /tmp/probe-torch-index.sh)"
echo "==> TORCH_INDEX=${TORCH_INDEX}"

pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true

mkdir -p /tmp/wheels
TORCH_FILE="torch-${TORCH_VERSION}+cu121-cp310-cp310-linux_x86_64.whl"
TV_FILE="torchvision-${TORCHVISION_VERSION}+cu121-cp310-cp310-linux_x86_64.whl"
TORCH_URL="${TORCH_INDEX}/torch-${TORCH_VERSION}%2Bcu121-cp310-cp310-linux_x86_64.whl"
TV_URL="${TORCH_INDEX}/torchvision-${TORCHVISION_VERSION}%2Bcu121-cp310-cp310-linux_x86_64.whl"

echo "==> [1/4] wget ${TORCH_FILE}"
wget -c --progress=dot:giga -O "/tmp/wheels/${TORCH_FILE}" "${TORCH_URL}"
echo "==> [1/4] wget ${TV_FILE}"
wget -c --progress=dot:giga -O "/tmp/wheels/${TV_FILE}" "${TV_URL}"

echo "==> [1/4] pip install wheels + 依赖（大包已 wget，依赖走清华）"
pip install --no-deps "/tmp/wheels/${TORCH_FILE}" "/tmp/wheels/${TV_FILE}"
pip install -i "${PIP_INDEX}" \
    filelock networkx jinja2 fsspec sympy pillow \
    nvidia-cuda-nvrtc-cu12==12.1.105 \
    nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cuda-cupti-cu12==12.1.105 \
    nvidia-cudnn-cu12==9.1.0.70 \
    nvidia-cublas-cu12==12.1.3.1 \
    nvidia-cufft-cu12==11.0.2.54 \
    nvidia-curand-cu12==10.3.2.106 \
    nvidia-cusolver-cu12==11.4.5.107 \
    nvidia-cusparse-cu12==12.1.0.106 \
    nvidia-nccl-cu12==2.21.5 \
    nvidia-nvtx-cu12==12.1.105 \
    triton==3.1.0

echo "==> [2/4] onnxruntime-gpu"
pip install -i "${PIP_INDEX}" "onnxruntime-gpu[cuda,cudnn]==${ORT_GPU_VERSION}"

echo "==> [3/4] ultralytics"
pip install -i "${PIP_INDEX}" --no-deps "ultralytics==${ULTRALYTICS_VERSION}"
pip install -i "${PIP_INDEX}" \
    matplotlib pyyaml scipy tqdm requests pillow psutil py-cpuinfo onnx \
    seaborn ultralytics-thop

pip uninstall -y onnxruntime 2>/dev/null || true

for pkg in $(pip list --format=freeze | sed -n 's/^\(nvidia-.*-cu13\)==.*/\1/p'); do
    pip uninstall -y "${pkg}" 2>/dev/null || true
done

echo "==> [4/4] 校验"
python3 -c "
import importlib.metadata as m
import onnxruntime as ort
import torch
for name in ('nvidia-cudnn-cu13', 'nvidia-cuda-runtime-cu13'):
    try:
        m.version(name)
        raise SystemExit(f'残留 {name}')
    except m.PackageNotFoundError:
        pass
assert str(torch.version.cuda).startswith('12'), torch.version.cuda
print('onnxruntime', ort.__version__, ort.get_available_providers())
print('torch', torch.__version__, 'cuda', torch.version.cuda)
"
