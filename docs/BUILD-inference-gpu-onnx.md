# 构建 `inference-lite-gpu-onnx` 镜像

GPU 推理推荐镜像：`visual-dps-inference-lite-gpu-onnx:<日期-tag>`。

## 快速命令

```bash
# 1. 先有基底（仅需一次）
./scripts/build-inference-lite-gpu-image.sh

# 2. 构建 gpu-onnx（自动打日期 tag，写入构建日志）
./scripts/build-inference-lite-gpu-onnx-image.sh

# 3. 构建后校验（有 GPU 的机器）
./scripts/verify-gpu-onnx-image.sh visual-dps-inference-lite-gpu-onnx:<tag>

# 4. 部署前探测 PyTorch 镜像站（可选，构建脚本内也会探测）
./scripts/probe-torch-mirror.sh
```

`.env` 示例：

```bash
PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
TORCH_INDEX=https://mirror.sjtu.edu.cn/pytorch-wheels/cu121
INFERENCE_LITE_GPU_ONNX_IMAGE=visual-dps-inference-lite-gpu-onnx:20260529-165748-5c6b720
```

## 镜像里锁定的栈（勿随意升级）

| 组件 | 版本 | 说明 |
|------|------|------|
| 基底 | `nvidia/cuda:12.1.1-runtime` | 与 4090 + 驱动 525+ 兼容 |
| torch / torchvision | `2.5.1` / `0.20.1`，**cu121** | 禁止 pip 拉到 `cu130` |
| onnxruntime-gpu | `1.20.2` + `[cuda,cudnn]` | 与 CUDA 12 + cuDNN 9 配套 |
| ultralytics | `8.4.0`，`--no-deps`（YOLO26 需 ≥8.4） | 避免升级 torch |

构建结束日志应出现：

```text
torch 2.5.1+cu121 cuda 12.1
onnxruntime 1.20.2 ... CUDAExecutionProvider ...
```

## 153 上 GPU 不行的真实原因（不是驱动版本）

旧镜像内 **混了两套 cuDNN**（`torch 2.12+cu130` + ORT 的 `cudnn-cu12`）→ `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` → 推理回退 CPU。

**修法**：换按上文锁栈构建的新镜像，不是去升 153 的 NVIDIA 驱动。

## 构建流程（Dockerfile 内）

1. `docker/probe-torch-index.sh`：探测含 `torch-2.5.1+cu121` 的索引（默认 **上海交大**，备用 official）。
2. `wget -c` 直拉 torch / torchvision wheel（有进度，约 30s/780MB）。
3. `pip install --no-deps` 本地 wheel，**nvidia-*-cu12 依赖从清华 PyPI 装**（勿用「清华主源 + extra 交大」装 torch，会静默拉数 GB 且无进度）。
4. 装 ORT、ultralytics（`--no-deps`），卸载 `nvidia-*-cu13` 残留。

## 镜像站探测结论（2026-05，须先测再用）

| URL | torch 2.5.1+cu121 |
|-----|-------------------|
| `mirror.sjtu.edu.cn/pytorch-wheels/cu121` | ✅（302 → `s3.jcloud.sjtu.edu.cn`） |
| `download.pytorch.org/whl/cu121` | ✅（慢，兜底） |
| `mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu121` | ❌ 404 |
| `mirrors.aliyun.com/pytorch-wheels/cu121/torch/` | ❌ 404 |

改 `TORCH_INDEX` 前在本机跑：`./scripts/probe-torch-mirror.sh`。

## 禁止项（避免再浪费一下午）

1. **未探测**就写死 PyTorch 镜像 URL。
2. `pip install torch` 用 **清华作主源 + extra 交大**（依赖走清华，日志只显示 torch）。
3. 在 `docker build` 里注入 **WSL `172.26` 代理** → `host.docker.internal` 解析失败。
4. 让 **ultralytics 默认依赖** 升级 torch 到 cu130。
5. 第二次 **force-reinstall torch** 且 `--no-cache-dir`（重复下 3GB）。

## 部署到目标机（全量离线）

构建校验通过后，走全量离线包（含本镜像），勿单独 save/load：

```bash
./scripts/export-offline-one-shot.sh
# 详见 docs/OFFLINE-DEPLOY-CHECKLIST.md
```

## 相关文件

| 路径 | 作用 |
|------|------|
| `Dockerfile.inference-lite-gpu-onnx` | 镜像定义 |
| `docker/install-gpu-onnx-pip.sh` | 构建时 pip/wget |
| `docker/probe-torch-index.sh` | 容器内探测索引 |
| `scripts/probe-torch-mirror.sh` | 宿主机探测 |
| `scripts/build-inference-lite-gpu-onnx-image.sh` | 构建入口 |
| `scripts/verify-gpu-onnx-image.sh` | 构建后校验 |
| `deploy/153.env.example` | 153 环境变量示例 |
