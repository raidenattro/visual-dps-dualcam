# 全量离线部署（新机 / 整栈恢复）

唯一推荐流程：源机打全量包 → rsync 目录 → 目标机 `install.sh`。

## 1. 源机

```bash
cd <repo-root>
./scripts/export-offline-one-shot.sh
```

一条命令完成：权重 → 构建 UI + 三档推理镜像 → 校验 gpu-onnx → 预检 → 打 `visual-dps-offline-complete-<时间戳>/`。

镜像已齐、仅重打离线包时：

```bash
./scripts/preflight-offline-export.sh --inference all
./scripts/export-offline-complete.sh
```

打包前 `.env` 建议：

| 变量 | 要求 |
|------|------|
| `INFERENCE_LITE_GPU_ONNX_IMAGE` | 构建后写入的**日期 tag**（勿长期仅用 `latest`） |
| `VISUAL_DPS_IMAGE_TAG` | 与 UI/Event 一致 |
| `TORCH_INDEX` | 默认交大 cu121；改前先 `./scripts/probe-torch-mirror.sh` |

gpu-onnx 构建细则：[BUILD-inference-gpu-onnx.md](BUILD-inference-gpu-onnx.md)

## 2. 分发

```bash
rsync -av --progress dist/visual-dps-offline-complete-*/ hqit@192.168.1.153:~/workspace/visual-dps/
```

- 直接同步**目录**，勿对 `bundle.tar` 再 gzip。
- 勿默认 `tar -czf` 整包（慢）；外传才用 `--archive tar.gz --compress pigz`。
- 第二实例端口见 `deploy/153.env.example`。

## 3. 目标机

```bash
cd visual-dps-offline-complete-*/
./verify-package.sh
# 必改 app/.env：REDIS_PASSWORD、MEDIAMTX_PUBLIC_HOST、UI/流媒体端口
./install.sh --host 192.168.1.153 --stop-infer
```

`install.sh` 会：`docker load` → 校验 gpu-onnx（`INFERENCE_USE_GPU=1`）→ 安装权重 → 生成 `mediamtx.yml` → `compose up`。

## 4. 验收

1. `http://<host>:<UI_PORT>/` 可开，页脚版本正常。
2. 推流就绪后启动各 cam 推理。
3. `docker logs visual-dps-infer-cam*`：`device=cuda` / YOLO `device=0`，无 `Pose26`、`SUBLIBRARY_VERSION_MISMATCH`。

## 排错

| 现象 | 处理 |
|------|------|
| 打包报权重不全 | `./scripts/download-model-weights.sh` |
| 打包报 gpu-onnx 失败 | [BUILD-inference-gpu-onnx.md](BUILD-inference-gpu-onnx.md) |
| install 后推理 CPU / Pose26 | 源机重建 gpu-onnx 后**重打全量包**再 rsync |
| WebRTC 不可用 | `camera_ips.json` 端口与 `.env` 一致；`regenerate-mediamtx-config.sh` |

相关：`.cursor/skills/visual-dps-offline-package/SKILL.md`、`deploy/OFFLINE-QUICKSTART.md`
