---
name: visual-dps-offline-package
description: >-
  Full offline deploy only: export-offline-one-shot.sh, install.sh, verify-package.sh.
  Use when the user mentions 离线包、全量离线、新机部署、整栈恢复、export-offline、
  install.sh、bundle.tar、或目标机无网恢复 Visual-DPS。
---

# Visual-DPS 全量离线打包与安装

仅支持 **新机 / 整栈恢复**（7 镜像 + weights）。入口：`./scripts/export-offline-one-shot.sh`。

## 包布局（v2，默认）

| 路径 | 内容 |
|------|------|
| `docker-images/bundle.tar` | `docker save` 全部镜像（**不要 gzip**） |
| `app/` | `docker-compose.deploy.yml`、`.env`、`app_config.json`、`localdata`（**无 models**） |
| `weights/` | 8 个推理权重 + `SHA256SUMS` |
| `install.sh` / `verify-package.sh` | 目标机安装与校验 |

全量通用包 = **7 镜像**（redis、mediamtx、ui、event-worker、inference-lite、lite-gpu、lite-gpu-onnx）。

## 源机：打全量包（推荐）

```bash
cd <repo-root>

# 一把过（权重 → 构建全部镜像 → 校验 gpu-onnx → 预检 → 打全量包）
./scripts/export-offline-one-shot.sh

# 或分步：
./scripts/download-model-weights.sh
./scripts/build-ui-image.sh
./scripts/build-inference-lite-gpu-onnx-image.sh
./scripts/verify-gpu-onnx-image.sh visual-dps-inference-lite-gpu-onnx:<tag>
./scripts/preflight-offline-export.sh --inference all
./scripts/export-offline-complete.sh
```

详见 `docs/OFFLINE-DEPLOY-CHECKLIST.md`。

输出：`dist/visual-dps-offline-complete-<YYYYMMDD-HHMMSS>/`

### 常用选项

| 选项 | 用途 |
|------|------|
| `--rebuild-ui` | 打包前重建 UI + event-worker 镜像 |
| `--archive tar` | 额外打未压缩 `.tar` 外壳（便于 U 盘） |
| `--archive tar.gz --compress pigz` | 外传用压缩包（**用 pigz，勿用默认 gzip 扫 14G**） |
| `--split 2G` | 对**外层归档**分卷 |
| `--inference lite\|gpu-onnx\|base\|all` | 控制打进 bundle 的推理镜像 |
| `--no-models` | 不打包 weights/ |
| `--allow-download-weights` | 源机不齐时：**先** `deploy/recover-model-weights.sh` 从 `dist/` 旧包拷贝，**再**联网补缺失项 |
| `FORCE_SAVE=1` | 忽略已有 `bundle.tar`，强制重做 `docker save` |
| `-o dist/my-pkg` | 指定输出目录 |

## 目标机：安装

```bash
cd <包根目录>
./verify-package.sh
# 必须修改 app/.env 中 REDIS_PASSWORD（不能用 change-me-before-install）
./install.sh --host <局域网IP> --stop-infer
```

| 选项 | 说明 |
|------|------|
| `--host IP` | 写入 `MEDIAMTX_PUBLIC_HOST` |
| `--weights-dir DIR` | 默认 `<包根>/weights`；旧包可用 `app/localdata/models` |
| `--stop-infer` | 删除旧 `visual-dps-infer-*` 容器 |

`install.sh` 在 `compose up` 前按 `.env` 生成 `app/localdata/mediamtx.yml`（不打包源机 yaml）。改 WebRTC/HLS 后：`app/deploy/regenerate-mediamtx-config.sh` 或 UI「应用 MediaMTX 配置」。

访问：`http://<MEDIAMTX_PUBLIC_HOST>:<UI_PORT>/`（默认 8045）

## 分发方式

| 场景 | 做法 |
|------|------|
| 内网 / rsync（**推荐**） | 直接同步**目录**，勿默认 `--archive --split`（多打一遍 tar 浪费时间） |
| U 盘单文件 ≤4G | 才用 `--archive tar --split 2G` |
| 带宽极有限 | `--archive tar.gz --compress pigz` |
```bash
# 153 测试机固定账号（勿用 xu@ / sugar@）
rsync -av --progress dist/visual-dps-offline-complete-*/ hqit@192.168.1.153:~/workspace/visual-dps-0529/
```

## 校验与排错

```bash
./verify-package.sh              # 包根执行
VERIFY_QUICK=1 ./verify-package.sh   # 跳过打印镜像 JSON

# 源机权重检查
source deploy/check-model-weights.sh
visual_dps_check_model_weights ./localdata
```

| 现象 | 处理 |
|------|------|
| export 报权重不全 | 源机 `./scripts/download-model-weights.sh` |
| 缺镜像 | 按 inference 模式 build 对应 Dockerfile 脚本 |
| complete 缺 lite 镜像 | 勿用 `.env` 的 `INFERENCE_LITE_IMAGE` 代替 lite；`all` 模式已固定收集三档 |
| install 报 REDIS_PASSWORD | 编辑 `app/.env` |
| WebRTC/HLS 不可用 | 确认 `camera_ips.json` 中 RTSP 端口与 `MEDIAMTX_RTSP_PORT` 一致；运行 `regenerate-mediamtx-config.sh` |
| 204 仅 docker-compose v1 | `install.sh` 已优先 `docker-compose.deploy.yml` |
| cam2 Pose26 / 推理 CPU | 源机重建 gpu-onnx 后重打全量包；见 `docs/BUILD-inference-gpu-onnx.md` |

## 相关文件

- `scripts/export-offline-one-shot.sh` — 源机一把过（推荐）
- `scripts/preflight-offline-export.sh` — 打包前预检
- `scripts/export-offline-package.sh` — 主入口（内嵌预检，`SKIP_PREFLIGHT=1` 可跳过）
- `scripts/export-offline-complete.sh` — 镜像已齐时仅重打离线包
- `deploy/verify-gpu-onnx-content.sh` — gpu-onnx 栈校验（构建/打包/install）
- `scripts/download-model-weights.sh` — 源机权重
- `deploy/install.sh`、`deploy/verify-package.sh`、`deploy/regenerate-mediamtx-config.sh`
- `deploy/model-weights-spec.sh`、`deploy/check-model-weights.sh`
- `deploy/PACKAGE-MANIFEST.md`、`deploy/OFFLINE-QUICKSTART.md`

## 性能原则（agent 须遵守）

1. **默认 `--archive none`**：产出目录，不对 14G `bundle.tar` 做 `tar -czf`。
2. **权重永远在 `weights/`**，不参与外层 gzip。
3. 只有用户明确要求外传压缩包时，才建议 `--archive tar.gz --compress pigz`。
4. `--rebuild-ui` 仅在有 UI/后端变更时使用（约 +10 分钟）。
