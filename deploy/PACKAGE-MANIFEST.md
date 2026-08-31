# Visual-DPS 离线部署包清单

用于在**另一台主机**上一把拉起整套服务。原则：**镜像只含代码与依赖；所有大文件（ONNX/PT 权重、业务数据）均走 `localdata/` 卷挂载，不打进镜像**。

---

## 1. Docker 镜像一览

### 1.1 第三方镜像（`docker pull`，不构建）

| 镜像 | Tag（建议固定） | 用途 | Compose 服务 |
|------|-----------------|------|----------------|
| `redis` | `7` | Redis（姿态流、事件） | `redis` |
| `bluenviron/mediamtx` | `1.11.3` | RTSP / HLS / WebRTC | `mediamtx` |

### 1.2 自研镜像（需在本机构建后 `docker save` 带走）

| 仓库名（Repository） | 典型 Tag | 构建方式 | 运行时 |
|---------------------|----------|----------|--------|
| `visual-dps-visual-dps-ui` | `latest` | `docker compose build visual-dps-ui` 或 `./scripts/build-ui-image.sh` | **常驻** UI + API |
| `visual-dps-event-worker` | `latest` | 同上（与 UI 同次 build） | **常驻** 碰撞/告警 |
| `visual-dps-inference-lite` | `YYYYMMDD-HHMMSS-<git>` 或 `latest` | `./scripts/build-inference-lite-image.sh` | **按需** 推理（CPU） |
| `visual-dps-inference-lite-gpu` | 同上 | `./scripts/build-inference-lite-gpu-image.sh` | **按需** 推理（GPU 基底） |
| `visual-dps-inference-lite-gpu-onnx` | 同上 | `./scripts/build-inference-lite-gpu-onnx-image.sh`（需先有 lite-gpu；见 `docs/BUILD-inference-gpu-onnx.md`） | **按需** 推理（GPU + ONNX，推荐） |

说明：

- Compose 里 UI / Event 固定为 `:latest`。
- 推理镜像构建脚本会打**日期 tag**（`scripts/lib/docker-image-tag.sh`）；`.env` 可设 `DOCKER_TAG_ALSO_LATEST=1` 同时打 `latest`。
- 推理容器名：`visual-dps-infer-<camera_id>`（**不在** compose 里，由 UI 通过 Docker API 创建）。

### 1.3 按部署场景最少需要哪些镜像

| 场景 | 必需镜像 |
|------|----------|
| 仅 Web + 流媒体 + 事件 | `redis:7`、`bluenviron/mediamtx:1.11.3`、`visual-dps-visual-dps-ui:latest`、`visual-dps-event-worker:latest` |
| + CPU 智能检测 | 上表 + `visual-dps-inference-lite:latest`（或指定日期 tag） |
| + GPU 智能检测 | 上表 + `visual-dps-inference-lite-gpu-onnx:<tag>`（无则回退 `lite-gpu` / `lite`） |

---

## 2. 端口与网络

### 2.1 宿主机端口（`.env` 可调）

| 变量 | 默认 | 说明 |
|------|------|------|
| `UI_PORT` | `8045` | Web / API |
| `MEDIAMTX_RTSP_PORT` | `8554` | RTSP |
| `MEDIAMTX_HLS_PORT` | `8888` | HLS |
| `MEDIAMTX_WEBRTC_PORT` | `8889` | WebRTC 信令 |
| `MEDIAMTX_WEBRTC_ICE_PORT` | `8189` | WebRTC ICE（**UDP+TCP** 都要映射） |

### 2.2 Docker 网络

| 名称 | 用途 |
|------|------|
| `visual-dps-internal` | compose 默认网络；UI 设 `DOCKER_NETWORK=visual-dps-internal`，推理容器加入同一网络 |

---

## 3. 卷挂载与宿主机路径（必带目录）

### 3.1 Compose 常驻服务

| 服务 | 宿主机路径 | 容器路径 | 模式 |
|------|------------|----------|------|
| **visual-dps-ui** | `./app_config.json` | `/app/app_config.json` | ro |
| | `./localdata` | `/app/localdata` | rw |
| | `/var/run/docker.sock` | `/var/run/docker.sock` | rw（用于拉起推理容器） |
| **visual-dps-event-worker** | `./app_config.json` | `/app/app_config.json` | ro |
| | `./localdata` | `/app/localdata` | rw |
| **mediamtx** | `./localdata/mediamtx.yml` | `/mediamtx.yml` | ro |
| **redis** | （无绑定卷，数据在容器内；生产可改持久化） | | |

`HOST_PROJECT_ROOT`：compose 中为 **`${PWD}`**（项目根目录绝对路径），推理容器挂载依赖此项。

### 3.2 推理容器（UI 动态 `docker run`）

| 宿主机路径 | 容器路径 | 模式 |
|------------|----------|------|
| `<项目根>/localdata` | `/app/localdata` | rw |
| `<项目根>/app_config.json` | `/app/app_config.json` | ro |

**模型权重应放在** `localdata/models/`（随卷进容器），**无需**打进推理镜像。  
**注意**：仅有文件不够，须通过**体积下限**校验（`deploy/model-weights-spec.sh`）；半截下载会被视为不完整并触发重下或阻断 export/install。

---

## 4. 部署包目录结构（建议）

```text
visual-dps-package/
├── docker-images/                    # docker save 导出的 tar（可选分卷）
│   ├── redis-7.tar
│   ├── mediamtx-1.11.3.tar
│   ├── visual-dps-ui-latest.tar
│   ├── visual-dps-event-worker-latest.tar
│   └── visual-dps-inference-lite-latest.tar   # 按场景增减
├── app/                              # 从仓库拷贝（或 git clone）
│   ├── docker-compose.yml
│   ├── docker-compose.override.yml   # 可选
│   ├── .env                          # 从 .env.example / deploy/153.env.example 改
│   ├── app_config.json
│   ├── version.json
│   ├── localdata/
│   │   ├── camera_ips.json           # 必填：摄像头列表
│   │   ├── mediamtx.yml              # 可由 UI 生成；无则参考 deploy/mediamtx.yml.template
│   │   ├── runtime_config.json       # 可选：全局设置覆盖
│   │   ├── auth_config.json          # 启用登录时
│   │   ├── auth_users.json
│   │   ├── json/
│   │   │   ├── precise_boxes_new.json
│   │   │   └── cameras/<id>.json     # 每路标注
│   │   ├── models/                   # 推荐随包携带（volume 加载）
│   │   │   ├── rtmpose_onnx/
│   │   │   │   ├── rtmdet_nano/end2end.onnx
│   │   │   │   ├── rtmpose_t/end2end.onnx
│   │   │   │   ├── rtmpose_s/end2end.onnx   # 可选档位
│   │   │   │   └── rtmpose_m/end2end.onnx
│   │   │   └── yolo_pose/
│   │   │       ├── yolo26n-pose.pt
│   │   │       ├── yolo26s-pose.pt
│   │   │       ├── yolo26m-pose.pt
│   │   │       └── yolo26l-pose.pt
│   │   ├── logs/                     # 可空目录
│   │   ├── inference/                # 运行时状态，可空
│   │   └── frames/                   # 缩略图，可空
│   └── deploy/
│       └── PACKAGE-MANIFEST.md         # 本文件
└── install.sh                          # 可选：load 镜像 + compose up
```

---

## 5. 关键配置文件

| 文件 | 是否必填 | 说明 |
|------|----------|------|
| `.env` | **是** | `REDIS_PASSWORD`、`UI_PORT`、`MEDIAMTX_PUBLIC_HOST`、`INFERENCE_USE_GPU`、推理镜像名等 |
| `app_config.json` | **是** | 全局路径、默认模型、推理参数 |
| `localdata/camera_ips.json` | **是** | 摄像头与流类型（`publisher` / `rtsp_pull` / `external`） |
| `localdata/mediamtx.yml` | 安装时生成 | 离线包**不**携带源机文件；`install.sh` / `deploy/regenerate-mediamtx-config.sh` 按 `.env` 生成 |
| `localdata/runtime_config.json` | 否 | 系统设置页保存的全局覆盖 |
| `localdata/auth_*.json` | 视需求 | 本地登录 |

### 5.1 `.env` 与推理镜像相关项

```bash
REDIS_PASSWORD=<强密码>
UI_PORT=8045
MEDIAMTX_PUBLIC_HOST=<浏览器访问的 IP>   # 本机 127.0.0.1；生产填服务器 IP

INFERENCE_USE_GPU=0|1
INFERENCE_LITE_IMAGE=visual-dps-inference-lite:latest
INFERENCE_LITE_GPU_IMAGE=visual-dps-inference-lite-gpu:latest
INFERENCE_LITE_GPU_ONNX_IMAGE=visual-dps-inference-lite-gpu-onnx:<日期-tag>   # GPU 推荐

# 构建用（离线包仅 load 镜像时可忽略）
GITHUB_PROXY_BASE=https://ghfast.top
APT_MIRROR=mirrors.aliyun.com
PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 6. 模型文件（volume，非镜像内）

| 后端 preset | 宿主机路径 |
|-------------|------------|
| `rtmpose_t` / `s` / `m` | `localdata/models/rtmpose_onnx/rtmdet_nano/end2end.onnx` + `rtmpose_{t,s,m}/end2end.onnx` |
| `yolo26n_pose` … `yolo26l_pose` | `localdata/models/yolo_pose/yolo26{n,s,m,l}-pose.pt` |

预下载（源机器；**已有文件自动跳过**；`.env` 中 `GITHUB_PROXY_BASE` / `OPENMMLAB_MIRROR_BASE` / `BUILD_HTTP_PROXY` 加速）：

```bash
./scripts/download-model-weights.sh
# 或分别：
# ./scripts/download-rtmpose-onnx-weights.sh
# ./scripts/download-yolo-pose-weights.sh
```

缺文件时推理容器仍会尝试从 OpenMMLab/GitHub 在线拉取（不推荐离线环境依赖此项）。

---

## 7. 导出 / 导入（v2 布局）

**源机器**

```bash
./scripts/download-model-weights.sh

# 全量通用包（默认目录，不 gzip 整包）
./scripts/export-offline-complete.sh

# 外传: --archive tar.gz --compress pigz
# 仅 Web+事件: --inference base --no-models
```

详见 `.cursor/skills/visual-dps-offline-package/SKILL.md`。

**目标机器**

```bash
cd visual-dps-offline-complete-*/
./verify-package.sh
./install.sh --host <IP> --stop-infer
```

`install.sh` 在启动前会生成 `app/localdata/mediamtx.yml`。改 `MEDIAMTX_PUBLIC_HOST` / 流媒体端口后需在 UI「应用 MediaMTX 配置」或运行 `app/deploy/regenerate-mediamtx-config.sh`。

`weights/` → `app/localdata/models/`。旧版单 tar.gz 包仍兼容。

---

## 8. 目标机一键启动检查表

- [ ] 已安装 Docker + Compose v2
- [ ] 已 `docker load` 全部必需镜像（`docker images` 核对上表）
- [ ] 项目根含 `.env`、`app_config.json`、`localdata/camera_ips.json`
- [ ] `weights/` 或 `localdata/models/` 权重齐全（`./verify-package.sh`）
- [ ] `MEDIAMTX_PUBLIC_HOST` 与防火墙已放行 8045、8554、8888、8889、8189
- [ ] UI 容器能访问 `/var/run/docker.sock`（开启检测时需要）
- [ ] 在项目根执行：`docker compose up -d`
- [ ] 浏览器打开 `http://<MEDIAMTX_PUBLIC_HOST>:<UI_PORT>/`
- [ ] （可选）总览页「开启检测」前确认推理镜像 tag 与 `.env` 一致

---

## 9. 构建与打包命令索引（源机器）

| 产物 | 命令 |
|------|------|
| UI + Event | `./scripts/build-ui-image.sh` |
| CPU 推理 | `./scripts/build-inference-lite-image.sh` |
| GPU 推理（基底） | `./scripts/build-inference-lite-gpu-image.sh` |
| GPU + ONNX 推理 | `./scripts/build-inference-lite-gpu-onnx-image.sh` |
| **全量离线（推荐）** | `./scripts/export-offline-one-shot.sh` |
| **仅重打离线包** | `./scripts/export-offline-complete.sh`（镜像已齐时） |
| **打包前预检** | `./scripts/preflight-offline-export.sh --inference all` |
| **权重清单** | `./deploy/generate-weights-manifest.sh weights/` |
| **包内校验** | `./verify-package.sh`（打包末自动执行） |
| **Skill** | `.cursor/skills/visual-dps-offline-package/SKILL.md` |

---

## 10. 版本记录

打包时建议写入：

| 项 | 值 |
|----|-----|
| Git 提交 | `git rev-parse HEAD` |
| 镜像 tag | `docker images \| grep visual-dps` |
| 打包日期 | `date -Iseconds` |

---

相关文档：[DEPLOY.md](../docs/DEPLOY.md)、[AGENTS.md](../AGENTS.md)、`.env.example`、`deploy/153.env.example`。
