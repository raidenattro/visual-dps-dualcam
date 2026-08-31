# 变更日志

## 2026-05-24 — RTMPose-t ONNX 后端 + 摄像头配置修复 + 矩阵页

### 推理后端 `rtmpose_onnx`

- 新增 `RTMPoseOnnxBackend`：RTMDet-nano（人体框）+ RTMPose-t（COCO-17），ONNX Runtime CPU。
- `Dockerfile.inference-lite` 预置 ONNX 权重；`app_config.json` 增加 `rtmpose_onnx_*` 路径/URL。
- UI「推理模型」可选 RTMPose-T；lite 镜像路由保留用户所选 backend（不再强制改回 mediapipe）。

### 配置与保存

- `runtime_config_service`：`_ALLOWED_BACKENDS` 纳入 `rtmpose_onnx`（修复保存后丢失）。
- `normalize_camera_settings(..., strict=True)` + `camera_store` 非法值返回明确错误。

### 推理日志（容器 stdout）

| 阶段 | 输出 |
|------|------|
| 选后端 | `ℹ️ 推理后端: rtmpose_onnx` |
| 加载模型 | `🚀 正在加载 RTMDet-nano + RTMPose-t（ONNX）…` |
| 缺权重 | `⬇️ 正在下载 ONNX 模型包: <url>` |
| 就绪 | `✅ RTMPose-t ONNX 已就绪: det=… pose=…` |
| 主循环 | `ℹ️ 推理参数: … frame_rate=… pose_frame_interval=…`（`inference_service`） |

说明：det = 人体检测（bbox），pose = 17 点姿态；碰撞在 event-worker，不在推理容器。

### 前端

- 事件矩阵页：自适应列宽、单元格紧凑展示（货位编码为主）。

---

## 2026-05-24 — Docker Compose 交付 + MediaMTX 容器化 + 播放修复

### 部署与 Compose

- **默认 `docker compose up -d` 拉起**：`redis`、`mediamtx`、`visual-dps-ui`、`visual-dps-event-worker`（推理容器仍由 UI 按摄像头 `docker run`）。
- **MediaMTX** 使用官方镜像 `bluenviron/mediamtx:1.11.3`，配置由 `services/mediamtx_service.py` 根据 `camera_ips.json` 生成到 `localdata/mediamtx.yml`。
- 新增 `.env.example`、`docker-compose.override.example.yml`、`deploy/`（模板与 Docker daemon 代理示例）。
- 新增 `Dockerfile.event-worker`、`Dockerfile.inference-lite-gpu` 及对应 build 脚本。
- 构建加速：`docker/debian-mirror.sh`、`docker/pip-install.sh`、`docker/ubuntu-mirror.sh`。
- 移除宿主机 MediaMTX 安装脚本：`scripts/install-mediamtx.sh`、`scripts/mediamtx.yml`；保留并更新 MP4 推流脚本（指向 compose 内 RTSP）。

### 推理 / 事件管道（S1–S5）

- 推理容器瘦身：仅检测 + 17 点姿态 → Redis `pose:*`。
- 新增 `event_worker.py` 与 `services/event_engine/`：碰撞、报警、Java 回调。
- 新增 `pose_bus` / `event_bus` / `live_bus`：UI SSE 合并姿态与事件 overlay。
- 详见 [PIPELINE_SPLIT.md](./PIPELINE_SPLIT.md)。

### 监控页播放（HLS / WebRTC / MJPEG）

| 问题 | 原因 | 修复 |
|------|------|------|
| HLS `bufferAppendError` | MediaMTX 默认 Low-Latency HLS（fMP4）与 hls.js 不兼容 | 生成配置时设 `hlsVariant: mpegts` |
| HLS `manifestLoadError` | MediaMTX 重启后无 RTSP publisher | 需重新执行 MP4/摄像头推流 |
| WebRTC 信令 201 但黑屏 | 容器内 ICE 通告 172.18.x，浏览器不可达 | `webrtcIPsFromInterfaces: false` + `webrtcAdditionalHosts` + 映射 **8189/udp+tcp** |
| WebRTC SDP 仍含私网 IP | 代理未改写 Answer | `mediamtx_proxy` 将 RFC1918 地址改写为 `MEDIAMTX_PUBLIC_HOST` |
| MJPEG 无帧 | 容器内 RTSP 仍指向 `127.0.0.1:8554` | `camera_service` 改写为 `mediamtx:8554` |

前端：`usePreviewStream.js`（HLS `withCredentials`、WHEP `credentials: 'include'`、`video.play()`、ICE 失败提示）；`MonitorPreviewStage.jsx` 布局在 `onPlaying` 时刷新。

### 环境变量（`.env.example`）

- `MEDIAMTX_PUBLIC_HOST` — 浏览器访问 ICE/RTSP 的宿主机地址（默认 `127.0.0.1`）。
- `MEDIAMTX_WEBRTC_ICE_PORT` — ICE UDP/TCP 端口（默认 `8189`，须与 compose 端口映射一致）。

### 运维备忘

```bash
# 一键栈
cp .env.example .env && docker compose build && docker compose up -d

# 演示多路 MP4 推流（MediaMTX 重启后需重跑）
./scripts/start-mp4-rtsp-multi.sh cam1 cam2 cam3 cam4 cam5 cam6 cam7 cam8

# UI 代码或 services 变更后
docker compose build visual-dps-ui && docker compose up -d visual-dps-ui

# mediamtx.yml 变更后
docker compose restart mediamtx
```

### 文档

- [DEPLOY.md](./DEPLOY.md) — 交付步骤、播放与 WebRTC 排障。
- [PIPELINE_SPLIT.md](./PIPELINE_SPLIT.md) — Redis 契约与 event-worker。
- [WHEP_INFERENCE_STREAM.md](./WHEP_INFERENCE_STREAM.md) — 推理侧 WHEP 说明。
