# 后续计划：统一配置 UI

## 目标

1. **UI 可改配置，尽量热生效** — 修改后无需重启服务（模型路径等设备级配置除外）
2. **配置入口集中** — 单页 Settings，不分散在 `/` 与 `/annotate`
3. **配置项尽量少** — 现场实施只暴露高频项，高级项用默认值

## 现场暴露配置（建议 5 项）

| 配置 | 字段 | 热生效 | 说明 |
|------|------|--------|------|
| 推理帧率 | `inference.frame_rate` | 是 | 默认 15 fps |
| 推理高度 | `inference.height` | 是 | 默认 480 px |
| 姿态间隔 | `inference.pose_frame_interval` | 是 | 默认 3 帧 |
| 视频流地址 | `source.stream_url` | 是* | 变更后重连流 |
| 预览开关 | `debug-info.enabled` | 是 | 关闭可显著降负载 |

其余项（`stream_buffer_size`、`preview_jpeg_quality`、`capture_height` 等）保持配置文件默认值，不在 UI 暴露。

## 实现要点

- 新增 `GET/PATCH /api/runtime_config`，读写内存配置并持久化到 `app_config.json`
- 推理循环每帧读取内存配置（frame_rate / height / pose_frame_interval）
- 标注页与监控页顶部共用 Settings 抽屉，不再跨页找配置
- 标注相关 `capture_height` 与上传转码高度跟随 `inference.height`，减少重复配置


## 讨论（已实现：见 [docs/PIPELINE_SPLIT.md](./docs/PIPELINE_SPLIT.md)）

1. **17 点姿态** — 每路独立推理容器，发布 `pose:live:{camera_id}`
2. **动作/事件** — 默认 1 个 `visual-dps-event-worker`；可按 `EVENT_WORKER_SHARD_COUNT/INDEX` 起 N 个（按摄像头分片，见 PIPELINE_SPLIT.md）
3. **UI** — `LiveHub` 合并 pose + event → SSE；推理容器不再做碰撞/回调

部署：`docker compose --profile ui up` 含 redis + ui + event-worker；按摄像头 `inference/start` 起推理容器。

## 新需求

1. **事件矩阵页（已实现）** — 导航「事件矩阵」`/matrix`，API `GET /api/matrix/overview`，按摄像头分组展示货架网格与碰撞/告警状态（约 1.5s 轮询）。
   - **已排查（2026-05-24）**：`event-worker` 本身在跑；根因是 **推理镜像过旧 + 投递方式不一致**：
     - 旧推理镜像的 `pose_bus` 只 `PUBLISH pose:live:*`，不写 `pose:stream`；
     - `event-worker` 配置 `POSE_DELIVERY=stream`，只消费 `pose:stream` → 无 `event:*`，监控页无碰撞/告警。
   - **已修复**：
     - 重建 `visual-dps-inference-lite`，启动推理容器时注入 `POSE_DELIVERY=stream` 等环境变量；
     - 修复 `_OpencvCaptureAdapter` 缺少 `get()` 导致 `RTSP_CAPTURE_BACKEND=opencv` 时推理秒退；
     - 验证：`pose:stream` 有写入，`event:snapshot:cam2` 正常更新（cam2 需有 RTSP 源与货框标注）。
   - **备注**：cam5/7/8 等若无货框标注仍不会有区域事件；无 RTSP 推流的摄像头推理容器会退出，需 `./scripts/start-mp4-rtsp-multi.sh` 或实机推流。

2. **服务拓扑页（已实现）** — 导航「服务拓扑」`/topology`，API `GET /api/topology/overview`，展示推流/MTX/推理/Redis/event-worker 链路与地址一致性（约 2.5s 轮询）。契约见 [`docs/TOPOLOGY_API.md`](docs/TOPOLOGY_API.md)。

3. **已澄清：「开启检测」到底开了什么？**（总览/监控页「开启智能检测」→ `POST /api/cameras/{id}/inference/start`）
   - **会新增**：每路一个推理容器 `visual-dps-infer-{camera_id}`（标签 `visual-dps.role=inference`），进程读 RTSP → 检测 + 17 点姿态 → Redis `pose:live:*`；不在 `docker compose` 常驻服务表里，需用 `docker ps --filter "label=visual-dps.role=inference"` 查看。
   - **不会新增**：`visual-dps-event-worker`（compose 已常驻，读姿态 + 标注做碰撞/报警/Java 回调）、redis、mediamtx、ui。
   - **与预览无关**：HLS/WebRTC 不依赖检测开关；骨架/碰撞 overlay 需该路推理在跑且 event-worker 正常。
   - **若看不到容器**：启动失败看 UI 报错；先构建推理镜像；UI 需挂 `/var/run/docker.sock`；秒退查 `docker ps -a` 与 `localdata/inference/{id}.status.json`。
   - **旧路径**：上传视频的 `POST /api/start_inference` 为进程内推理，不起 Docker 容器（与当前按路容器模式不同）。详见 [docs/DEPLOY.md](./docs/DEPLOY.md)、[docs/PIPELINE_SPLIT.md](./docs/PIPELINE_SPLIT.md)。

4. ~~监控界面上的“开启检测”开关，旁边默认显示一横，表示全局没有配模型？但实际上全局配好了的，有些摄像头有问题，不全是。另外摄像头个性配置中选择了指定的算法，监控界面上也不会变，很奇怪！需要核实并解决。~~ 似乎已经解决了，现在的问题就是更新话覆盖模型后，如果启动失败会自动fallback到全局默认。

4. **CPU 姿态模型（已实现 RTMPose-t ONNX）** — 后端名 `rtmpose_onnx`（RTMDet-nano + RTMPose-t，ONNX Runtime，COCO-17）；UI「推理模型」可选；需 `visual-dps-inference-lite` 镜像。调研备选（未集成）：RTMPose-s、MoveNet Lightning。

5. 需要生成一个svg Logo，作为本系统的Logo；本系统名字DiDPS（意思是深度智能DPS，Digital Picking System（数字拣选系统））