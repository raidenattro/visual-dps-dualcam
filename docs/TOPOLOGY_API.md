# 服务拓扑 API 契约

> 页面路由：`/topology`（与「事件矩阵」`/matrix` 平级）  
> 实现状态：**已实现** — `GET /api/topology/overview`，页面 `/topology`

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/topology/overview` | 聚合拓扑：节点、边、每路摄像头链路、全局告警 |

### 查询参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `probe` | `boolean` | `false` | 是否对 infer 的 `stream_url` 做 ffprobe 探测（8 路约 6s；勾选「RTSP 探测」或 `probe=true`） |
| `include_logs_hint` | `boolean` | `false` | 是否在 `issues[].hint` 附带容器日志末行（仅 error 节点） |

### 鉴权

与现有 `/api/*` 一致（`auth_middleware`）；未登录返回 `401`。

### 响应 Envelope

```json
{
  "status": "success",
  "generated_at": 1710000000.0,
  "poll_recommended_ms": 2500,
  "graph": { "nodes": [], "edges": [] },
  "paths": [],
  "issues": [],
  "capabilities": {
    "docker": true,
    "mediamtx_api": true,
    "redis_info": false
  }
}
```

失败时：

```json
{
  "status": "error",
  "message": "无法连接 MediaMTX API: …",
  "generated_at": 1710000000.0
}
```

---

## 图模型（前端直接渲染）

前端使用 **有向图**：`graph.nodes` + `graph.edges`。节点用 `id` 唯一标识；边用 `from` / `to` 引用节点 `id`。

### 节点 `TopologyNode`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | `string` | ✓ | 稳定 ID，如 `mtx`、`redis`、`infer:cam2`、`source:cam2` |
| `kind` | `string` | ✓ | 见下表 `NodeKind` |
| `label` | `string` | ✓ | 展示名，如 `MediaMTX`、`infer-cam2` |
| `health` | `string` | ✓ | `ok` \| `warn` \| `error` \| `unknown` |
| `host` | `string` | | 主机名 / 容器名 / `external` |
| `hostname` | `string` | | DNS 名（compose 服务名或容器 hostname） |
| `ip` | `string` | | 容器 IP；外部源可为空 |
| `ports` | `PortBinding[]` | | 端口列表 |
| `meta` | `object` | | 种类相关扩展字段 |

**`NodeKind`**

| 值 | 含义 |
|----|------|
| `source` | 流媒体源（摄像头 / 宿主机 ffmpeg / 外部 RTSP） |
| `mediamtx` | MediaMTX 实例 |
| `mediamtx_path` | MTX 单路 path（可选：与 `mediamtx` 合并为一层时不用） |
| `inference` | 推理容器 |
| `redis` | Redis |
| `event_worker` | 事件 Worker |
| `ui` | visual-dps-ui（可选，表示 API/代理） |
| `client` | 浏览器预览（可选） |

**`PortBinding`**

```json
{ "name": "rtsp", "protocol": "tcp", "host_port": 8554, "container_port": 8554, "bind": "0.0.0.0" }
```

`name` 枚举建议：`rtsp` | `api` | `hls` | `webrtc` | `redis` | `http` | `other`。

---

### 边 `TopologyEdge`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | `string` | ✓ | 稳定 ID，如 `e:cam2:source->mtx` |
| `from` | `string` | ✓ | 源节点 `id` |
| `to` | `string` | ✓ | 宿节点 `id` |
| `direction` | `string` | ✓ | `push` \| `pull` |
| `protocol` | `string` | ✓ | `rtsp` \| `redis_stream` \| `redis_pubsub` \| `http` \| `docker` |
| `role` | `string` | | `publish` \| `play` \| `pose` \| `event` \| `preview` \| `config` |
| `endpoint` | `string` | ✓ | 人类可读端点，如 `rtsp://mediamtx:8554/cam2` |
| `health` | `string` | ✓ | `ok` \| `warn` \| `error` \| `unknown` |
| `bytes_per_sec` | `number` | | 可选，来自 MTX `bytesReceived` 差分 |
| `meta` | `object` | | 扩展 |

---

## 每路链路 `TopologyPath`（按摄像头）

与 `graph` 互补：表格/侧栏用结构化字段，避免前端从图反查。

| 字段 | 类型 | 说明 |
|------|------|------|
| `camera_id` | `string` | 摄像头 ID |
| `camera_name` | `string` | 显示名 |
| `source_type` | `string` | `publisher` \| `rtsp_pull` \| `external` |
| `configured` | `object` | 配置侧地址，见 `ConfiguredEndpoints` |
| `runtime` | `object` | 运行时实测，见 `RuntimeEndpoints` |
| `mediamtx` | `object` | 见 `MediamtxPathStatus` |
| `inference` | `object` | 见 `InferencePathStatus` |
| `event` | `object` | 见 `EventPathStatus` |
| `health` | `string` | 该路汇总：`ok` \| `warn` \| `error` |
| `issues` | `string[]` | 该路 issue code 列表 |

### `ConfiguredEndpoints`

```json
{
  "playback_url": "rtsp://127.0.0.1:8554/cam2",
  "pull_url": "",
  "annotation_camera_url": "rtsp://127.0.0.1:8554/cam2",
  "mtx_path": "cam2",
  "hls": "/api/cameras/cam2/hls/index.m3u8",
  "whep": "/api/cameras/cam2/whep"
}
```

### `RuntimeEndpoints`

```json
{
  "infer_stream_url": "rtsp://mediamtx:8554/cam2",
  "infer_stream_probe": {
    "reachable": false,
    "latency_ms": null,
    "error": "DESCRIBE 404 Not Found"
  },
  "external_publish_hint": "rtsp://192.168.0.204:8554/cam2"
}
```

`external_publish_hint`：当检测到「配置/内部 MTX 无流，但其它 host:port 同名 path 有流」时填写（启发式，见实现说明）。

### `MediamtxPathStatus`

```json
{
  "path": "cam2",
  "ready": false,
  "available": false,
  "source": null,
  "readers": [],
  "bytes_received": 0,
  "bytes_sent": 0
}
```

`source` / `readers` 与 MediaMTX `GET /v3/paths/get/{name}` 对齐：

```json
{ "type": "rtspSession", "id": "…" }
```

### `InferencePathStatus`

与现有 `get_inference_status` 字段对齐并扩展：

```json
{
  "status": "running",
  "container_name": "visual-dps-infer-cam2",
  "container_id": "338975da9981",
  "docker_status": "running",
  "backend": "yolo26s_pose",
  "stream_url": "rtsp://mediamtx:8554/cam2",
  "message": "",
  "hostname": "338975da9981",
  "ip": "172.18.0.12",
  "gpu": {
    "requested": false,
    "available": false,
    "warning": "NVIDIA Driver was not detected"
  },
  "pose_publish": {
    "delivery": "stream",
    "stream_key": "pose:stream",
    "group": "event-workers"
  }
}
```

### `EventPathStatus`

```json
{
  "worker_container": "visual-dps-event-worker",
  "consumer_group": "event-workers",
  "consumer_name": "event-worker-1",
  "redis_url": "redis://:***@redis:6379/0",
  "last_pose_age_sec": null
}
```

`last_pose_age_sec`：P1，来自 Redis `XINFO STREAM pose:stream` 与最新 entry 时间差。

---

## 全局问题 `TopologyIssue`

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | `string` | 机器可读，见下表 |
| `severity` | `string` | `error` \| `warn` \| `info` |
| `camera_id` | `string` | 可空，全局问题无此字段 |
| `message` | `string` | 中文简述 |
| `hint` | `string` | 运维建议 |

**`code` 枚举（首批）**

| code | 典型场景 |
|------|----------|
| `MTX_PATH_NOT_READY` | path 无 publisher，`ready=false` |
| `INFER_STREAM_MISMATCH` | infer 拉的 URL 与「有流」实例不一致 |
| `INFER_NO_FRAMES` | infer running 但 `stream_url` 探测失败 |
| `INFER_POSE_FROZEN` | 有推流且 infer 在跑，但 stream 连续帧 `frame_idx` 增加而 `persons` 不变（缓冲旧帧） |
| `POSE_STALE` | infer 运行中但 Redis pose 快照超过 `TOPOLOGY_POSE_STALE_SEC`（默认 3s）未更新 |
| `INFER_GPU_UNAVAILABLE` | 容器无 GPU 但配置期望 GPU |
| `ANNOTATION_URL_MISMATCH` | `annotation camera_url` 与 `stream_url` 主机不一致 |
| `DOCKER_UNAVAILABLE` | UI 无法访问 docker.sock |
| `MTX_API_UNAVAILABLE` | 无法访问 `MEDIAMTX_API_URL` |
| `EVENT_WORKER_DOWN` | event-worker 非 running |
| `REDIS_UNREACHABLE` | Redis ping 失败 |

---

## 完整示例响应

见 [`schemas/topology-overview.example.json`](./schemas/topology-overview.example.json)。  
JSON Schema：[`schemas/topology-overview.schema.json`](./schemas/topology-overview.schema.json)。

---

## 后端聚合来源（实现对照）

| 数据 | 来源 |
|------|------|
| 摄像头配置 | `load_cameras` / `camera_ips.json` |
| MTX path 状态 | `GET {MEDIAMTX_API_URL}/v3/paths/list` |
| 推理容器 | Docker `label=visual-dps.role=inference` + `INFERENCE_*` env |
| Worker | Docker `container_name=visual-dps-event-worker` |
| Redis / MTX 静态节点 | compose 环境变量 |
| infer 流探测 | `ffprobe` / RTSP DESCRIBE（`probe=true`） |
| 地址一致性 | 对比 `normalize_rtsp_url`、path ready、probe 结果 |

---

## 前端约定

- 轮询间隔：使用响应中的 `poll_recommended_ms`（默认 `2500`）。
- 布局：分层 DAG（见 [`TOPOLOGY.md`](./TOPOLOGY.md)）；节点颜色 = `health`。
- 边标签：`direction` + `protocol` + `endpoint` 截断显示；hover 展示完整 `endpoint` 与 `meta`。

## 版本

| `schema` | 说明 |
|----------|------|
| `1` | 初版（本文档） |

响应根级可增加 `"schema": 1`（与 pose/event 帧一致）。
