# 服务拓扑（页面与架构图）

- **路由**：`/topology`（导航与「事件矩阵」平级）
- **API 契约**：[`TOPOLOGY_API.md`](./TOPOLOGY_API.md)
- **JSON Schema**：[`schemas/topology-overview.schema.json`](./schemas/topology-overview.schema.json)
- **示例响应**：[`schemas/topology-overview.example.json`](./schemas/topology-overview.example.json)

---

## 全局拓扑（compose 默认）

```mermaid
flowchart LR
  subgraph host["宿主机 / 现场"]
    SRC["流媒体源<br/>ffmpeg · 摄像头 · 外部 RTSP"]
  end

  subgraph compose["Docker · visual-dps-internal"]
    MTX["MediaMTX<br/>mediamtx:8554<br/>API :9997"]
    UI["visual-dps-ui<br/>:8045"]
    REDIS[("Redis<br/>redis:6379")]
    EW["event-worker<br/>visual-dps-event-worker"]
    INF1["infer-cam*<br/>docker run / 标签 inference"]
  end

  subgraph client["浏览器"]
    BR["监控 / 矩阵 / 拓扑页"]
  end

  SRC -->|"push RTSP<br/>PUBLISH"| MTX
  INF1 -->|"pull RTSP<br/>PLAY"| MTX
  UI -->|"pull HLS/WHEP<br/>代理"| MTX
  BR -->|"HTTP SSE / API"| UI

  INF1 -->|"XADD pose:stream<br/>redis_stream"| REDIS
  REDIS -->|"XREADGROUP event-workers<br/>redis_stream"| EW
  EW -.->|"event:snapshot / live"| REDIS
  UI -.->|"读 snapshot / SSE"| REDIS

  UI -->|"Docker API<br/>启停 infer"| INF1
```

---

## 单路摄像头数据流（publisher 模式）

```mermaid
flowchart TB
  subgraph cfg["配置层"]
    C1["camera.url<br/>rtsp://127.0.0.1:8554/cam2"]
    C2["annotation camera_url"]
    C3["infer INFERENCE_STREAM_URL<br/>rtsp://mediamtx:8554/cam2"]
  end

  SRC["源 push"] -->|"rtsp://host:8554/cam2"| MTX["MTX path=cam2<br/>ready / readers"]
  MTX -->|"pull"| INF["infer 容器"]
  INF -->|"姿态"| R[("pose:stream")]
  R --> EW["event-worker"]
  EW --> EV["碰撞 / 告警 / 回调"]

  C1 -.->|"运维应对齐"| SRC
  C3 -.-> INF
```

**健康判定要点**

| 检查项 | 正常 | 异常示例 |
|--------|------|----------|
| MTX `ready` | `true` | `false` → 无推流 |
| infer `stream_url` probe | 可达 | 404 → 空转无骨架 |
| 配置 URL 与 infer URL | 同一 MTX 实例 | 推到 `192.168.x.x`，拉 `mediamtx` |
| infer → Redis | XADD 有增量 | worker 无 pose 输入 |
| pose 新鲜度 | `last_ts_age_sec` ≤ 阈值且未 `frozen` | `INFER_POSE_FROZEN`：有流但关键点不随帧变 |
| pose 停更 | MTX 有流但 snapshot 过期 | `POSE_STALE`：infer 未再 publish |

---

## 故障态示例（本次踩坑）

```mermaid
flowchart LR
  FF["ffmpeg<br/>192.168.0.204:8554/cam2"] -->|"push ✓"| MTX_LAN["MTX @ LAN<br/>有流"]
  INF["infer-cam2"] -->|"pull ✗ 404"| MTX_DC["MTX @ compose<br/>mediamtx:8554<br/>ready=false"]

  style MTX_LAN fill:#2d6a4f,color:#fff
  style MTX_DC fill:#9d0208,color:#fff
  style INF fill:#9d0208,color:#fff
```

拓扑页应对 `INFER_STREAM_MISMATCH` 高亮：**绿** = 有流实例，**红** = infer 实际拉流。

---

## 页面线框

主区域为全宽拓扑图；**链路详情**在右侧抽屉打开（与摄像头设置抽屉同款）。触发：**仅**点击拓扑图中的源/推理节点（连线不可点）。

- 节点色：`health` → ok 绿 / warn 黄 / error 红 / unknown 灰
- 拓扑图：**始终显示全部**节点与连线；点选某路时仅加粗该路连线（其它线变淡），节点不变灰；关抽屉或再点同一节点取消选择
- 抽屉「链路连接」：仅当前选中摄像头相关边；点具体节点时收窄为与该节点相连的边
- 抽屉其余块：`paths[]` 问题码 / 配置 / 运行时

---

## 与现有文档关系

| 文档 | 关系 |
|------|------|
| [`PIPELINE_SPLIT.md`](./PIPELINE_SPLIT.md) | Redis pose/event 契约 |
| [`USER_MANUAL.md`](./USER_MANUAL.md) § FFmpeg 推流 | 源侧 push 操作 |
| [`TOPOLOGY_API.md`](./TOPOLOGY_API.md) | 本页后端接口 |
