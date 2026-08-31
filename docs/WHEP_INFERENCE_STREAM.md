# 推理取流方案：WHEP 与 Redis/SSE 对接

> 现状：浏览器预览走 **WebRTC（WHEP）**；推理容器走 **RTSP（OpenCV）**。  
> 短期已做 **RTSP 低延迟** 试水（`services/rtsp_capture.py`）。  
> 本文描述中期目标：**推理也经 WHEP 取帧**，并与现有 **Redis + SSE** overlay 衔接。

---

## 1. 为什么要改

| 问题 | 说明 |
|------|------|
| 双连接不同步 | WebRTC 预览与 RTSP 推理各订一路，骨架叠在「最新画面」上易滞后 |
| RTSP 客户端积缓冲 | OpenCV/FFmpeg 默认排队旧帧，算力跟不上时延迟放大 |
| MJPEG 不适合推理 | 带宽/CPU 大，仅作浏览器备选预览，**推理输入不应使用 MJPEG** |

`pose_frame_interval` 只降低姿态 **更新频率**，不能当作允许 **时间延迟** 的理由。

---

## 2. 目标架构（WHEP 阶段）

```text
                    MediaMTX (path=cam2)
                           │
           ┌───────────────┼───────────────┐
           │               │               │
      WHEP 订阅 A      WHEP 订阅 B     RTSP 发布源
      (浏览器)         (推理 Worker)    (ffmpeg/摄像头)
           │               │
      <video> WebRTC    解码帧 + timestamp
                           │
                    检测 / 姿态（仅姿态帧）
                           │
                    Redis Pub/Sub + 快照
                           │
                    UI LiveHub → SSE /live/stream
                           │
                    浏览器 SVG 骨架（与 WebRTC 画面叠加）
```

**原则**

- 预览与推理尽量 **同协议（WHEP）**、同出口（MediaMTX path）。
- Overlay 仍走 **Redis + SSE**（已落地），不在此路径传 JPEG。
- 每帧 LiveFrame 带 **`captured_at`**（及可选 `frame_id`），为 P1 坐标归一化与时间对齐打基础。

---

## 3. 推理 Worker：WHEP 订阅设计

### 3.1 信令与 URL

与前端一致，经 UI 同源代理（容器内可直连 MediaMTX）：

- 浏览器：`POST /api/cameras/{id}/whep`（SDP offer）→ MediaMTX `/{path}/whep`
- 推理：`POST http://mediamtx:8889/{path}/whep` 或 `http://host.docker.internal:8889/{path}/whep`

环境变量建议：

| 变量 | 含义 |
|------|------|
| `INFERENCE_STREAM_MODE` | `rtsp`（默认）\| `whep` |
| `MEDIAMTX_WEBRTC_BASE` | 与 UI 一致，如 `http://host.docker.internal:8889` |
| `INFERENCE_MTX_PATH` | 如 `cam2`（或由 camera_id 映射） |

### 3.2 Python 实现选型

| 方案 | 优点 | 缺点 |
|------|------|------|
| **aiortc** | 纯 Python、易嵌入 asyncio 推理循环 | 镜像体积增大，需维护重连 |
| **GStreamer webrtcbin** | 性能好 | 依赖重，Docker 镜像复杂 |
| **FFmpeg + WHIP/WHEP** | 命令行成熟 | 与现有 asyncio 循环集成较绕 |

**推荐**：推理镜像内 **aiortc** + 小型 `WhepFrameReader`：

```text
WhepFrameReader.start()
  → POST offer SDP → answer SDP
  → on_track(video) → 最新帧队列 (maxsize=1，满则丢旧)
WhepFrameReader.read_latest() → (ndarray BGR, captured_at)
```

与当前 `read_latest_frame(cap)` **接口对齐**，推理主循环可切换：

```python
if stream_mode == "whep":
    frame, ts = whep_reader.read_latest()
else:
    ok, frame, ts = read_latest_frame(cap)
```

### 3.3 与姿态帧节奏的关系

保持现有策略（与 RTSP 低延迟一致）：

- **仅姿态帧**：检测 + 姿态 + `publish_live_frame`
- **非姿态帧**：不检测、不发布（headless）；或仅 drain 等待下一姿态节拍
- 节拍仍由 `frame_rate` 与 `pose_frame_interval` 决定，但输入帧来自 WHEP「最新帧」队列

---

## 4. 与 Redis / SSE 的衔接（不变更协议，只 enrich）

### 4.1 发布侧（推理 Worker）

仍在姿态帧调用 `publish_live_frame()`，建议扩展 LiveFrame（P1）：

```json
{
  "schema": 2,
  "ts": 1716543210.12,
  "captured_at": 1716543210.05,
  "source": "whep",
  "infer_width": 640,
  "infer_height": 480,
  "skeletons": [...],
  "collisions": [...],
  "alarm_collisions": []
}
```

- `ts`：发布时刻（Redis 写入时间）
- `captured_at`：WHEP/解码拿到帧的时刻（用于衡量 **端到端延迟**）
- `source`：`rtsp` | `whep` 便于排查

Redis 操作不变：

- `PUBLISH inference:live:{camera_id}`
- `SET inference:snapshot:{camera_id}` TTL 10s

### 4.2 消费侧（UI 进程）

不变：

- 启动时 `LiveHub` 订阅 `inference:live:*`
- `GET /api/cameras/{id}/live/stream` → SSE `event: frame`

可选增强（P1+）：

- SSE 首包 snapshot 后，前端用 `captured_at` 与 `performance.now()` 显示「骨架延迟 xx ms」（调试开关）
- 不做复杂 A/V sync 时，仍靠 **降低 captured_at → 显示 的差值** 验收

### 4.3 与 WebRTC 预览的关系

- **不合并** 为一路流：预览继续 WebRTC，骨架继续 SSE 透明叠加（保留「骨架/ROI 开关」）。
- WHEP 推理与浏览器 WHEP **同源同 path**，缩短「两路 RTSP/WebRTC 时间差」，但不保证像素级同帧；严格同帧需 `frame_id` 或服务端画骨架（远期）。

---

## 5. 分阶段落地

| 阶段 | 内容 | 状态 |
|------|------|------|
| **P0** | Redis + SSE；姿态帧发布 | 已完成 |
| **P0.5** | RTSP 低延迟：`open_rtsp_capture` / `read_latest_frame` / headless 仅姿态帧检测 | 已完成 |
| **P1** | LiveFrame 归一化坐标 + `captured_at` | 待做 |
| **P2** | status 文件仅生命周期 | 待做 |
| **P3** | `INFERENCE_STREAM_MODE=whep`；`WhepFrameReader`；推理镜像 aiortc | 待做 |
| **P4** | 可选：单 path 帧广播服务（一路解码，推理 + 指标共用） | 远期 |

---

## 6. 部署与回退

- 默认 `INFERENCE_STREAM_MODE=rtsp`，行为与现网一致（含低延迟优化）。
- 设 `whep` 时需保证推理容器能访问 MediaMTX WebRTC 端口（与 UI 相同 `visual-dps-internal` / `host.docker.internal:8889`）。
- 回退：环境变量改回 `rtsp`，重启推理容器即可。

---

## 7. 验收建议（WHEP 阶段）

1. 浏览器 WebRTC 预览流畅，SSE `live/stream` 持续 `frame` 事件。
2. 对比 `source=whep` 与 `rtsp`：`captured_at` 到显示延迟中位数下降（同机同 path）。
3. 骨架与人体明显错位时，查 Redis 快照里 `captured_at` 与推理日志是否积压。

---

## 8. 相关文件

| 文件 | 职责 |
|------|------|
| `services/rtsp_capture.py` | RTSP 低延迟采帧 |
| `services/live_bus.py` | Redis 发布 / SSE fan-out |
| `services/inference_service.py` | 推理主循环、姿态帧发布 |
| `web/src/hooks/usePreviewStream.js` | 浏览器 WHEP |
| `services/mediamtx_proxy.py` | UI 代理 WHEP |
