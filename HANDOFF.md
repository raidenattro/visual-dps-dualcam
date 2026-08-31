# 合并与重构交接说明

本文档面向接手 `visual-dps` 的同事，说明本次从两个独立项目合并为单一服务的**思路、意图和原则**，以及后续工作的入口。

---

## 1. 为什么要合并

原先现场需要同时维护两个服务：

| 原项目 | 分支 | 职责 |
|--------|------|------|
| `box_human_det` | dev-backend | 人体检测 + 姿态 + 货框碰撞 + Java 回调 |
| `annotation_tool` | dev-multi | 摄像头/视频抓帧、多货架标注、保存 JSON |

两者**没有直接调用关系**，靠标注 JSON 文件路径手动对齐。现场部署两个进程、配两套端口，且推理端不支持 `annotation_tool` 的多货架 JSON 格式，容易配错、卡顿难调。

**合并目标：** 一个进程、一个配置文件、一套 API，标注结果直接进入推理，减少现场实施时间。

---

## 2. 合并策略（做了什么、没做什么）

### 做了

- 以 `box_human_det` 的 **FastAPI 分层结构** 为主干（`app_factory` + `services` + `core`）
- 将 `annotation_tool` 的后端能力**改写**为 FastAPI 路由（不是把 Flask 嵌进来）
- 保留两个前端 HTML，通过路由区分职责：
  - `/` → `frontend/index.html`（推理监控）
  - `/annotate` → `frontend/annotation_tool.html`（标注配置）
- 推理端增加对 `shelves[].boxes[]` 多货架 JSON 的兼容（见 `annotation_service.flatten_annotation_boxes`）
- 针对部署卡顿做了**少量性能重构**（见第 4 节）
- 配置项重命名为视频/推理领域常用术语（见第 5 节）

### 刻意没做

- **没有把两个 HTML 合成一个页面** — 职责不同，硬合并改动大、回归风险高
- **没有保留 Flask / `/api/shutdown`** — 会 kill 整个进程，不适合后台/容器部署
- **没有在本迭代做 UI 改配置** — 已写入 [ROADMAP.md](./ROADMAP.md)，是下一迭代重点
- **没有改模型选型与碰撞算法** — 业务逻辑与原先 `box_human_det` 一致

---

## 3. 架构意图

```
main.py
  └── app_factory.create_app()     # 路由注册、依赖组装
        ├── core/config.py         # 启动时加载 app_config.json（严格校验）
        ├── core/state.py          # 会话级运行态 STATE（视频源、json 路径等）
        └── services/
              ├── video_service.py       # 上传、转码、首帧
              ├── camera_service.py      # 摄像头 IP、抓帧（来自 annotation_tool）
              ├── annotation_service.py  # 标注 JSON 读写、多货架展平
              ├── inference_service.py   # 推理主循环（仅检测+姿态 → pose Redis）
              ├── mediamtx_service.py / mediamtx_proxy.py  # MediaMTX 配置与 HLS/WHEP 代理
              ├── pose_bus.py / event_bus.py / live_bus.py
              ├── event_engine/          # 碰撞等事件算法（event_worker 进程）
              └── callback_reporter.py   # 回调 Java（由 event_worker 启动）
```

**原则：**

1. **入口薄、服务厚** — 新 API 优先加到 `services/`，在 `app_factory.py` 挂路由
2. **配置与状态分离** — 静态配置在 `app_config.json`；当前会话（上传编号、流地址、json 路径）在 `STATE`
3. **推理不阻塞事件循环** — 读帧、检测、姿态放在 `ThreadPoolExecutor`（`inference_service._executor`）
4. **标注 → 推理只通过 JSON 文件** — 不引入标注与推理之间的额外 RPC

### 数据流（现场典型路径）

```
/annotate 抓帧或上传 → 保存 JSON（localdata/json/）
       ↓
app_config.json 中 source.annotation_json 指向该文件
       ↓
启动服务（流模式自动推理）或 / 页面手动 start_inference
       ↓
推理容器：检测/姿态 → `pose:live:*`；event-worker：读标注 JSON → 碰撞/报警 → `event:live:*` + 回调 Java；UI SSE 合并二者
```

---

## 4. 性能相关重构（为何这样改）

现场反馈部署后卡顿，排查后做了以下**最小改动**：

| 问题 | 原状 | 现做法 |
|------|------|--------|
| 后台推理仍编码 JPEG | 每帧 base64，即使不开预览 | 仅 `debug-info.enabled=true` 且 WebSocket 可视化时才编码 |
| 同步阻塞 asyncio | `cap.read()`、GPU 推理在主协程 | 放入 `_executor` 线程池 |
| 处理帧率无上限 | 尽量跑满源流 | `inference.frame_rate` 节流（默认 15 fps） |
| RTSP 旧帧堆积 | 默认缓冲 | `stream_buffer_size=1` |
| 检测/姿态双间隔难理解 | `det_interval` + `pose_interval` 叠加 | **每 processed 帧都检测**；仅姿态按 `pose_frame_interval` 跳帧 |

**姿态跳帧保留的原因：** 检测框短时稳定，姿态模型更耗 GPU，隔 N 帧跑一次是常见折中；中间帧复用上次骨架做碰撞判断。

---

## 5. 配置命名原则

配置键采用视频/推理管线常见术语，避免自造缩写：

| 字段 | 含义 |
|------|------|
| `video.transcode_height` | 上传转码高度（px） |
| `video.capture_height` | 标注抓帧高度（px） |
| `inference.frame_rate` | 推理处理帧率（fps） |
| `inference.height` | 推理输入高度（px，等比缩放） |
| `inference.pose_frame_interval` | 姿态估计帧间隔 |
| `inference.stream_buffer_size` | 网络流解码缓冲帧数 |

**原则：** 用 `height` / `frame_rate` / `frame_interval`，不用 `stream_target_height`、`det_interval` 等混合命名。

启动时由 `core/config.py` **严格校验**必填字段；缺字段直接 `SystemExit`，避免带病运行。

---

## 6. 运行模式说明

| 模式 | 触发条件 | 行为 |
|------|----------|------|
| 文件上传 | `POST /api/upload_video` | 转码 → 标注 → 手动 `start_inference` |
| 网络流 | `source.enabled=true` | 启动即后台推理（无需打开页面） |
| 可视化预览 | `debug-info.enabled=true` + 连接 `WS /ws/inference` | 推送 JPEG + 骨架；会显著增加负载 |
| 生产推荐 | `debug-info.enabled=false` | 仅后台推理 + Java 回调，性能最佳 |

流模式下：用户打开调试 WebSocket 会先停后台任务再切可视化；关闭页面后自动恢复后台推理（见 `inference_service.websocket_inference` 的 `finally` 块）。

---

## 7. 标注 JSON 兼容

推理读取 JSON 后，通过 `flatten_annotation_boxes()` 统一为 `boxes` 列表：

1. **新格式（优先）：** 顶层 `shelves[]`，每项含 `shelf_code` + `boxes[]`
2. **旧格式（兼容）：** 顶层直接 `boxes[]`（原 `box_human_det` 前端保存）

坐标映射仍优先 `video_polygon_norm` + `annotation_size`，与合并前 `box_human_det` 行为一致。

---

## 8. 仓库与分支现状

- 远程：**https://github.com/HQIT/visual-dps**（fork 自 `raidenattro/box_human_det`）
- `main`：已含合并提交 `99d90d1`
- PR #1（`feat/unified-visual-dps`）：补充 ROADMAP 文档，待 review merge

本地启动：

```bash
python main.py
# 默认 http://0.0.0.0:8045
# 部署前请按环境修改 app_config.json 中 models 路径
```

---

## 9. 后续工作入口（ROADMAP 摘要）

产品侧已确认的方向（详见 [ROADMAP.md](./ROADMAP.md)）：

1. **UI 可改配置，尽量热生效** — 新增 `GET/PATCH /api/runtime_config`，推理循环读内存值
2. **配置 UI 单页集中** — Settings 抽屉，监控页与标注页共用，不跨路由找配置
3. **现场配置项尽量少** — UI 只暴露约 5 项：`frame_rate`、`height`、`pose_frame_interval`、`stream_url`、预览开关；其余用默认值

建议实现顺序：

```
runtime_config API → inference 热读取 → 共用 Settings 前端组件 → 精简 UI 字段
```

---

## 10. 改动时请遵守的原则

1. **最小 diff** — 不顺手重构无关模块；碰撞/回调逻辑已稳定，慎动
2. **单一服务** — 不再引入第二个 HTTP 进程
3. **配置可预测** — 新配置项需有行业通用名 + `config.py` 校验 + README 说明
4. **现场优先** — 默认值为「能跑且省 GPU」；高级项不进 UI
5. **热生效范围** — 仅 `frame_rate` / `height` / `pose_frame_interval` / 预览开关等运行时参数；模型路径变更仍需重启

---

## 11. 关键文件速查

| 文件 | 关注什么 |
|------|----------|
| `app_factory.py` | 所有 HTTP/WS 路由 |
| `services/inference_service.py` | 推理主循环、性能、WebSocket |
| `services/annotation_service.py` | JSON 保存、多货架展平 |
| `services/camera_service.py` | 摄像头 IP、抓帧 |
| `core/config.py` | 配置 schema 校验 |
| `core/state.py` | 会话状态字段 |
| `app_config.json` | 部署参数（现场主要改这个，直到 UI 做好） |
| `frontend/annotation_tool.html` | 标注 UI（多货架、box_id 编辑） |
| `frontend/index.html` | 推理监控 UI |

如有疑问，可先对照原仓库：

- 推理逻辑来源：`raidenattro/box_human_det` @ `dev-backend`
- 标注逻辑来源：`raidenattro/annotation_tool` @ `dev-multi`
