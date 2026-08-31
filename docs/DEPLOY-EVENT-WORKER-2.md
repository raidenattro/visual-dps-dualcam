# Event Worker：双路 3D（本仓唯一事件进程）

本仓 **只跑** `visual-dps-event-worker` → `DualcamRedisWorker`（`contact_slots` 贴墙即报）。

历史上 visual-dps 有两套事件进程，名字容易混：

| 旧称呼 | 容器 / 入口 | 算法 | 本仓 |
|--------|-------------|------|------|
| worker-1 | `visual-dps-event-worker` + `event_worker.py` | 每路独立 **2D** 多边形碰撞 | **已改为 3D**，入口转到 `event_worker_2.py` |
| worker-2 | `visual-dps-event-worker-2` + pick_state `v5_gated` | 单路 gated 打分 | **已删除默认服务**，禁止再启 |

`docker compose up -d` 只起 redis / mediamtx / ui / **一个 3D event-worker**。  
可选第二实例：`docker compose --profile worker-dual up -d visual-dps-event-worker-b`（同一 3D 镜像，按 shard 切分）。

**硬性约定：不要再跑 pick_state worker-2，也不要和任何旧 2D worker 抢 `event-workers` 消费组。**

## 启动

```bash
cd /home/hqit/workspace/visual-dps-dualcam-exp
docker compose up -d redis mediamtx visual-dps-ui visual-dps-event-worker
docker logs visual-dps-event-worker --tail 20
# 期望：Event worker-2 已启动 ... dualcam-3d ...
```

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `POSE_DELIVERY` | `stream` | Redis Stream |
| `POSE_STREAM_GROUP` | `event-workers` | 消费组 |
| `EVENT_WORKER_ENABLE_CALLBACKS` | `1` | Java 回调（JSON 契约不变，只由左路触发） |

`PICK_STATE_CONFIG` **无效**，请勿再配。
