# event-worker-2 已移除

本仓不再提供 pick_state / `visual-dps-event-worker-2`。

碰撞只跑双路 3D：`visual-dps-event-worker` → `event_worker.py` → `DualcamRedisWorker`（`Dockerfile.event-worker`）。

水平扩展用同一 3D 镜像的 `visual-dps-event-worker-b`（compose profile `worker-dual`），不要再构建或启动 `event-worker-2`。
