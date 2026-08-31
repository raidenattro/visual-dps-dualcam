# 交接：visual-dps-dualcam-exp

更新：2026-08-31

从 `visual-dps` 拷出的双路 3D 实验仓。碰撞只用 `contact_slots`（贴墙即报）。

- 路径：`/home/hqit/workspace/visual-dps-dualcam-exp`
- 标注：`/aisle` 勾选同一组 → 墙四角 → 反解 → 层线。未成组、未反解、无层线 **禁止开推理**。
- Worker：`visual-dps-event-worker` + `-b`（默认两实例，shard 0–7 / 8–15）→ `event_worker.py` → `DualcamRedisWorker`，分片键 `aisle_id`。

全局标定分辨率、巷道 AABB、相机先验在设置页「双路 3D 几何」，单巷道/单路可在巷道标注页或摄像头抽屉覆盖。
