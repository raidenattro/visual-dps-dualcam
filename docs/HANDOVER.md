# 交接：visual-dps-dualcam-exp

更新：2026-08-31

从 `visual-dps` 拷出的双路 3D 实验仓。碰撞用 pick-state 的 `contact_slots`（贴墙即报），不接 `pipeline.v5_gated`。

- 路径：`/home/hqit/workspace/visual-dps-dualcam-exp`
- 标注：`/aisle` 勾选同一组 → 墙四角 → 反解 → 层线。未成组禁止开推理。
- Worker：`event_worker_2.py` → `DualcamRedisWorker`，分片键 `aisle_id`。
