# visual-dps-dualcam-exp

双路 3D 拣货实验仓。从 `visual-dps` 拷出，接入 pick-state 的巷道四角反解 + `contact_slots`。

## 硬性约束

1. **不改** `/home/hqit/workspace/visual-dps` 与 `visual-dps-pick-state`（后者只读拷代码）
2. 碰撞是 **3D `contact_slots`**（贴墙即报），不接 `pipeline.v5_gated`
3. **Dualcam**：标注必须勾选 **同一组**（左/右路）；**Legacy 单路**：未成组 + `json/cameras/<path>.json`（如 `aisle1-L.json`、`cam1.json`，与 `camera_ips.path` 一致）货框，走 2D 碰撞
4. 成组后分片键是 `aisle_id`，L/R 同一 dualcam worker；单路走 `event_worker_legacy.py`（Redis group `event-workers-legacy`）
5. 产物写本仓 `localdata/`；不走 153 离线包

## 关键路径

- 几何：`dualcam/solve.py`、`dualcam/geom.py`、`dualcam/lift.py`
- 成组/标定：`services/aisle_store.py`、`services/aisle_routes.py`
- Worker：`event_worker.py` → `DualcamRedisWorker`（`visual-dps-event-worker` + `-b`）；`event_worker_legacy.py` → `EventRedisWorker`（`visual-dps-event-worker-legacy`）
- 标注页：`/aisle`（`web/src/pages/AisleAnnotatePage.jsx`）

## 本地测试

```bash
cd /home/hqit/workspace/visual-dps-dualcam-exp
python3 -m pytest tests/test_aisle_group.py tests/test_aisle_shard.py tests/test_dualcam_contact.py tests/test_dualcam_slots.py tests/test_lift_point.py tests/test_dualcam_scale.py -q
```
