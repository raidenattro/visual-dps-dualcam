# pick_state（自包含算法包，本仓运行时不用）

从试验场 `visual-dps-pick-state` 并入的拣货态打分与门控。**visual-dps-dualcam-exp 的 event-worker 已改为双路 3D `contact_slots`，不再加载 `pipeline.v5_gated`。** 本目录仅保留对照/离线脚本。

| 项 | 路径 |
|----|------|
| 历史生产配置 | `configs/pipeline.v5_gated.json`（本仓不挂到 worker） |
| 容器冒烟配置 | `configs/pipeline.v5_smoke.json` |
| 模型 | `models/v5_base/model.json`、`models/action_gate_v1/model.onnx` |

- 进程内 smoke：`python scripts/run_pick_state_local.py`
- 线上入口：**`event_worker_2.py` → `DualcamRedisWorker`**（不是 `PickStatePipeline`）
