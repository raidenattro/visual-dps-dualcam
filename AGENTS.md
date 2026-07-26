# visual-dps-pick-state — Agent 说明

拣货态识别的**独立实验仓**。目标：用可学习权重替代硬规则，降低误报同时保住召回。

先读 [docs/HANDOVER.md](docs/HANDOVER.md) 了解当前进度与下一步。

## 硬性约束

1. **只读 collector**：`/home/hqit/workspace/visual-dps-data-collector` 只准读，禁止写入其 `localdata`
2. **不碰 visual-dps**：`/home/hqit/workspace/visual-dps` 本仓验证稳定前零改动；本仓与它无 import 依赖
3. **产物只写本仓 `output/`**
4. 本仓**不需要** docker 构建部署（`visual-dps` 那套 build/deploy 规则不适用）

## Python 环境（两套，别混）

| 用途 | 解释器 | 原因 |
|------|--------|------|
| 本仓所有脚本 | `.venv/bin/python` | 系统 python 缺 pandas/pyarrow/sklearn |
| collector 评估脚本 | 系统 `python3` | collector 依赖 fastapi 等，装在系统环境 |

```bash
cd /home/hqit/workspace/visual-dps-pick-state
python3 -m venv .venv   # 首次
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
.venv/bin/python -m pytest tests/ -q
```

## 标准回归流程

```bash
# 1) 导出（本仓 venv）
.venv/bin/python scripts/export_manifest28.py \
  --config configs/pipeline.logistic_v1.json \
  --out output/export/<包名>

# 2) 评估（collector 目录 + 系统 python）
cd /home/hqit/workspace/visual-dps-data-collector
python3 scripts/data/evaluate_inference_upload.py --in-place \
  --dirs /home/hqit/workspace/visual-dps-pick-state/output/export/<包名>

# 3) 对比（基准固定用 pickstate-nogate-prod-test）
python3 scripts/data/compare_export_false_alarms.py \
  --baseline /home/hqit/workspace/visual-dps-pick-state/output/export/pickstate-nogate-prod-test \
  --experiment /home/hqit/workspace/visual-dps-pick-state/output/export/<包名> \
  --out-dir /home/hqit/workspace/visual-dps-pick-state/output/compare
```

## 不变量（改代码时别破坏）

- 导出帧号**必须**复用 baseline 导出包的 `frame_idx`；无检测的帧也要产出空行，否则告警连续帧计数与对照包错位
- `is_picking` 等价于 `rule_alarm_collisions` 非空
- 货框 token 格式 `Box_{box_id}`
- 对照基准用本仓 `pickstate-nogate-prod-test`，**不要**直接对 collector 的 `rule-baseline-local-prod-test`（两者货框标注来源不同，会把标注差异算进算法效果）

## 训练

```bash
.venv/bin/python train/build_dataset.py          # → output/train/dataset_v1.npz
.venv/bin/python train/fit_logistic.py           # → output/train/logistic_v1/
```

标签：帧号落在 `verified_true` 条目上为正。数据只有 **28 条 record**，分组泛化样本量就是 28，
用 `GroupKFold(groups=record_id)`，慎用高容量模型。

## 可视化

```bash
.venv/bin/python scripts/render_fn_fp_frames.py \
  --report output/export/<包名>/accuracy_report.json \
  --out output/viz/<包名>
```

输出「监控画面 + 骨架 + 货框」JPG，分 `fn/` 与 `fp/`。
