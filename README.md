# visual-dps-pick-state

独立实验目录：**先判拣货态，再判货格触发**。  
本阶段 **只读** `visual-dps-data-collector/localdata`，**不修改** `visual-dps`。

## 原则

1. FeatureBank：速度 / 加速度 / 关节角等多维特征（可扩展）
2. PickStateScorer：拟合权重得到 `pick_score`（可短窗平滑）
3. 仅当拣货态成立时，才跑 BoxTrigger（手腕 ∩ 货框）
4. Pipeline stages / experts **可拆装、可配置**（`configs/*.json`）
5. 现有速度门控 / triple90 等保留为 **expert**，不是唯一硬逻辑

## 目录

| 路径 | 说明 |
|------|------|
| `pipeline/` | 装配与运行：FeatureBank → Scorer → BoxTrigger → Alarm |
| `features/` | 特征提取器注册表 |
| `experts/` | 可插拔专家（规则门控、线性、以后 HGB） |
| `adapters/` | 只读 collector 路径、manifest、review |
| `train/` | 用标真拟合权重（后续） |
| `eval/` | 导出 upload JSON，调用 collector 评估脚本 |
| `configs/` | pipeline 配置 |
| `scripts/` | CLI 入口 |

## 默认数据路径（153）

```text
COLLECTOR_ROOT=/home/hqit/workspace/visual-dps-data-collector
LOCALDATA=$COLLECTOR_ROOT/localdata
MANIFEST=$LOCALDATA/export/rule-baseline-local-prod-test/_manifest.json
```

可用环境变量覆盖：`PICK_STATE_COLLECTOR_ROOT`。

## 环境

系统 python 缺 pandas/pyarrow/sklearn，必须用本仓 venv：

```bash
cd /home/hqit/workspace/visual-dps-pick-state
python3 -m venv .venv
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/check_data_access.py
```

## 回归流程

```bash
# 1) 导出（写本仓 output/，不碰 collector）
.venv/bin/python scripts/export_manifest28.py \
  --config configs/pipeline.nogate.json --out output/export/pickstate-nogate-prod-test
.venv/bin/python scripts/export_manifest28.py \
  --config configs/pipeline.baseline_rule_expert.json --out output/export/pickstate-rule-v0-prod-test

# 2) 评估 / 对比（用 collector 脚本，系统 python 已装 fastapi）
cd /home/hqit/workspace/visual-dps-data-collector
python3 scripts/data/evaluate_inference_upload.py --in-place --dirs \
  /home/hqit/workspace/visual-dps-pick-state/output/export/pickstate-nogate-prod-test \
  /home/hqit/workspace/visual-dps-pick-state/output/export/pickstate-rule-v0-prod-test
python3 scripts/data/compare_export_false_alarms.py \
  --baseline .../pickstate-nogate-prod-test --experiment .../pickstate-rule-v0-prod-test \
  --out-dir /home/hqit/workspace/visual-dps-pick-state/output/compare
```

导出帧号直接复用 baseline 导出包的 `frame_idx`，无检测帧也产出空行，保证告警连续帧计数可比。

## 当前结果（28-clip，标真 156 段）

| 包 | TP | FP | FN | 召回 |
|----|----|----|----|------|
| collector `rule-baseline-local-prod-test` | 147 | 430 | 9 | 94.23% |
| 本仓 `pickstate-nogate`（无门控自基线） | 147 | 442 | 9 | 94.23% |
| 本仓 `pickstate-rule-v0`（硬规则门控） | 146 | 355 | 10 | 93.59% |
| 本仓 `pickstate-logistic-v1-t025`（**当前最好**） | 147 | 306 | 9 | 94.23% |

完整阈值扫描、特征权重、误报归因见 **[docs/HANDOVER.md](docs/HANDOVER.md)**。

nogate 与 collector baseline 召回一致，FP 差 12：本仓货框取 record 自带 `manifest.annotation`（与 baseline 用的 reflection 临时标注文件不同版本），且 BoxTrigger 去掉了原实现命中首框即 `break` 的限制。以 `timeline.parquet` 做逐帧几何校验，19369 帧中 19367 帧完全一致，2 帧差异均为重叠框多记，符合预期。

## 训练与可视化

```bash
.venv/bin/python train/build_dataset.py     # → output/train/dataset_v1.npz
.venv/bin/python train/fit_logistic.py      # → output/train/logistic_v1/
.venv/bin/python scripts/render_fn_fp_frames.py \
  --report output/export/<包名>/accuracy_report.json --out output/viz/<包名>
```

## 与两仓关系

- **collector**：只读 pose / review / export；评估继续用其 `evaluate_inference_upload.py`
- **visual-dps**：本仓验证稳定前 **零改动**；以后单独 PR 迁入 `event_engine`

## 状态

阶段 1、2 完成：只读适配、5 维特征、BoxTrigger + Alarm、28-clip 回归、logistic 拟合权重。  
下一步：判定单元改为 `(人, 货框)` 对 + 补手-货框几何特征。理由与数据见 [docs/HANDOVER.md](docs/HANDOVER.md)。
