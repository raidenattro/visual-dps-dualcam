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

## 快速检查

```bash
cd /home/hqit/workspace/visual-dps-pick-state
python scripts/check_data_access.py
```

## 与两仓关系

- **collector**：只读 pose / review / export；评估继续用其 `evaluate_inference_upload.py`
- **visual-dps**：本仓验证稳定前 **零改动**；以后单独 PR 迁入 `event_engine`

## 状态

骨架阶段：接口与配置已立，拟合与全量回归待下一步。
