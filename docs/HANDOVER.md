# 交接：当前进度与下一步

更新时间：2026-08-07　分支：`exp/tagged-aug85-v1`

## 一句话

误报四档审完（真误报≈95%）；**A 动作门控 + B 邻框深度**已写入 pipeline（可配开关/阈值）。
评估集扩到 8.6/8.7（`tagged_aug85_v4`）。工作点建议：**先开 A（稳）**，B 视召回预算。

## 数据集

| 项 | 内容 |
|----|------|
| 主评估 manifest | `output/manifests/tagged_aug85_v4.json` |
| 规模 | 36 record、**862** 段（train 431 / val 431） |
| 标签 | 8.3–8.7 新标注（相对 v2 多 8 条：6×8.6 + 2×8.7） |
| 对照基线 | `tagged_aug85_v2.json`（28 / 477，val 238） |
| 切段 | 同框、间隔 ≤15 帧；**跟踪 ID 不参与切段** |
| 评估 | **只看事件级**；FP 覆盖用 train+val 全部真值段 |
| 线上 | 推理 `--sample-fps 15`；训练构建不传 sample-fps（25fps 逐帧） |

v1 作废；v3 补标重训（v6）端到端无收益，勿再推。

## 模型与配置

| 项 | 路径 |
|----|------|
| 配对模型 | `output/train/v5_base/model.json` |
| 无门控配置 | `configs/pipeline.v5.json` |
| **门控配置** | `configs/pipeline.v5_gated.json` |
| 动作门控模型 | `output/train/action_gate_v1/model.joblib` |
| 无门控分数（v4） | `output/scores/v5_on_v4/` |
| 审核结果 | `output/audit/fp_audit.json` |

`pair_state.action_gate` / `pair_state.box_gate`：`enabled` + 阈值，**不要写死**。

默认门控工作点：动作 ≥0.30，depth_ratio ≥0.25，配对阈值 0.20，连续 1 帧。

## 关键数字（配对阈值 0.20）

**v2 val（离线 OOF，研发对照）：**

| | 召回 | 漏 | 误报 |
|--|------|-----|------|
| v5 基线 | 95.80% | 10 | 1588 |
| A+B | 92.02% | 19 | 681 |

**v4 val（已训 `action_gate_v1` 叠门控）：**

| 子集 | 方案 | 召回 | 漏 | 误报 |
|------|------|------|-----|------|
| 全量 431 | 基线 | 95.59% | 19 | 2668 |
| 全量 431 | A | 95.36% | 20 | 1319 |
| 全量 431 | A+B | 91.42% | 37 | 909 |
| 仅新 188 | 基线 | 94.68% | 10 | 1073 |
| 仅新 188 | A+B | 87.23% | 24 | 516 |

新摄像头上 B 伤召回更明显；若优先保召回，生产可先只开 A。

## 审核结论

四档（258 条）：wrong_box / not_pick 各 ≈46.5%，missing ≈7%。  
工具：`scripts/fp_audit.py`（端口曾用 8910）。

## 已否掉

窗口投票 / 迟滞 / 一人一框互斥 / 补标重训 v6 — 无收益或几乎无效。

## 标准命令

```bash
# 落分（无门控分数；门控在策略重放或完整导出时生效）
.venv/bin/python scripts/dump_pair_scores.py --config configs/pipeline.v5.json \
  --manifest output/manifests/tagged_aug85_v4.json --split-role val \
  --sample-fps 15 --out output/scores/<run>

# 门控完整导出评估
.venv/bin/python scripts/export_manifest28.py --config configs/pipeline.v5_gated.json \
  --manifest output/manifests/tagged_aug85_v4.json --split-role val \
  --out output/export/<包名>
.venv/bin/python scripts/eval_tagged_val.py --pkg output/export/<包名>

# 重训动作门控
.venv/bin/python train/fit_action_gate.py \
  --manifest output/manifests/tagged_aug85_v2.json \
  --cache output/train/action_sequences_v2.json \
  --out-dir output/train/action_gate_v1
```

离线扫门控：`scripts/eval_action_gate.py`、`scripts/eval_box_gate.py`。

## 下一步

1. 定生产默认：只 A vs A+B（看新摄像头召回预算）
2. `v5_gated` 端到端导出核对
3. 漏报若还要砍：上游姿态（NO_PAIR），不在本仓策略层
