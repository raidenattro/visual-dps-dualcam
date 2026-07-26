# 交接：当前进度与下一步

更新时间：2026-07-26

## 一句话

链路已跑通并对齐基线，硬规则已被 logistic 取代（同召回、更低误报）；
瓶颈不在模型容量，在**判定单元**和**手-货框几何特征**的缺失。

## 结论数据（28-clip，标真 156 段）

| 包 | TP | FP | FN | 召回 |
|----|----|----|----|------|
| collector `rule-baseline-local-prod-test`（团队基线，仅供参考） | 147 | 430 | 9 | 94.23% |
| `pickstate-nogate`（**本仓对照基准**，无门控） | 147 | 442 | 9 | 94.23% |
| `pickstate-rule-v0`（硬规则门控） | 146 | 355 | 10 | 93.59% |
| **`pickstate-logistic-v1-t025`（当前最好）** | **147** | **306** | **9** | **94.23%** |
| `pickstate-logistic-v1-t030` | 143 | 283 | 13 | 91.67% |
| `pickstate-logistic-v1-t035` | 141 | 242 | 15 | 90.38% |
| `pickstate-logistic-v1-t040` | 135 | 222 | 21 | 86.54% |
| `pickstate-logistic-v1`（thr=0.5，过狠） | 117 | 163 | 39 | 75.00% |

logistic thr=0.25 相对无门控：召回不变，FP 减 31%。相对硬规则：召回更高且 FP 更低，硬规则无保留价值。

注意：以上为**同一 28-clip 上训练 + 评估**，数字偏乐观。

## 特征权重（`output/train/logistic_v1/feature_importance.md`）

帧级 GroupKFold OOF AUC = 0.8147，样本 4980（正 1650 / 负 3330），28 组。

| 特征 | coef（标准化） |
|------|---------------|
| `arm_torso_angle_max` | **+1.34** |
| `elbow_angle_mean` | +0.30 |
| `ankle_max_speed_norm` | −0.28 |
| `wrist_elevation_angle_max` | +0.02 |
| `shoulder_hip_knee_angle_min` | +0.02 |

后两维系数接近 0，基本是噪声。现有 5 维已接近榨干。

## 关键诊断：误报到底来自哪

对 `logistic-v1-t025` 的 306 个 FP 按原因分类（时间窗 ±45 帧 ≈ ±3s）：

| 类型 | 数量 | 占比 | 更强的拣货态模型能修吗 |
|------|------|------|----------------------|
| 同框、贴近标真段（判定边界） | 141 | 46% | 不能 |
| **邻框归属错**（人在拣，算到隔壁格子） | 120 | 39% | **不能** |
| 时间外（真·拣货态误判） | 45 | 15% | 能 |

**85% 的误报与「这个人是否在拣货」无关。**

根因是结构性的：当前判定单元是**人**，一旦判为拣货态，他手腕碰到的**所有**货框全部告警。
特征里没有任何描述「手与某个具体货框关系」的维度，邻框误报在这个架构下无解。

另外：306 个 FP 实际只对应约 **150 次独立接触**，其余是同一次接触在 `cooldown_frames=0` 下
逐帧重复计数。该口径与团队 baseline 一致（比较是公平的），但绝对值被放大。

## 下一步方案（已与用户讨论，待执行）

优先级从高到低：

1. **判定单元从 `人` 改为 `(人, 货框)` 对** — 直接攻 39% 的邻框误报
2. **补 G 组手-货框几何特征** — 进框深度（`BoxTrigger` 已输出 `depth_ratio`）、停留帧数、
   趋近速度、与次优框的 `margin_gap`、手腕相对框的方位
3. **模型仍先用 logistic** — 28 组数据下可解释、不易过拟合；GBDT 只作对照跑一次看差距
4. **补事件级去重口径**作辅助观察（主口径沿用团队现有的，保证可比）

预计涉及：`pipeline/box_trigger.py`（输出每框特征）、新增 `features/box_geometry.py`、
`pipeline/runner.py`（按对打分）、`train/build_dataset.py`（样本变成对）、
新配置 `configs/pipeline.pairwise_v1.json`。

**未定**：训练集只有 28 条 record 是硬瓶颈，扩标注的收益可能大于任何模型改动。

## 已验证的正确性

`BoxTrigger` 与 record 自带 `timeline.parquet` 逐帧比对：19369 帧中 19367 帧碰撞 token 完全一致。
2 帧差异是本仓去掉了原实现「命中首框即 `break`」的限制，手腕落在重叠区时多记一个框——这是刻意的，
邻框消歧需要这个信息。

## 产物位置

| 内容 | 路径 |
|------|------|
| 导出包 + 评估报告 | `output/export/<包名>/`（含 `accuracy_report.md/.json`） |
| 包间对比 | `output/compare/` |
| 训练数据与模型 | `output/train/dataset_v1.npz`、`output/train/logistic_v1/` |
| FN/FP 截图 | `output/viz/rule-v0-fn-fp/`（10 张 FN + 355 张 FP，另有 `.zip`） |

## 坑（踩过的）

- 导出必须按 baseline 帧号**全量**遍历，缺检测的帧也要产出空行。第一次漏了这点，
  clip_0013 少了 1356 帧，告警连续帧计数与对照包错位。
- 本仓货框取 record 自带 `manifest.annotation`，与 collector baseline 用的 reflection 临时标注
  文件不是同一版本，所以 nogate 比团队 baseline 多 12 个 FP。**对照一律用本仓 nogate。**
- collector 的评估脚本要用**系统 python3**（需 fastapi），本仓脚本要用 `.venv/bin/python`（需 pyarrow）。
