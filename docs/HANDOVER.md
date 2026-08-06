# 交接：当前进度与下一步

更新时间：2026-08-06 晚　分支：`exp/tagged-aug85-v1`

## 一句话

段定义已修正为「一次拣货」；**漏报基本触底（瓶颈在上游姿态）**，当前卡点是
**标注漏标导致误报数字虚高**。误报审核工具已搭好，明天继续用**新四档口径**重审。

## 数据集与口径（以 v2 为准）

| 项 | 内容 |
|----|------|
| manifest | `output/manifests/tagged_aug85_v2.json` |
| 规模 | 28 条 record、**477** 次拣货（train 239 / val 238） |
| 切段规则 | 同一货框、相邻标注帧间隔 ≤15 帧（≈1s@15fps）算一次；**跟踪 ID 不参与切段** |
| 来源 | `data.db`「8.3/8.4/8.5 新标注」 |
| 评估 | **只看事件级**漏报/误报；覆盖判定用 train+val 全部真值段 |
| 线上 | `--sample-fps 15`；训练构建也用 15fps（`pairs_v5.npz`） |

v1（210 段）已作废：旧逻辑把同框同人整段连在一起，空档被当成正样本，指标虚高/失真。

`tagged_aug85_v3.json` = v2 + 审核补的 87 段（仅进 train）。**补标重训（v6）端到端无收益，勿再推**；
v3 只用于「评估时把已确认漏标从误报里扣掉」。

## 当前模型与基线（v5）

| 项 | 值 |
|----|-----|
| 配置 | `configs/pipeline.v5.json` |
| 模型 | `output/train/v5_base/model.json`（OOF AUC **0.8759**） |
| 训练集 | `output/train/pairs_v5.npz`（15fps + wrist 0.15，17523 样本，正 32.4%） |
| 分数 | `output/scores/v5/` |
| 手腕触发 | 0.15（角度特征仍 0.3，不可混） |

val 238 段、阈值 0.20 / 连续 1 帧（漏报优先）：

| | 召回 | 漏段 | 误报 | 算法误报 |
|--|------|------|------|----------|
| v5 + v2 标注 | **95.80%** | 10 | 1588 | 1156 |
| v5 + v3（扣掉已审漏标） | 95.80% | 10 | 1493 | 1061 |

漏报 10 段归因（`scripts/analyze_v5_errors.py`）：**8 段 NO_PAIR**（上游没给出手进该框的配对），
2 段 LOW_SCORE。策略层（连续帧 / 窗口投票 / 迟滞双阈值 / 互斥）**已扫过，到顶**。

## 标注审核（未完成，明天优先）

高分「误报」人工抽检：**约九成是漏标**，不是模型错。但旧审核只有三档，**没区分
「在拣这个红框」vs「在拣货但红框不对」**——后者对红框是真误报。旧判定标了 `verdict_v=1`，需重判。

| 工具 | 命令 |
|------|------|
| 分层抽样清单 | `scripts/build_fp_audit_set.py` → `output/audit/fp_audit_set.json` |
| 审核网页 | `.venv/bin/python scripts/fp_audit.py --events output/audit/fp_audit_set.json --min-peak 0 --out output/audit/fp_audit.json` |
| 备份 | `output/audit/fp_audit.v1backup.json`（旧三档结果） |

**四档口径（verdict_v=2）：**

1. 在拣这个红框（标注漏了）→ `missing`
2. 在拣货，但不是这个框 → `wrong_box`（对红框算真误报）
3. 没在拣货 → `not_pick`
4. 不确定 → `unsure`

看右侧峰值帧时：黄框=触发误报的人，红框=被报的货框，黄箭头=手腕→货框；左侧视频看动作过程。

## 已否掉（今晚验证）

| 尝试 | 结论 |
|------|------|
| 窗口内累计投票 | 与连续帧持平，无额外收益 |
| 迟滞双阈值 | 中等召回区能压一点算法误报，换不来更高召回 |
| 一人一框互斥 | 误报几乎不动 |
| 用审核漏标补标重训（v6） | OOF AUC↑（0.89）但端到端变差——自我确认，别再做 |

## 明天继续

1. **重启审核工具**，用四档口径把 `fp_audit_set.json`（258 条，含分层抽样）审完；旧结果带「旧」标签建议重判。
2. 按层漏标率加权外推，算出**可信误报数 / 精确率**，再定工作点。
3. 漏报侧若还要砍：只能动上游姿态（那 8 段 NO_PAIR）。
4. 时序动作模型可做，但须建立在干净标签上；先审完再开。

## 常用命令

```bash
# manifest（一次拣货切段）
.venv/bin/python scripts/build_tagged_manifest.py   # → tagged_aug85_v2.json

# 训练 / 落分 / 扫策略（v5 基线）
.venv/bin/python train/build_pair_dataset.py \
  --manifest output/manifests/tagged_aug85_v2.json \
  --out output/train/pairs_v5.npz --sample-fps 15 --wrist-score-min 0.15
.venv/bin/python train/fit_logistic.py --dataset output/train/pairs_v5.npz \
  --out-dir output/train/v5_base --name pair_logistic_v5
.venv/bin/python scripts/dump_pair_scores.py --config configs/pipeline.v5.json \
  --manifest output/manifests/tagged_aug85_v2.json --split-role val \
  --sample-fps 15 --out output/scores/v5
.venv/bin/python scripts/sweep_policy.py --scores output/scores/v5 \
  --manifest output/manifests/tagged_aug85_v2.json --exclusive no \
  --out-dir output/sweep/v5

# 误报归因 + 分层审核清单
.venv/bin/python scripts/analyze_v5_errors.py --threshold 0.20 --min-frames 1
.venv/bin/python scripts/build_fp_audit_set.py
.venv/bin/python scripts/fp_audit.py --events output/audit/fp_audit_set.json \
  --min-peak 0 --out output/audit/fp_audit.json

# 拣货事件播放器（确认「一次拣货」定义）
.venv/bin/python scripts/event_player.py --manifest output/manifests/tagged_aug85_v2.json
```

## 坑

- **跟踪 ID 不能进切段判据**：同一次拣货 ID 会 #2/#4 抖动，会造大量假单帧段。
- **审核对象是「这个人 × 这个红框」**，不是「画面里有没有人在拣货」。
- 用模型高分片段补正样本再训练 = 自我确认，AUC 好看、线上没用。
- 不要写「每小时误报」：数据集是高密度剪辑段，不能线性外推。
- 手腕门槛两套、抽帧必须在 pipeline 前、误报覆盖要用全部真值段——见 `AGENTS.md`。
