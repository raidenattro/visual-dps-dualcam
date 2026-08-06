# 交接：当前进度与下一步

更新时间：2026-08-06　分支：`exp/tagged-aug85-v1`

## 一句话

换到只含「8.3/8.4/8.5 新标注」的高质量数据集后重做了一遍，**漏报的瓶颈已确认在上游姿态估计
而非拣货判定**（18 段漏报无一是判定算错）；模型侧还能压误报，但工作点未定。

28-clip 时代的结论已作废（标注质量问题），归档在
[`archive/HANDOVER-28clip-2026-07-27.md`](archive/HANDOVER-28clip-2026-07-27.md)，
其中的否定结论（站位代理、单目深度）仍有参考价值。

## 数据集与口径

| 项 | 内容 |
|----|------|
| manifest | `output/manifests/tagged_aug85_v1.json`（`scripts/build_tagged_manifest.py` 生成） |
| 规模 | 26 条 record、210 段，段级 5:5 划分（train 105 / val 105） |
| 来源 | `data.db` 中带「8.3新标注 / 8.4新标注 / 8.5新标注」标签的 record，其余全部排除 |
| 评估 | **只看段（事件）级漏报与误报，不看帧级**；`scripts/eval_tagged_val.py` |
| 优先级 | **漏报最少优先，在此基础上误报最少** |
| 线上口径 | 工作系统按 **15fps 抽帧**处理；离线数据是逐帧 25fps，须用 `--sample-fps 15` 模拟 |

## 当前配置与模型

| 项 | 值 |
|----|-----|
| 配置 | `configs/pipeline.pairwise_v4_wscore.json` |
| 模型 | `output/train/v4_wscore/model.json`（22 维，OOF AUC 0.8803） |
| 训练集 | `output/train/pairs_v4_wscore.npz`（25fps 逐帧 + 手腕门槛 0.15，23572 样本） |
| 手腕触发门槛 | **0.15**（角度特征仍走 0.3，不可混） |
| 阈值 / 连续帧 | **未定**，见下方取舍前沿 |

**训练用 25fps、推理用 15fps** 是刻意的：训练集按 15fps 重建过，样本少 30% 反而变差。

## 取舍前沿（val 105 段，41.5 分钟）

| 漏段 | 召回 | 阈值 / 连续帧 | 误报事件 | 每小时误报 |
|------|------|--------------|---------|-----------|
| 18 | 82.86% | 0.10 / 1帧 | 1528 | 2212 |
| 24 | 77.14% | 0.15 / 2帧 | 745 | 1078 |
| 26 | 75.24% | 0.20 / 2帧 | 608 | 880 |
| 28 | 73.33% | 0.24 / 2帧 | 526 | 761 |
| 33 | 68.57% | 0.24 / 5帧 | 157 | 227 |
| 36 | 65.71% | 0.35 / 6帧 | 75 | 109 |

严格按漏报优先取 0.10 / 1帧，但每小时 2212 次误报实际不可用；最严档位仍有每小时 109 次。
误报口径为「未被任何标注段覆盖的告警事件」，偏严（详见日报第六节）。

## 结论汇总

已采纳：

| 改动 | 效果 |
|------|------|
| 触发手角度特征（替代左右聚合） | OOF AUC 0.8915 → 0.8989 |
| 手腕触发门槛 0.30 → 0.15 | 召回上限 74.29% → 82.86%（15fps），同等漏报下误报也更低 |
| 手腕置信度进特征 | 段级误报降 2%~19% |
| 缺失值改均值填充（原 fail-open 0.55 = 必报） | 修掉训练/推理口径不一致 |
| 提高连续帧要求 | 压误报效率最高的杠杆，优于提高分数阈值 |
| 离线策略扫描工具 | 64 组合 13 秒，此前 3 组合 3 分钟 |

已否掉（不要再试）：

| 尝试 | 为什么不行 |
|------|-----------|
| 头部朝向特征（鼻/眼/耳 5 点） | 全线零贡献，头部方位两维有害（0.8989 → 0.8922）。头部测的是「人站在哪、朝哪看」，不是「手有没有伸进去」 |
| 左右镜像增强 | 特征本身左右对称，镜像后 21 维里只有 `wrist_bearing_x` 变号；左右手触发本就 7277:7270 均衡 |
| 一人一帧只报一个框（互斥） | 只减 1~9 个误报。62% 的「邻框错」是不同人或不同时刻，互斥覆盖不到 |
| 训练集按 15fps 重建 | 样本少 30%，损失盖过帧率口径一致的收益 |

## 漏报根因（26 段全部拆开，已出图）

| 根因 | 段数 | 放宽门槛到 0.15 |
|------|------|----------------|
| 人在，手腕置信度不足 0.3（最低 0.19，因头/身体遮挡） | 12 | 部分救回 |
| 能用的手不在框内，在框内的手被丢弃 | 7 | 部分救回 |
| 手确实进框了，但进的是隔壁那格 | 7 | 救不回 |

**没有一段是判定算法的问题。** 业务侧已确认「进隔壁格」那 7 段标注是对的，
即姿态估计把手腕估偏了一格 —— 推理分辨率 852×480，货框内切半径仅 18~21 像素。

图在 `output/viz/missed_segments/`，文件名前缀即根因分类，
用 `scripts/render_missed_segments.py` 重出。

## 下一步

1. **定工作点** —— 需业务侧给出漏报/误报的可接受区间。
2. **上游姿态精度是硬瓶颈** —— 更准的姿态模型或更高输入分辨率，是继续降漏报的唯一途径。
3. **孤立误报（占 22.5%）尚未逐个定性** —— 用 `scripts/event_player.py` 在选定工作点下逐事件回放。
4. **样本量仍是瓶颈** —— 26 条 record，加特征边际收益已很小（手腕置信度只换 +0.0006 AUC）。

## 常用命令

```bash
# 1) 生成 manifest（data.db 三标签 + 段级 5:5）
.venv/bin/python scripts/build_tagged_manifest.py

# 2) 构建训练集（25fps 逐帧 + 手腕门槛 0.15）
.venv/bin/python train/build_pair_dataset.py \
  --manifest output/manifests/tagged_aug85_v1.json \
  --wrist-score-min 0.15 --out output/train/pairs_v4_wscore.npz

# 3) 训练
.venv/bin/python train/fit_logistic.py \
  --dataset output/train/pairs_v4_wscore.npz \
  --out-dir output/train/v4_wscore --name v4_wscore --features <逗号分隔特征>

# 4) 落分数（线上口径 15fps），跑一次即可
.venv/bin/python scripts/dump_pair_scores.py \
  --config configs/pipeline.pairwise_v4_wscore.json \
  --split-role val --sample-fps 15 --out output/scores/v4_wscore_15fps

# 5) 扫判定策略（秒级，任意组合）
.venv/bin/python scripts/sweep_policy.py --scores output/scores/v4_wscore_15fps \
  --thresholds 0.10,0.20,0.24,0.30,0.35 --min-frames 1,2,4,5,6 \
  --out-dir output/sweep/v4_wscore_15fps

# 6) 完整导出 + 段级评估（定下工作点后做）
.venv/bin/python scripts/export_manifest28.py --config <config> \
  --manifest output/manifests/tagged_aug85_v1.json --split-role val --out output/export/<包名>
.venv/bin/python scripts/eval_tagged_val.py --pkg output/export/<包名>

# 7) 漏段出图 / 事件回放
.venv/bin/python scripts/render_missed_segments.py --wrist-min 0.15
.venv/bin/python scripts/event_player.py --pkg output/export/<包名>   # 网页播放器
```

## 产物位置

| 内容 | 路径 |
|------|------|
| manifest | `output/manifests/tagged_aug85_v1.json` |
| 训练集与模型 | `output/train/pairs_v4_wscore.npz`、`output/train/v4_wscore/` |
| 配对分数（扫描输入） | `output/scores/<run>/` |
| 策略扫描表 | `output/sweep/<run>/sweep.md` |
| 导出包 + 段级评估 | `output/export/<包名>/eval_tagged_val.json` |
| 漏段图 | `output/viz/missed_segments/` |

## 坑

- **手腕门槛与角度特征门槛不是一回事**：`BoxTrigger` 用自己的 `wrist_score_min`（0.15），
  角度特征仍走 `features/geometry.KPT_SCORE_MIN`（0.3）。混了会与训练口径不一致。
- **抽帧必须在喂进 pipeline 之前做**：时序特征、平滑、连续帧计数都逐帧推进，
  在全帧结果上后处理得到的参数搬不到线上。
- 导出必须按 baseline 帧号全量遍历，无检测的帧也要产出空行，否则连续帧计数错位。
- `build_pair_dataset.py` 与 `pipeline/runner.py` 两边的时序特征推进口径必须一致。
- collector 评估脚本用**系统 python3**，本仓脚本用 `.venv/bin/python`。
- 本仓段级评估（`eval_tagged_val.py`）中，误报判定用 **train+val 全部真值段**做覆盖检查，
  否则会把 train 段上的正确告警算成误报。
