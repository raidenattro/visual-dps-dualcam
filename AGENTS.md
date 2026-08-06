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

## 标准评估流程

数据集为 `output/manifests/tagged_aug85_v1.json`（`data.db` 中「8.3/8.4/8.5 新标注」三标签，
26 条 record、210 段，段级 5:5）。**只看段（事件）级漏报与误报，不看帧级。**

调参不要走完整导出——判定策略只影响「打完分之后怎么用分数」，落一次分数后秒级重放即可：

```bash
# 1) 落分数（线上口径 15fps），跑一次
.venv/bin/python scripts/dump_pair_scores.py --config <config> \
  --split-role val --sample-fps 15 --out output/scores/<run>

# 2) 扫阈值 / 连续帧（64 组合约 13 秒）
.venv/bin/python scripts/sweep_policy.py --scores output/scores/<run> \
  --thresholds 0.10,0.20,0.30 --min-frames 1,2,4,6 --out-dir output/sweep/<run>

# 3) 定下工作点后再完整导出 + 段级评估
.venv/bin/python scripts/export_manifest28.py --config <config> \
  --manifest output/manifests/tagged_aug85_v1.json --split-role val --out output/export/<包名>
.venv/bin/python scripts/eval_tagged_val.py --pkg output/export/<包名>
```

28-clip 时代那套（collector `evaluate_inference_upload.py` + `pickstate-nogate-prod-test` 基准）
已作废，标注质量不可靠，别再拿来对照。

## 不变量（改代码时别破坏）

- 导出帧号**必须**复用 baseline 导出包的 `frame_idx`；无检测的帧也要产出空行，否则告警连续帧计数与对照包错位
- `is_picking` 等价于 `rule_alarm_collisions` 非空
- 货框 token 格式 `Box_{box_id}`
- **手腕门槛是两套，别混**：`BoxTrigger` 用自己的 `wrist_score_min`（当前 0.15）决定能否触发，
  角度特征仍走 `features/geometry.KPT_SCORE_MIN`（0.3）。改任一边都要重建训练集
- **抽帧必须在喂进 pipeline 之前做**（`adapters/frame_sampling.py`）。时序特征、分数平滑、
  连续帧计数都逐帧推进，在全帧结果上后处理得到的参数搬不到线上
- 段级评估算误报覆盖时要用 **train+val 全部真值段**，否则 train 段上的正确告警会被算成误报

## 训练

```bash
.venv/bin/python train/build_pair_dataset.py \
  --manifest output/manifests/tagged_aug85_v1.json \
  --wrist-score-min 0.15 --out output/train/pairs_v4_wscore.npz
.venv/bin/python train/fit_logistic.py --dataset output/train/pairs_v4_wscore.npz \
  --out-dir output/train/v4_wscore --name v4_wscore --features <逗号分隔的特征子集>
```

**训练用 25fps 逐帧，推理用 15fps**：训练集按 15fps 重建过，样本少 30%，损失盖过口径一致的收益。
所以 `build_pair_dataset.py` 不要传 `--sample-fps`，而 `dump_pair_scores.py` 必须传 `15`。

时序特征（`features/pair_temporal.py`）有状态，**必须逐帧推进，包括没有骨架的空帧**，
否则停留计数不清零。`build_pair_dataset.py` 与 `pipeline/runner.py` 两边口径必须一致。

标签复刻 collector 的**区间**口径（连续同 token 的条目合并成段，配对落在段内且 token 匹配为正），
与段级评估指标同口径。

数据只有 **26 条 record**，分组泛化样本量就是 26，用 `GroupKFold(groups=record_id)`，慎用高容量模型。
加特征的边际收益已经很小，别指望靠堆维度翻盘。

## 可视化

```bash
# 漏报段逐段出图（红框=标注框，黄框=手实际进的框，手腕标置信度，低于门槛画红圈）
.venv/bin/python scripts/render_missed_segments.py --wrist-min 0.15

# 网页逐事件回放（正确/误报/漏报都停下等继续）
.venv/bin/python scripts/event_player.py --pkg output/export/<包名>
```
