# 交接：当前进度与下一步

更新时间：2026-08-26　分支：`exp/dualcam-1-3`（基线成果在 `exp/tagged-aug85-v1`；门控在 `exp/action-gate-perf`）

## 一句话

门控与 v4 评估已落地；轨迹诊断表明骨架+框再堆形态门控希望有限。生产仍建议 **先开 A**。  
**算法/模型已并入产品仓** `/home/hqit/workspace/visual-dps-xugang-dev`（`app/pick_state` + `event-worker-2`，远程 HQIT `exp/event-worker-2`）；本仓继续作试验场，运行时勿再被产品依赖。

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

## 事件轨迹诊断（`exp/event-confirm`，已做）

脚本：`scripts/analyze_event_trajectories.py`、`scripts/analyze_rich_trajectories.py`  
报告：`output/sweep/event_confirm/trajectory_*.md`（gitignore，需本地重跑）

| 发现 | 含义 |
|------|------|
| FP 里 ~42% 为单帧命中事件；TP 几乎都是多帧 | 时序有油，但「连续帧」太粗 |
| score/depth/center 平均曲线有结构 | 肉眼像拣货过程 |
| 纯形态 AUC 多 <0.65；水平量（peak/span/action 均值）才强 | 形态未独立于分高段长 |
| 富轨迹（腕速/臂角/动作分相位） | 动作包络、臂角水平强，与 A 门控同源；斜率增量有限 |

结论：不宜再在骨架+框上堆连续帧/形态门控；换信息源。

## aisle3d 3D 查看器（验证工具，未进推理链路）

把 2D 骨架抬成 3D，用来看清人到底在做什么动作。详见 `docs/daily/DAILY-2026-08-13.md`、`DAILY-2026-08-14.md` 第七节。

```bash
.venv/bin/python scripts/serve_aisle3d.py   # 默认 8765，浏览器打开根路径
```

| 项 | 内容 |
|----|------|
| 查看器 | `scripts/aisle3d_viewer.html`（Three.js + 求解器全在这一个文件里） |
| 场景/标定服务 | `scripts/serve_aisle3d.py` |
| 四角反解 | `scripts/solve_scene.py`；批量重跑 `scripts/resolve_calib.py` |
| 标定结果 | `output/calib/<摄像头>.json`（**手工标注，已入库**，勿删） |

**卷尺基准（2026-08-14 实测）**：相机高 2.84 m、到近端 1.56 m、货架 2.2×2.0 m、巷道净宽 2.0 m、底沿贴地 `base=0`。
五路已按此重跑；「恢复反解」回到该机位 `solved`，不是卷尺初值。

**坑**：拣货面底沿贴地时 `base` 必须填 **0**。填 0.3 会让整个场景连同相机竖直平移 30cm，
而反投影残差一点都不变，画面上看不出来。旧默认面高 1.7 m 同样会系统性压矮相机（解出 ~2.45 m），
残差几乎看不出来——1-1-1 换成 2.0 m 后面高后残差 20.42 → 4.38 px。

**解剖先验**：反投影对若干自由度完全不敏感（翻过去残差一样），全靠 `poseEnergy` 里的软先验钉住——
`armBack` 肩后伸、`headFwd` 眼在耳前、`noseFwd` 鼻在肩线前、`elbowFwd` 肘不反折。
动权重前先量化再验残差，别凭手感调。

## 双路三角化（2026-08-26，验证，未进推理）

1-3 组巷道两端对打拼接片，两路标同一拣货面，各自反解后 Umeyama 对到 A 的巷道系，再三角化 17 点。  
详见 `docs/daily/DAILY-2026-08-26.md`。

```bash
.venv/bin/python scripts/serve_dualcam.py   # 8767  / 标注  /play 3D回放
# 姿态须 visual-dps conda + LD_LIBRARY_PATH（本仓 .venv 无 rtmlib）
# /home/hqit/miniconda3/envs/visual-dps/bin/python scripts/dualcam_lift.py 10
.venv/bin/python scripts/dump_skel3d.py
```

| 项 | 内容 |
|----|------|
| 标定 | `output/calib/dual_1-3.json`（**手工标注，已入库**，勿删） |
| 残差 | L 7.66 px / R 6.01 px；相机高 ≈2.95 m，基线 ≈5 m |
| 全片 2.5fps | 2558 帧、1565 配对；三角缝中位 **4 cm** |
| 腕到面 | p50 0.56 m；15.5% 贴面/伸进（d≤0.10）；71% 停在通道（d>0.40） |
| 产物 | `output/dualcam/`（gitignore，视频/npz/skel3d 本地重跑） |

结论：双路能把「伸进筐」和「停在框前」在深度上分开；单路射线∩面做不到。还没到改线上的时候。

**坑**：`align_rms=0` 只说明墙矩形尺寸一致，不证明点的是同一物理角。左右路必须按 16:9 显示半幅，格子拉满会裁歪。

## 下一步（已入 backlog）

1. **双路验收** — `/play` 上对照伸进筐 vs 路过的 `d_stereo`；残差大再微调四角重解（不必重跑姿态）
2. **框 ROI 时序变化**（光流/帧差/纹理）— 不依赖腕点是否配对
3. **专用手/前臂检测** — 缓解全身 RTMPose 的 NO_PAIR / 抖腕
4. **事件级 Temporal Action Localization** — 片段输入，非逐帧人-框配对

残留（骨架路径收尾，可选）：定生产只 A vs A+B；`v5_gated` 端到端导出核对。
