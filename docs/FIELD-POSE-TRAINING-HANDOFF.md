# 现场姿态模型微调 — 交接说明

> 仓库：`visual-dps-dualcam-exp`（双路 3D 拣货实验仓）  
> 目的：供在 **其他仓库 / MMPose 训练环境** 接手域内微调时使用。  
> 更新：2026-03-26（与当前分支 `feature/unified-stack-legacy-dualcam` 行为对齐）

---

## 1. 业务与链路（为什么要关心 2D）

### 1.1 系统在做什么

- 巷道 **左/右两路相机** 成组标定，infer 容器输出 **COCO-17 2D 关键点** → Redis pose 流。
- **Dualcam event-worker** 对 L/R 对齐后 **逐关节三角化**（`dualcam/lift.py` 的 `lift_point`），在 3D 货格上做 **`contact_slots` 贴墙碰撞**；连续 N 帧命中才告警。
- 前端巷道直播：**2D 叠在画面上**，**3D 窗** 绘制 worker 下发的 `persons_3d`（与贴墙判定同源几何，见下文）。

### 1.2 2D 在链路中的位置

```
RTSP → infer（RTMDet 检人 + RTMPose 姿态）→ 2D keypoints
     → worker（lift_point + clamp）→ 3D xyz → contact_slots / SSE
```

**3D 与贴墙没有独立「真值传感器」**：腕是否进格，完全依赖 **2D 像素 + 双目标定 + 三角化**。  
因此：**2D 不准时，后端几何优化（IK、整体抬升、平滑）只能减轻离谱形变，无法提高关键点真值上限。**

---

## 2. 现场问题是什么（训练要对准什么）

### 2.1 主要现象（产品/现场反馈）

| 现象 | 与 2D/3D 的关系 |
|------|------------------|
| **3D 骨架扭曲、异常肘腕角** | 左右 2D 不一致（gap 大）、单路 fallback、错配对；根因常是 **2D 或配对/标定** |
| **腕部抖动** | 7.5Hz 有效 pose（`pose_frame_interval=2`）+ **无直播时序平滑**；2D 噪声在三角化中被放大 |
| **贴墙与 3D 视觉不一致**（历史） | 曾用 clamp 前 raw 腕算 token、clamp 后 xyz 画图；**已改为同一套 clamp 后坐标** |
| **头部关键点贴地/乱线** | 五官 3D 易失败；**已改为 worker 不抬 0–4，前端不画五官** |

### 2.2 已确认的非主因（避免在训练仓重复踩坑）

- **`app_config.json` 里 `rtmpose_onnx_det_size` / `pose_size`**：推理 **不读**，尺寸由 `services/inference_backends/model_registry.py` 与 `models.backend` / `models.det` 决定。
- **删掉 `skel3d_smooth.py`**：0902 起直播 **从未开启** `smooth_2d/3d`；删的是死代码，**不改变**线上抖动特性。
- **仅调 det 输入尺寸 JSON**：无效；换档用 `models.det`（nano/m）和 `models.backend`（rtmpose_t/s/m）。

### 2.3 训练「预期效果」应如何定义

微调 **不能替代**：

- 双路 **时间对齐**（`dualcam.pair_window_*`）、L/R **标定残差**、infer **吞吐** 导致的采样间隔。

微调 **应当改善**（在域内数据上可验收）：

- 腕/肘 **2D 位置与 score** 在遮挡、俯角、工装场景下更稳；
- 双路 **三角化 gap**（尤其腕）分布收窄；
- **贴墙误报/漏报**（需 clip 或事件级标注更佳）；
- 3D 窗 **离谱骨角** 比例下降。

---

## 3. 本仓已做过的后端尝试（为何转向训练）

| 方案 | 结果 |
|------|------|
| 直播 2D/3D 窗平滑（`skel3d_smooth`） | 0902 因 **~50ms/帧** 关闭；已 **移除代码**，保留 `dualcam/pose_timing.py`（hold 时间） |
| **方案 A：整体抬升 `lift_person17`**（躯干锚 + 四肢 depth_ref） | **已回滚**：扭曲略好但 **腕抖加重**（肩锚 depth 替代 prev） |
| 腕单独 `lift_point` | 试过后仍不满意，随整体抬升一并回滚 |
| **贴墙与 `persons_3d` 均用 clamp 后腕** | **保留** |
| **五官不抬、3D 不画 0–4** | **保留** |
| 当前 3D | **17 点各 `lift_point`** + `_clamp_flying_wrists`（肩肘 0.42m、肩腕 0.85m） |

**结论（组内共识）**：在 **RTMPose-m + RTMDet-m + 720p** 下，后端约束 **边际有限**；现场 **有大量视频**，适合 **域内微调 det/pose** 后在实验仓 **替换 ONNX A/B**。

---

## 4. 目前推理与配置现状

### 4.1 现场常用运行时（`localdata/runtime_config.json` 示例）

| 项 | 典型值 |
|----|--------|
| `models.backend` | `rtmpose_m` |
| `models.det` | `m`（RTMDet-m，640×640 letterbox） |
| `inference.height` | **720** |
| `inference.pose_frame_interval` | **2**（约 15fps 源 → **~7.5Hz pose**） |
| `inference.frame_rate` | 15 |

`app_config.json` 默认可能是 `rtmpose_t` + nano + 480p；**以 runtime 覆盖为准**。

### 4.2 模型与部署路径

- 预设与 URL：`services/inference_backends/model_registry.py`
- 权重目录：`localdata/models/rtmpose_onnx/`
  - `rtmdet_m/end2end.onnx`（或 `rtmdet_nano`）
  - `rtmpose_m/end2end.onnx`（t/s/m 同结构，pose 输入 **256×192** SimCC）
- 下载脚本：`scripts/download-rtmpose-onnx-weights.sh`
- 推理容器：GPU ONNX 镜像见 `scripts/build-inference-lite-gpu-onnx-image.sh`
- **本仓无训练脚本**；微调产物需 **自行导出 ONNX** 后替换上述 `end2end.onnx` 并重启 infer。

### 4.3 Worker / 3D（替换 ONNX 后 **无需改** 的部分）

- 抬升：`services/event_engine/dualcam_processor.py` → `_lift_joints` → `lift_point`
- 五官：`FACE_JOINTS = {0..4}` **跳过抬升**
- 贴墙：`contact_slots` 使用 **clamp 后** `xyz[9/10]`
- Hold：`dualcam/pose_timing.py` 中 `HOLD_SEC`、`pose_time()`
- 前端 3D：`web/src/lib/cocoSkeleton.js` 的 `AISLE_3D_EDGES` / `AISLE_3D_HEAD_JOINTS`，`AisleScene3D.jsx`

拓扑保持 **COCO-17**，微调时不要改关节定义。

---

## 5. 为什么要做域内训练（论证摘要）

1. **域偏移**：固定安装俯角、巷道景深、货架遮挡、工装 → 与 Body7/COCO 预训练分布不一致；已用 **最大档 t/s/m + det m** 仍不够。
2. **误差传播**：腕 2D 偏几像素 → 双路交会 gap 变大 → `lift_point` fallback → 3D 扭曲或抖；**修 lift 治不了标定意义上的 2D 误差**。
3. **数据条件**：现场 **视频量充足**，瓶颈在 **标注与训练流水线**，不在采数。
4. **工程边界**：直播 **不能** 再开整窗平滑（吞吐）；整体抬升 **已证伪**；下一步性价比最高的是 **2D 模型域适应**。

---

## 6. 建议在训练仓实施的路径

### 6.1 训练对象

- **优先**：**RTMPose-m** 微调（COCO-17），框若已稳则先只训 pose。
- **并行或第二阶段**：**RTMDet-m** 微调（person），若漏检/误检仍多。

### 6.2 数据

- 从现场视频 **按场景分层抽帧**（L/R、远/近、入架、遮挡、多人），分辨率与线上一致（**720p**）。
- **Hard mining**：用当前模型跑一遍，专标 **低分腕、高 3D gap、误报 clip**。
- 标注：COCO keypoints + person bbox；划分按 **巷道/日期/相机**，避免同 clip 泄漏进 val。
- 规模：**POC 500–1500 张/路** 看趋势；上线级 **3k–10k+** 多巷道。

### 6.3 训练与导出

- 环境：**MMPose v1 + MMDetection**，与 OpenMMLab `onnx_sdk` 同系列 config。
- 加载官方预训练 → 小学习率微调 → **导出与现网 shape 一致的 ONNX**。
- 在训练机用 rtmlib/ONNXRuntime 抽测 val，再拷入本仓 `localdata/models/...` 做 **infer A/B**。

### 6.4 验收指标（建议写进训练项目 README）

| 层级 | 指标 |
|------|------|
| 2D val | 腕/肘 PCK 或 OKS，相对 **现网 ONNX** 提升 |
| 2D 在线 | 固定 clip 上腕 **score、帧间像素 std** |
| 3D | 腕 **triangulation gap** 中位数/P95（可离线 replay worker 逻辑） |
| 业务 | 贴墙 **误报/漏报**（有事件标注更佳） |
| 性能 | infer **det_ms/pose_ms** 不明显退化 |

---

## 7. 仓库约束（接手时注意）

- **不要改** `/home/hqit/workspace/visual-dps`、`visual-dps-pick-state` 源仓（只读参考）。
- 实验产物、模型、标注建议落在 **本仓 `localdata/`** 或训练仓自有目录；全量离线包流程见 skill `visual-dps-offline-package`（本议题通常 **只换 ONNX** 即可）。
- Git：本仓近期相关提交主题包括结构化抬升（已回滚）、贴墙/clamp 对齐、五官不绘制、移除未使用平滑。

---

## 8. 关键文件索引

| 主题 | 路径 |
|------|------|
| 2D→3D 抬升 | `dualcam/lift.py`（`lift_point`） |
| Worker 贴墙 | `services/event_engine/dualcam_processor.py` |
| 模型预设 | `services/inference_backends/model_registry.py` |
| 推理循环 | `services/inference_service.py` |
| 运行时覆盖 | `localdata/runtime_config.json` |
| 3D 前端 | `web/src/components/AisleScene3D.jsx`，`web/src/lib/cocoSkeleton.js` |
| 单测 | `tests/test_lift_point.py`，`tests/test_dualcam_contact.py` |
| 0902 吞吐与平滑结论 | `docs/DAILY-2026-09-02.md` |

---

## 9. 联系上下文（可选阅读）

- 碰撞语义：**3D `contact_slots`**，非 2D 货框 legacy（未成组单路除外）。
- 双路 infer 需 **同时启动**；worker 分片键 `aisle_id`。
- 若 POC 成功：在实验仓替换 ONNX → 现场看 3D 与告警 → 再考虑多巷道扩标与镜像固化。

---

*文档由 visual-dps-dualcam-exp 维护，供跨仓库训练接手；训练实现细节以 MMPose / MMDetection 官方文档为准。*
