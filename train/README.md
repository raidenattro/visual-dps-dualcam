# train

用 collector 标真拟合 PickStateScorer 权重。

## 脚本

| 脚本 | 作用 | 产物 |
|------|------|------|
| `build_dataset.py` | 28-clip → 帧级特征表 | `output/train/dataset_v1.npz` + `.meta.json` |
| `fit_logistic.py` | L2 logistic + GroupKFold | `output/train/logistic_v1/`（model / importance / cv_metrics） |

```bash
.venv/bin/python train/build_dataset.py
.venv/bin/python train/fit_logistic.py
```

## 标签定义

帧号（`source_frame_idx`）出现在 `event_review.json` 的 `verified_true` 条目中即为正样本。

## 采样策略

每帧只取一个人：优先有货框命中者，其次腕抬升角最大者。
负样本全保留「有货框命中」的硬负例，无命中的负例按稳定哈希抽 ~1/12，避免正负比失衡。

## 注意

- 分组必须用 `record_id`（`GroupKFold`），只有 **28 组**，慎用高容量模型
- 当前是同集训练 + 评估，报告数字偏乐观
