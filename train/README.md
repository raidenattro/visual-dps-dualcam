# train

用 collector 标真（`verified_true` 段为正，段外为负）拟合 PickStateScorer 权重。

计划：

1. 从 baseline manifest 28 条导出帧级特征表（只读 parquet）
2. 逻辑回归 / 小模型：特征 → `pick_score`
3. 规则 expert 输出也可作为输入特征（规则保留、权重学会）

本目录脚本待下一步实现。
