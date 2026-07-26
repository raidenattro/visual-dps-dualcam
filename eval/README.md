# eval

离线回归：本仓 pipeline 跑 28-clip → upload JSON 写到 `output/export/<包名>/`，再调 collector 评估。

```bash
# 1) 导出（本仓 venv）
cd /home/hqit/workspace/visual-dps-pick-state
.venv/bin/python scripts/export_manifest28.py --config configs/<配置>.json --out output/export/<包名>

# 2) 评估（collector 目录 + 系统 python3，需 fastapi）
cd /home/hqit/workspace/visual-dps-data-collector
python3 scripts/data/evaluate_inference_upload.py --in-place \
  --dirs /home/hqit/workspace/visual-dps-pick-state/output/export/<包名>

# 3) 对比
python3 scripts/data/compare_export_false_alarms.py \
  --baseline /home/hqit/workspace/visual-dps-pick-state/output/export/pickstate-nogate-prod-test \
  --experiment /home/hqit/workspace/visual-dps-pick-state/output/export/<包名> \
  --out-dir /home/hqit/workspace/visual-dps-pick-state/output/compare
```

## 基准选择

对照**固定用本仓 `pickstate-nogate-prod-test`**，不要用 collector 的 `rule-baseline-local-prod-test`：
两者货框标注来源不同（record 自带 `manifest.annotation` vs reflection 临时文件），
直接对比会把标注差异算进算法效果。

默认 **不写回** collector `timeline.parquet`。
