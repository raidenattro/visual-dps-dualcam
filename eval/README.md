# eval

离线回归：对本仓 pipeline 跑 28-clip → 写出 upload JSON 到 `output/export/<run>/`  
再调用 collector：

```bash
python /home/hqit/workspace/visual-dps-data-collector/scripts/data/evaluate_inference_upload.py \
  --dir /home/hqit/workspace/visual-dps-pick-state/output/export/<run> --in-place

python /home/hqit/workspace/visual-dps-data-collector/scripts/data/compare_export_false_alarms.py \
  --baseline /home/hqit/workspace/visual-dps-data-collector/localdata/export/rule-baseline-local-prod-test \
  --experiment /home/hqit/workspace/visual-dps-pick-state/output/export/<run>
```

默认 **不写回** collector `timeline.parquet`。
