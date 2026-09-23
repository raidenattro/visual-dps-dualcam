#!/usr/bin/env bash
# 144/24 同侧：重拼 src → 左右半幅 RTMDet-M + RTMPose-M → dump_skel3d
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYDPS=/home/hqit/miniconda3/envs/visual-dps/bin/python
NV=/home/hqit/miniconda3/envs/visual-dps/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$(find "$NV" -type d -name lib 2>/dev/null | paste -sd:)" || true

echo "== stitch src.mp4 =="
.venv/bin/python scripts/stitch_dualcam_src.py

echo "== pose infer (stride=1, half-frame L/R) =="
"$PYDPS" scripts/dualcam_lift.py --stride 1 --skip-analyze \
  --out-npz output/dualcam/poses_test_144_24.npz

echo "== skel3d =="
.venv/bin/python scripts/dump_skel3d.py \
  --calib output/calib/dual_144-24.json \
  --npz output/dualcam/poses_test_144_24.npz

echo "done. 重启 serve_dualcam 并强刷 /play"
