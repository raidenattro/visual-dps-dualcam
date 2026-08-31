# YOLO Pose 后端

已实现 `yolo_pose`（Ultralytics **YOLO26n/s/m/l-pose**），与 `rtmpose_onnx`（**RTMPose t/s/m**）并列，通过 `models.backend` 预设选择。

| preset id | 说明 |
|-----------|------|
| `yolo26n_pose` | 路数多、延迟低 |
| `yolo26s_pose` | 默认推荐平衡 |
| `yolo26m_pose` / `yolo26l_pose` | 更高精度 |

需 **lite** / **lite-gpu-onnx** 镜像（含 `ultralytics`）。权重目录：`localdata/models/yolo_pose/`。

构建前在宿主机下载权重（推荐 `./scripts/download-model-weights.sh`，YOLO 可走 `.env` 的 `GITHUB_PROXY_BASE`）：

```bash
./scripts/download-model-weights.sh
./scripts/build-inference-lite-image.sh
```
