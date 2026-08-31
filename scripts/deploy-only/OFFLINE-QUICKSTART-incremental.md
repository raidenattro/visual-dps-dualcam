# Visual-DPS 增量离线包（UI + 双路 3D event-worker）

本包只升级 **UI** 与 **event-worker**（`DualcamRedisWorker` / `contact_slots`）。不含 redis / mediamtx / 推理镜像。

## 安装

```bash
./verify-images.sh
./install.sh --host <局域网IP> --stop-infer
```

`--worker-2` 已废弃，传入会被忽略。不要再启动 `visual-dps-event-worker-2`。

访问：`http://<IP>:<UI_PORT>/`（默认 8045）。
