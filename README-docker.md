# Docker Compose 部署（推荐）

详见 [docs/DEPLOY.md](./docs/DEPLOY.md)。

```bash
cp .env.example .env   # 编辑 REDIS_PASSWORD
docker compose build
docker compose up -d
```

- UI: `http://127.0.0.1:8045`
- **使用手册（含 FFmpeg 推流）**: [docs/USER_MANUAL.md](./docs/USER_MANUAL.md)
- MediaMTX RTSP: `rtsp://127.0.0.1:8554/<path>`
- 监控页支持 MJPEG / HLS / WebRTC（见 [docs/DEPLOY.md](./docs/DEPLOY.md) 播放与排障）

推理容器由总览页启动，需先 `docker compose build visual-dps-inference-lite`（或 gpu 版）。  
变更记录：[docs/CHANGELOG.md](./docs/CHANGELOG.md)
