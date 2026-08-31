# Visual-DPS 全量离线包（v2 布局）

> 新机 / 整栈恢复只看本文与 `docs/OFFLINE-DEPLOY-CHECKLIST.md`。

## 目录结构

```text
visual-dps-offline-complete-<时间戳>/
├── docker-images/bundle.tar   # docker save（勿再 gzip）
├── app/                       # compose、.env、配置（无 models）
├── weights/                   # 8 个权重 + SHA256SUMS
├── install.sh
├── verify-package.sh
└── PACKAGE_INFO.txt
```

## 源机打全量包

```bash
# 推荐：构建 + 校验 + 打包
./scripts/export-offline-one-shot.sh
# 产出: dist/visual-dps-offline-complete-<ts>/

# 或仅打包（镜像已齐）
./scripts/download-model-weights.sh
./scripts/preflight-offline-export.sh --inference all
./scripts/export-offline-complete.sh

# 需重建 UI 镜像时
./scripts/export-offline-complete.sh --rebuild-ui

# 外传可选（pigz 多核，比 tar -czf 快很多）
./scripts/export-offline-complete.sh --archive tar.gz --compress pigz

# 仅未压缩 tar 外壳（仍不 gzip bundle）
./scripts/export-offline-complete.sh --archive tar
```

## 目标机安装

```bash
# 若用了 --archive tar.gz
tar xzf visual-dps-offline-complete-*.tar.gz

cd visual-dps-offline-complete-*/
./verify-package.sh
# 编辑 app/.env 设置 REDIS_PASSWORD、MEDIAMTX_PUBLIC_HOST、端口等
./install.sh --host <本机IP> --stop-infer
```

`install.sh` 会在 `compose up` 前按 `app/.env` 与 `camera_ips.json` 自动生成 `localdata/mediamtx.yml`。日后改 WebRTC/HLS 或公网 IP：改 `app/.env` 后在 UI「应用 MediaMTX 配置」，或执行 `app/deploy/regenerate-mediamtx-config.sh` 并重启 `mediamtx`。

权重默认从 `weights/` 安装到 `app/localdata/models/`。旧版单 tar.gz 包若含 `app/localdata/models` 仍兼容。

## 体积说明

| 部分 | 约 |
|------|-----|
| bundle.tar（7 镜像） | ~14G |
| weights/ | ~276M |
| app/ | 数 MB |

慢的根源是对 14G 做 `tar -czf`；v2 **默认不压整包**，只 rsync/拷贝目录。

详细命令见 `docs/OFFLINE-DEPLOY-CHECKLIST.md`、`.cursor/skills/visual-dps-offline-package/SKILL.md`。
