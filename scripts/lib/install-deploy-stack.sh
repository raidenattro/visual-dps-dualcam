#!/usr/bin/env bash
# install.sh 共用：启动 compose 栈（须已定义 compose_cmd）
# 只启双路 3D visual-dps-event-worker；停掉本机残留的 pick_state worker-2。

visual_dps_compose_up_stack() {
  echo "==> 启动服务（双路 3D event-worker）"
  compose_cmd up -d
  compose_cmd stop visual-dps-event-worker-2 visual-dps-event-worker-2-b 2>/dev/null || true
  docker stop visual-dps-event-worker-2 visual-dps-event-worker-2-b 2>/dev/null || true
}
