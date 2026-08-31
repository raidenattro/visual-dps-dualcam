#!/usr/bin/env bash
# 生成本项目 Docker 镜像 tag：YYYYMMDD-HHMMSS-<git短哈希>（同日多次构建可区分）
# 用法: source scripts/lib/docker-image-tag.sh && TAG=$(visual_dps_image_tag)

visual_dps_image_tag() {
  local date_part time_part git_part
  date_part="$(date +%Y%m%d)"
  time_part="$(date +%H%M%S)"
  git_part="nogit"
  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_part="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
  fi
  printf '%s-%s-%s' "${date_part}" "${time_part}" "${git_part}"
}

# 为 compose 服务镜像名打 tag：visual-dps-inference-lite-gpu-onnx:20260528-120000-abc1234
visual_dps_tag_image() {
  local repo="$1"
  local tag="${2:-$(visual_dps_image_tag)}"
  printf '%s:%s' "${repo}" "${tag}"
}
