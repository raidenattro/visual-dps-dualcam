# 共享：镜像加速 URL、跳过已有文件、断点续传（由 download-*-weights.sh 引用）
# 环境变量：GITHUB_PROXY_BASE、OPENMMLAB_MIRROR_BASE、DOWNLOAD_HTTP_PROXY

vdps_github_url() {
  url="$1"
  if [ -n "${GITHUB_PROXY_BASE:-}" ]; then
    printf '%s/%s' "${GITHUB_PROXY_BASE%/}" "${url}"
  else
    printf '%s' "${url}"
  fi
}

vdps_openmmlab_base() {
  if [ -n "${OPENMMLAB_MIRROR_BASE:-}" ]; then
    printf '%s' "${OPENMMLAB_MIRROR_BASE%/}"
  else
    printf '%s' "https://download.openmmlab.com"
  fi
}

# 若目标已满足最小字节则跳过；否则 wget/curl 断点续传到 dest
vdps_fetch_file() {
  url="$1"
  dest="$2"
  min_bytes="${3:-1}"
  label="${4:-$(basename "$dest")}"

  if [ -f "${dest}" ]; then
    size="$(wc -c < "${dest}" | tr -d ' ')"
    if [ "${size}" -ge "${min_bytes}" ]; then
      echo "==> 已有且完整 ${dest} (${size} >= ${min_bytes} bytes)"
      return 0
    fi
    echo "==> 重新下载 ${label}（不完整: ${size} < ${min_bytes} bytes）"
    rm -f "${dest}"
  fi

  part="${dest}.part"
  if [ -f "${part}" ]; then
    psize="$(wc -c < "${part}" | tr -d ' ')"
    echo "==> 续传 ${label}（已有 .part ${psize} bytes）"
  else
    echo "==> 下载 ${label}"
  fi
  echo "    ${url}"

  if command -v wget >/dev/null 2>&1; then
    if [ -n "${DOWNLOAD_HTTP_PROXY:-}" ]; then
      http_proxy="${DOWNLOAD_HTTP_PROXY}" https_proxy="${DOWNLOAD_HTTP_PROXY}" \
        wget -c --timeout=120 --tries=3 -O "${part}" "${url}"
    else
      wget -c --timeout=120 --tries=3 -O "${part}" "${url}"
    fi
    mv -f "${part}" "${dest}"
  elif command -v curl >/dev/null 2>&1; then
    if [ -n "${DOWNLOAD_HTTP_PROXY:-}" ]; then
      curl -fL --retry 3 --connect-timeout 30 -C - -o "${part}" \
        -x "${DOWNLOAD_HTTP_PROXY}" "${url}"
    else
      curl -fL --retry 3 --connect-timeout 30 -C - -o "${part}" "${url}"
    fi
    mv -f "${part}" "${dest}"
  else
    echo "错误: 需要 wget 或 curl" >&2
    return 1
  fi

  if [ ! -f "${dest}" ]; then
    echo "错误: 下载失败 ${dest}" >&2
    return 1
  fi
  size="$(wc -c < "${dest}" | tr -d ' ')"
  if [ "${size}" -lt "${min_bytes}" ]; then
    echo "错误: ${dest} 下载后仍不完整 (${size} < ${min_bytes} bytes)" >&2
    rm -f "${dest}"
    return 1
  fi
  echo "    -> ${dest} (${size} bytes, 完整)"
}
