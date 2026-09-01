"""根据摄像头配置生成 MediaMTX paths 配置。"""

import json
import os
import re
import urllib.error
import urllib.request
from typing import List
from urllib.parse import urlparse

# 可作 YAML 未加引号映射键：字母开头，仅含字母数字 _ -
_YAML_PLAIN_PATH_KEY_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")
_YAML_RESERVED_PATH_KEYS = frozenset(
    {"true", "false", "yes", "no", "on", "off", "null", "~"}
)


def mediamtx_yaml_path_key(path: str) -> str:
    """MediaMTX paths 下的 YAML 键名。纯数字或以数字开头须加引号，否则会被解析为数字类型。"""
    s = str(path or "").strip()
    if not s:
        return '""'
    if not _YAML_PLAIN_PATH_KEY_RE.match(s) or s.lower() in _YAML_RESERVED_PATH_KEYS:
        return json.dumps(s, ensure_ascii=False)
    return s

MEDIAMTX_RTSP_HOST = os.environ.get("MEDIAMTX_RTSP_HOST", "127.0.0.1")
MEDIAMTX_INTERNAL_HOST = os.environ.get("MEDIAMTX_INTERNAL_HOST", "mediamtx")
MEDIAMTX_RTSP_PORT = int(os.environ.get("MEDIAMTX_RTSP_PORT", "8554"))
MEDIAMTX_HLS_PORT = int(os.environ.get("MEDIAMTX_HLS_PORT", "8888"))
MEDIAMTX_WEBRTC_PORT = int(os.environ.get("MEDIAMTX_WEBRTC_PORT", "8889"))
MEDIAMTX_PUBLIC_HOST = os.environ.get("MEDIAMTX_PUBLIC_HOST", "127.0.0.1")
MEDIAMTX_WEBRTC_ICE_PORT = int(os.environ.get("MEDIAMTX_WEBRTC_ICE_PORT", "8189"))
MEDIAMTX_API_URL = os.environ.get("MEDIAMTX_API_URL", "http://127.0.0.1:9997").rstrip("/")
MEDIAMTX_API_TIMEOUT = float(os.environ.get("MEDIAMTX_API_TIMEOUT", "3"))

SOURCE_RTSP_PULL = "rtsp_pull"
SOURCE_PUBLISHER = "publisher"
SOURCE_EXTERNAL = "external"

MANAGED_SOURCE_TYPES = {SOURCE_RTSP_PULL, SOURCE_PUBLISHER}


def build_playback_url(path: str, host: str | None = None, port: int | None = None) -> str:
    h = host or MEDIAMTX_RTSP_HOST
    p = port or MEDIAMTX_RTSP_PORT
    slug = str(path or "").strip().strip("/")
    return f"rtsp://{h}:{p}/{slug}"


def build_hls_url(path: str, host: str | None = None, port: int | None = None) -> str:
    slug = str(path or "").strip().strip("/")
    if not slug:
        return ""
    h = host or MEDIAMTX_PUBLIC_HOST
    p = port or MEDIAMTX_HLS_PORT
    return f"http://{h}:{p}/{slug}/index.m3u8"


def build_webrtc_whep_url(path: str, host: str | None = None, port: int | None = None) -> str:
    slug = str(path or "").strip().strip("/")
    if not slug:
        return ""
    h = host or MEDIAMTX_PUBLIC_HOST
    p = port or MEDIAMTX_WEBRTC_PORT
    return f"http://{h}:{p}/{slug}/whep"


def build_camera_playback_urls(camera_id: str, path: str, *, proxied: bool = True) -> dict:
    """返回浏览器可访问的播放地址；默认走 UI 同源代理，避免跨域与 Docker 网络问题。"""
    slug = str(path or "").strip().strip("/")
    cid = str(camera_id or "").strip()
    if not slug or not cid:
        return {"hls": "", "whep": ""}
    if proxied:
        return {
            "hls": f"/api/cameras/{cid}/hls/index.m3u8",
            "whep": f"/api/cameras/{cid}/whep",
        }
    return {
        "hls": build_hls_url(slug),
        "whep": build_webrtc_whep_url(slug),
    }


def is_mediamtx_managed(camera: dict) -> bool:
    return camera.get("source_type") in MANAGED_SOURCE_TYPES and camera.get("enabled", True)


def is_mediamtx_playback_available(camera: dict) -> bool:
    """是否可走 MediaMTX 的 HLS/WebRTC（含 external 但 RTSP 指向本机 MediaMTX 路径）。"""
    if not camera.get("enabled", True):
        return False
    if camera.get("source_type") in MANAGED_SOURCE_TYPES:
        return True
    path = str(camera.get("path") or camera.get("id") or "").strip()
    url = str(camera.get("url") or "").strip()
    if not path or not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme.lower() != "rtsp":
        return False
    host = (parsed.hostname or "").lower()
    port = parsed.port if parsed.port is not None else MEDIAMTX_RTSP_PORT
    local_hosts = {
        MEDIAMTX_RTSP_HOST.lower(),
        MEDIAMTX_PUBLIC_HOST.lower(),
        MEDIAMTX_INTERNAL_HOST.lower(),
        "127.0.0.1",
        "localhost",
        "mediamtx",
    }
    if host not in local_hosts or port != MEDIAMTX_RTSP_PORT:
        return False
    return path_from_url(url) == path


def path_from_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    parts = [p for p in parsed.path.split("/") if p]
    return parts[-1] if parts else ""


def cameras_needing_mediamtx_paths(cameras: List[dict]) -> List[dict]:
    """需在 MediaMTX 上存在 path 的摄像头（托管源 + 指向本机 MTX 的 external）。"""
    out: List[dict] = []
    seen: set[str] = set()
    for cam in cameras:
        if not cam.get("enabled", True):
            continue
        path = str(cam.get("path") or cam.get("id") or "").strip()
        if not path or path in seen:
            continue
        if is_mediamtx_managed(cam) or is_mediamtx_playback_available(cam):
            seen.add(path)
            out.append(cam)
    return out


def generate_mediamtx_yaml(cameras: List[dict]) -> str:
    lines = [
        "# 由 visual-dps 根据摄像头配置自动生成，请勿手改（可在 Dashboard 管理）",
        "",
        "logLevel: info",
        "api: yes",
        "apiAddress: :9997",
        f"rtspAddress: :{MEDIAMTX_RTSP_PORT}",
        "rtmpAddress: ''",
        f"hlsAddress: :{MEDIAMTX_HLS_PORT}",
        "hlsVariant: mpegts",
        f"webrtcAddress: :{MEDIAMTX_WEBRTC_PORT}",
        "webrtcAllowOrigin: '*'",
        "# Docker：勿向浏览器通告容器网桥 IP，仅通告宿主机可访问地址",
        "webrtcIPsFromInterfaces: false",
        f"webrtcAdditionalHosts: ['{MEDIAMTX_PUBLIC_HOST}']",
        f"webrtcLocalUDPAddress: :{MEDIAMTX_WEBRTC_ICE_PORT}",
        f"webrtcLocalTCPAddress: :{MEDIAMTX_WEBRTC_ICE_PORT}",
        "srtAddress: ''",
        "",
        "# compose 内 UI 经 Docker 网桥访问 API，需放开私网段（默认仅 localhost 免鉴权）",
        "authInternalUsers:",
        "  - user: any",
        "    pass:",
        "    ips: []",
        "    permissions:",
        "      - action: publish",
        "        path:",
        "      - action: read",
        "        path:",
        "      - action: playback",
        "        path:",
        "  - user: any",
        "    pass:",
        "    ips: ['127.0.0.1', '::1', '172.16.0.0/12', '10.0.0.0/8', '192.168.0.0/16']",
        "    permissions:",
        "      - action: api",
        "      - action: metrics",
        "      - action: pprof",
        "",
        "paths:",
    ]

    mtx_paths = cameras_needing_mediamtx_paths(cameras)
    if not mtx_paths:
        lines.append("  # 暂无需要在 MediaMTX 注册的流路径")
        return "\n".join(lines) + "\n"

    for cam in mtx_paths:
        path = str(cam.get("path") or cam.get("id") or "").strip()
        if not path:
            continue
        source_type = cam.get("source_type")
        lines.append(f"  {mediamtx_yaml_path_key(path)}:")

        if source_type == SOURCE_EXTERNAL:
            lines.append("    source: publisher")
            lines.append("")
            continue

        if source_type == SOURCE_RTSP_PULL:
            pull_url = str(cam.get("pull_url") or "").strip()
            lines.append(f"    source: {pull_url}")
        elif source_type == SOURCE_PUBLISHER:
            lines.append("    source: publisher")
        lines.append("")

    return "\n".join(lines)


def camera_to_path_conf(cam: dict) -> dict:
    source_type = cam.get("source_type")
    if source_type == SOURCE_EXTERNAL:
        if is_mediamtx_playback_available(cam):
            return {"source": "publisher"}
        return {}
    if source_type == SOURCE_RTSP_PULL:
        pull_url = str(cam.get("pull_url") or "").strip()
        # 空 source 会把已有 path 配坏，推流入口应保留 publisher
        return {"source": pull_url} if pull_url else {"source": "publisher"}
    if source_type == SOURCE_PUBLISHER:
        return {"source": "publisher"}
    if is_mediamtx_playback_available(cam):
        return {"source": "publisher"}
    return {}


def _mediamtx_api(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{MEDIAMTX_API_URL}{path}"
    payload = None
    headers = {}
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=payload, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=MEDIAMTX_API_TIMEOUT) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def fetch_path_snapshot() -> dict:
    """一次拉取 MediaMTX 已配置 path 与运行时 ready 状态。"""
    configured: set[str] = set()
    runtime: dict = {}
    api_ok = False
    if not MEDIAMTX_API_URL:
        return {"configured": configured, "runtime": runtime, "api_ok": False}
    try:
        cfg = _mediamtx_api("GET", "/v3/config/paths/list?itemsPerPage=200")
        api_ok = True
        for it in cfg.get("items") or []:
            name = str(it.get("name") or it.get("confName") or "").strip()
            if name:
                configured.add(name)
    except Exception:
        pass
    try:
        live = _mediamtx_api("GET", "/v3/paths/list?itemsPerPage=200")
        api_ok = True
        for it in live.get("items") or []:
            name = str(it.get("name") or "").strip()
            if name:
                runtime[name] = it
    except Exception:
        pass
    return {"configured": configured, "runtime": runtime, "api_ok": api_ok}


def describe_stream_error(
    cam: dict,
    snap: dict | None,
    *,
    online: bool,
    probed: bool = True,
) -> str:
    """离线原因：未注册 / 未推流 / 编码无法解码。在线则空。

    probed=False 表示尚未实际读帧：MediaMTX 已 ready 时不要误报编码错误。
    """
    if online:
        return ""
    if cam.get("enabled") is False:
        return "该路已停用"
    path = str(cam.get("path") or cam.get("id") or "").strip() or "unknown"
    source_type = str(cam.get("source_type") or "")
    snap = snap or {}
    configured = snap.get("configured")
    runtime = snap.get("runtime") or {}
    api_ok = bool(snap.get("api_ok"))
    item = runtime.get(path) or {}
    ready = bool(item.get("ready"))

    if api_ok and isinstance(configured, set) and path not in configured:
        return f"通道 {path} 未注册（path is not configured）"
    if api_ok and not ready:
        if source_type == SOURCE_PUBLISHER or not source_type or (
            source_type == SOURCE_EXTERNAL and is_mediamtx_playback_available(cam)
        ):
            return f"尚未推流到 {path}（no one is publishing）"
        if source_type == SOURCE_RTSP_PULL:
            pull = str(cam.get("pull_url") or "").strip()
            return f"上游拉流未就绪" + (f"：{pull}" if pull else "")
        return f"视频流未就绪（{path}）"
    if api_ok and ready:
        if probed:
            return "画面无法解码，请检查推流编码（需 H.264）"
        return ""
    return "无法连接视频流，请检查地址与网络"


def reload_mediamtx_runtime(cameras: List[dict]) -> dict:
    """通过 Control API 热更新托管路径；不可用时静默跳过（仍依赖配置文件热重载）。"""
    mtx_paths = cameras_needing_mediamtx_paths(cameras)
    if not MEDIAMTX_API_URL:
        return {"reloaded": False, "skipped": True, "reason": "api_disabled"}
    desired_names = {str(c.get("path") or c.get("id") or "").strip() for c in mtx_paths}
    desired_names.discard("")
    # 空列表禁止删路径：否则一次空配置会把 cam1/cam2 从运行时清掉，推流即报 not configured
    if not desired_names:
        return {"reloaded": False, "skipped": True, "reason": "no_desired_paths"}

    try:
        listed = _mediamtx_api("GET", "/v3/config/paths/list?itemsPerPage=100")
        current_names = {
            str(item.get("name") or item.get("confName") or "").strip()
            for item in listed.get("items", [])
            if item.get("name") or item.get("confName")
        }

        for name in current_names:
            if name in desired_names or name in ("all_others", "all", ""):
                continue
            try:
                _mediamtx_api("DELETE", f"/v3/config/paths/delete/{name}")
            except urllib.error.HTTPError:
                pass

        for cam in mtx_paths:
            name = str(cam.get("path") or cam.get("id") or "").strip()
            if not name:
                continue
            conf = camera_to_path_conf(cam)
            if not conf.get("source"):
                continue
            try:
                _mediamtx_api("POST", f"/v3/config/paths/replace/{name}", conf)
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    _mediamtx_api("POST", f"/v3/config/paths/add/{name}", conf)
                else:
                    raise

        return {"reloaded": True, "paths": sorted(desired_names)}
    except Exception as exc:
        return {"reloaded": False, "skipped": True, "reason": str(exc)}


def _mediamtx_config_needs_rewrite(config_path: str) -> bool:
    """旧版 apiAddress 绑定 127.0.0.1 时，compose 内其它容器无法访问 API。"""
    if not os.path.isfile(config_path):
        return True
    try:
        with open(config_path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return True
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("apiAddress:"):
            continue
        if ":9997" in s and "127.0.0.1" not in s:
            return False
        return True
    return True


def write_mediamtx_config(config_path: str, cameras: List[dict]) -> str:
    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    content = generate_mediamtx_yaml(cameras)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    return config_path


def ensure_mediamtx_config(config_path: str, cameras: List[dict]) -> dict | None:
    """启动时修正遗留 mediamtx.yml（仅写文件，不调用运行时 API）。"""
    if not _mediamtx_config_needs_rewrite(config_path):
        return None
    path = write_mediamtx_config(config_path, cameras)
    return {
        "rewritten": True,
        "config_path": path,
        "hint": "已修正 mediamtx.yml 的 apiAddress；若 MediaMTX 已在运行请 docker compose restart mediamtx",
    }


def sync_mediamtx_config(config_path: str, cameras: List[dict]) -> dict:
    path = write_mediamtx_config(config_path, cameras)
    reload = reload_mediamtx_runtime(cameras)
    mtx_paths = cameras_needing_mediamtx_paths(cameras)
    if reload.get("reloaded"):
        hint = "视频流设置已自动生效。"
    else:
        hint = "视频流配置已更新；若画面未变化，请稍候片刻。"
    return {
        "config_path": path,
        "managed_paths": [c.get("path") or c.get("id") for c in mtx_paths],
        "reload": reload,
        "reload_hint": hint,
    }
