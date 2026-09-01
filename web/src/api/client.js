const fetchOpts = { credentials: 'include' };

async function parseJson(resp) {
  const data = await resp.json();
  if (resp.status === 401 && !String(window.location.pathname).startsWith('/login')) {
    window.location.assign(`/login?from=${encodeURIComponent(window.location.pathname)}`);
    throw new Error('未登录或会话已过期');
  }
  return data;
}

export async function apiGet(path) {
  const resp = await fetch(path, fetchOpts);
  return parseJson(resp);
}

export async function apiPost(path, body) {
  const resp = await fetch(path, {
    ...fetchOpts,
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return parseJson(resp);
}

export async function apiDelete(path, body) {
  const resp = await fetch(path, {
    ...fetchOpts,
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  return parseJson(resp);
}

export async function apiPut(path, body) {
  const resp = await fetch(path, {
    ...fetchOpts,
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return parseJson(resp);
}

export async function apiPatch(path, body) {
  const resp = await fetch(path, {
    ...fetchOpts,
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return parseJson(resp);
}

export function thumbnailUrl(cameraId, lastFrameAt, cacheBust) {
  const t = cacheBust ?? (lastFrameAt ? Math.floor(lastFrameAt) : Date.now());
  return `/api/cameras/${encodeURIComponent(cameraId)}/thumbnail?t=${t}`;
}

/** MJPEG 实时预览流（浏览器 img 直接播放，勿轮询抓帧） */
export function cameraStreamUrl(cameraId, height = 480) {
  const h = Number(height) || 480;
  return `/api/cameras/${encodeURIComponent(cameraId)}/stream?height=${h}`;
}

export function cameraPlaybackUrl(cameraId) {
  return `/api/cameras/${encodeURIComponent(cameraId)}/playback`;
}

/** SSE：推理 overlay（骨架/碰撞），与姿态帧同频推送 */
export function openCameraLiveStream(cameraId, { onFrame, onReady, onError, stallMs = 4000 } = {}) {
  const url = `/api/cameras/${encodeURIComponent(cameraId)}/live/stream`;
  let es = null;
  let closed = false;
  let lastAt = Date.now();

  const bind = (socket) => {
    const handleFrame = (ev) => {
      lastAt = Date.now();
      try {
        const data = JSON.parse(ev.data);
        onFrame?.(data);
      } catch (e) {
        onError?.(e);
      }
    };
    socket.addEventListener('frame', handleFrame);
    socket.addEventListener('ready', (ev) => {
      lastAt = Date.now();
      try {
        onReady?.(JSON.parse(ev.data));
      } catch {
        onReady?.(null);
      }
    });
    socket.onerror = () => {
      onError?.(new Error('实时 overlay 连接中断'));
    };
  };

  const connect = () => {
    if (closed) return;
    if (es) {
      es.close();
      es = null;
    }
    es = new EventSource(url);
    bind(es);
  };

  connect();
  const wait = Math.max(1000, Number(stallMs) || 4000);
  const tick = window.setInterval(() => {
    if (closed) return;
    if (Date.now() - lastAt < wait) return;
    lastAt = Date.now();
    connect();
  }, Math.min(2000, wait));

  return () => {
    closed = true;
    window.clearInterval(tick);
    if (es) es.close();
    es = null;
  };
}

export function formatDuration(seconds) {
  const s = Math.max(0, parseInt(seconds, 10) || 0);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return [h, m, sec].map((n) => String(n).padStart(2, '0')).join(':');
}
