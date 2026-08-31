export const STREAM_HEIGHTS = [320, 480, 720];
export const STREAM_FORMATS = ['mjpeg', 'hls', 'webrtc'];
export const DEFAULT_STREAM_PREFS = { format: 'webrtc', height: 320 };

function storageKey(cameraId) {
  return `monitorStreamPrefs:v2:${cameraId}`;
}

export function loadStreamPrefs(cameraId) {
  if (!cameraId) return { ...DEFAULT_STREAM_PREFS };
  try {
    const raw = localStorage.getItem(storageKey(cameraId));
    if (!raw) return { ...DEFAULT_STREAM_PREFS };
    const parsed = JSON.parse(raw);
    const height = STREAM_HEIGHTS.includes(parsed.height) ? parsed.height : DEFAULT_STREAM_PREFS.height;
    const format = STREAM_FORMATS.includes(parsed.format) ? parsed.format : DEFAULT_STREAM_PREFS.format;
    return { format, height };
  } catch {
    return { ...DEFAULT_STREAM_PREFS };
  }
}

export function saveStreamPrefs(cameraId, prefs) {
  if (!cameraId || !prefs) return;
  localStorage.setItem(storageKey(cameraId), JSON.stringify(prefs));
}

export function heightLabel(h) {
  return `${h}p`;
}
