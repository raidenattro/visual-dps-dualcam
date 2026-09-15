/** 巷道直播页路径（双路检测入口）。 */
export function aisleLivePath(aisleId) {
  const id = String(aisleId || '').trim();
  return id ? `/live/${encodeURIComponent(id)}` : null;
}

/** 已编入巷道的相机应进巷道直播，不再打开旧单路监控页。 */
export function cameraMonitorPath({ aisleId, cameraId } = {}) {
  return aisleLivePath(aisleId) || (cameraId
    ? `/monitor?camera=${encodeURIComponent(String(cameraId))}`
    : null);
}

/** 单路 Legacy 2D 货框标注（未成组 camera）。 */
export function legacyCameraAnnotatePath(cameraId) {
  const id = String(cameraId || '').trim();
  return id ? `/legacy/annotate?camera=${encodeURIComponent(id)}` : '/legacy/annotate';
}
