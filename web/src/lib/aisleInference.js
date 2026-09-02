import { apiPost } from '../api/client.js';

/** 同一巷道左右路一起开/停推理。一路失败也会把已返回的结果带上。 */
export async function startAisleInference(camL, camR) {
  const ids = [camL, camR].filter(Boolean);
  if (ids.length < 2) {
    return { ok: false, error: '巷道未绑定左右路两台摄像头' };
  }
  const results = [];
  for (const id of ids) {
    const data = await apiPost(`/api/cameras/${encodeURIComponent(id)}/inference/start`, {});
    results.push({ id, data });
    if (data.error) {
      return { ok: false, error: data.error, results };
    }
  }
  return { ok: true, results };
}

export async function stopAisleInference(camL, camR) {
  const ids = [camL, camR].filter(Boolean);
  if (!ids.length) {
    return { ok: false, error: '巷道未绑定摄像头' };
  }
  const results = [];
  let error = '';
  for (const id of ids) {
    const data = await apiPost(`/api/cameras/${encodeURIComponent(id)}/inference/stop`, {});
    results.push({ id, data });
    if (data.error && !error) error = data.error;
  }
  return { ok: !error, error, results };
}

function inferStatusOf(cam) {
  return cam?.inference?.status || 'stopped';
}

function isManualStop(cam) {
  const msg = String(cam?.inference?.message || '');
  return inferStatusOf(cam) === 'stopped' || msg === '已手动停止' || msg === '已停止';
}

export function aisleInferOn(camL, camR) {
  const running = (c) => {
    const st = inferStatusOf(c);
    return st === 'running' || st === 'starting';
  };
  return running(camL) || running(camR);
}

/** 巷道两路合成状态。关容器时一路先停不当 starting，拆除中的 error 不当异常。 */
export function aisleInferStatus(camL, camR) {
  const rawA = inferStatusOf(camL);
  const rawB = inferStatusOf(camR);
  const a = rawA === 'error' && isManualStop(camL) ? 'stopped' : rawA;
  const b = rawB === 'error' && isManualStop(camR) ? 'stopped' : rawB;
  if (a === 'error' || b === 'error') return 'error';
  if (a === 'starting' || b === 'starting') return 'starting';
  if (a === 'running' || b === 'running') return 'running';
  return 'stopped';
}

export const AISLE_INFER_LABEL = {
  stopped: '巷道检测未启动',
  starting: '骨架推理启动中',
  running: '巷道检测运行中',
  error: '检测异常',
};
