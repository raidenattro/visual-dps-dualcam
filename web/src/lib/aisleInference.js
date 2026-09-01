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

export function aisleInferOn(camL, camR) {
  const running = (c) => c?.inference?.status === 'running' || c?.inference?.status === 'starting';
  return running(camL) || running(camR);
}

/** 巷道两路合成状态：一路还在加载模型时整巷道算 starting */
export function aisleInferStatus(camL, camR) {
  const a = camL?.inference?.status || 'stopped';
  const b = camR?.inference?.status || 'stopped';
  if (a === 'error' || b === 'error') return 'error';
  if (a === 'starting' || b === 'starting') return 'starting';
  if (a === 'running' && b === 'running') return 'running';
  if (a === 'running' || b === 'running') return 'starting';
  return 'stopped';
}

export const AISLE_INFER_LABEL = {
  stopped: '巷道检测未启动',
  starting: '骨架推理启动中',
  running: '巷道检测运行中',
  error: '检测异常',
};
