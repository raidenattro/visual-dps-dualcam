/** 摄像头流类型与表单字段（与后端 source_type 对齐） */

export const DEFAULT_SOURCE_TYPE = 'rtsp_pull';

export const CAMERA_SOURCE_TYPES = [
  {
    value: 'rtsp_pull',
    label: '拉取外部流',
    hint: '从外部 RTSP 拉流。上游地址必填，本机播放地址可留空自动生成。',
  },
  {
    value: 'publisher',
    label: '外部推流',
    hint: '向本机通道推流。本机播放地址可留空自动生成。',
  },
  {
    value: 'external',
    label: '直连 RTSP',
    hint: '直接填写 RTSP 地址，不经 MediaMTX。',
  },
];

/** 已废弃类型（仅展示旧数据） */
const LEGACY_SOURCE_LABELS = {
  v4l2: '本地摄像头（已废弃）',
};

export const DEFAULT_RTSP_HOST = '127.0.0.1';
export const DEFAULT_RTSP_PORT = '8554';

const LOCAL_MTX_HOSTS = new Set(['127.0.0.1', 'localhost', 'mediamtx']);

export function defaultPlaybackUrl(path, host = DEFAULT_RTSP_HOST, port = DEFAULT_RTSP_PORT) {
  const slug = String(path || '').trim().replace(/^\/+/, '');
  if (!slug) return '';
  return `rtsp://${host}:${port}/${slug}`;
}

/** 本机播放地址是否仍是按通道号自动生成的（空也算），改通道号时跟着刷新 */
export function isAutoPlaybackUrl(url, path) {
  const play = defaultPlaybackUrl(path);
  const cur = String(url || '').trim();
  return !cur || cur === play;
}

/** 通道号变了且播放地址未手改时，写成新的本机地址（写入 value，不用灰色 placeholder） */
export function syncAutoPlaybackUrl(side, prevPath) {
  const type = side?.source_type || DEFAULT_SOURCE_TYPE;
  if (type === 'external') return side;
  if (!isAutoPlaybackUrl(side.url, prevPath)) return side;
  return { ...side, url: defaultPlaybackUrl(side.path) };
}

function parseRtspPath(url) {
  try {
    const u = new URL(url);
    const parts = u.pathname.split('/').filter(Boolean);
    return parts[parts.length - 1] || '';
  } catch {
    return '';
  }
}

/** 播放地址是否指向本机 MediaMTX 的该通道 */
export function isLocalMediamtxPlaybackUrl(url, path) {
  const slug = String(path || '').trim();
  if (!slug || !url) return false;
  try {
    const u = new URL(url);
    if (!['rtsp:', 'rtsps:'].includes(u.protocol)) return false;
    const host = (u.hostname || '').toLowerCase();
    if (!LOCAL_MTX_HOSTS.has(host)) return false;
    return parseRtspPath(url) === slug;
  } catch {
    return false;
  }
}

/** 加载表单时规范化流类型（v4l2 迁移为外部推流） */
export function normalizeSourceTypeForForm(cam) {
  const path = cam?.path || cam?.id || '';
  const raw = cam?.source_type || DEFAULT_SOURCE_TYPE;
  let source_type = raw;
  let url = cam?.url || '';
  let pull_url = cam?.pull_url || '';

  if (raw === 'v4l2') {
    source_type = 'publisher';
    if (!url || !isLocalMediamtxPlaybackUrl(url, path)) {
      url = defaultPlaybackUrl(path) || url;
    }
    pull_url = '';
  }

  return { source_type, url, pull_url };
}

export function emptyCameraForm() {
  return {
    path: '',
    name: '',
    source_type: DEFAULT_SOURCE_TYPE,
    url: '',
    pull_url: '',
    enabled: true,
    settings: {},
  };
}

/** 从 API 摄像头记录填充抽屉表单（勿把 pull_url 填入 url） */
export function cameraToForm(cam) {
  if (!cam) return emptyCameraForm();
  const { source_type, url, pull_url } = normalizeSourceTypeForForm(cam);
  const path = cam.path || cam.id || '';
  return {
    path,
    name: cam.name || '',
    source_type,
    url: url || (source_type === 'external' ? '' : defaultPlaybackUrl(path)),
    pull_url,
    enabled: cam.enabled !== false,
    settings: { ...(cam.settings || {}) },
  };
}

/** 保存前组装 payload */
export function formToCameraPayload(form) {
  const path = String(form.path || '').trim();
  const name = String(form.name || '').trim();
  const source_type = form.source_type || DEFAULT_SOURCE_TYPE;
  const payload = {
    path,
    name,
    source_type,
    enabled: form.enabled !== false,
    settings: form.settings || {},
  };

  if (source_type === 'rtsp_pull') {
    payload.pull_url = String(form.pull_url || '').trim();
    payload.url = String(form.url || '').trim() || defaultPlaybackUrl(path);
  } else if (source_type === 'external') {
    payload.url = String(form.url || '').trim();
    payload.pull_url = '';
  } else {
    payload.url = String(form.url || '').trim() || defaultPlaybackUrl(path);
    payload.pull_url = '';
  }
  return payload;
}

export function sourceTypeLabel(value) {
  return (
    CAMERA_SOURCE_TYPES.find((t) => t.value === value)?.label
    || LEGACY_SOURCE_LABELS[value]
    || value
    || '—'
  );
}

/** 切换流类型时一并改相关字段。巷道左右路若连续 onChange 会互相覆盖，必须一次写入。 */
export function sourceTypeFormPatch(form, nextType) {
  const source_type = nextType || DEFAULT_SOURCE_TYPE;
  const patch = { source_type };
  if (source_type === 'publisher' || source_type === 'rtsp_pull') {
    const play = defaultPlaybackUrl(form?.path);
    if (play && !form?.url) patch.url = play;
  }
  if (source_type === 'external') patch.pull_url = '';
  return patch;
}

/** 表单字段更新：支持 (field, value) 或一次写入的 patch 对象 */
export function applyFormFields(prev, field, value) {
  if (field && typeof field === 'object' && value === undefined) {
    return { ...prev, ...field };
  }
  return { ...prev, [field]: value };
}

export function emptyAisleCreateForm() {
  return {
    aisle_id: '',
    left: emptyCameraForm(),
    right: emptyCameraForm(),
  };
}

/** 改巷道编号时，未手改过的通道号/名称跟着变成 {id}-L / {id}-R */
export function applyAisleIdToCreateForm(form, aisleId) {
  const prev = String(form.aisle_id || '').trim();
  const id = String(aisleId || '').trim();
  const autoPath = (role) => (prev ? `${prev}-${role}` : '');
  const autoName = (side) => (prev ? `${prev} ${side}` : '');
  const left = { ...form.left };
  const right = { ...form.right };
  const prevLeftPath = left.path;
  const prevRightPath = right.path;
  if (!left.path || left.path === autoPath('L')) left.path = id ? `${id}-L` : '';
  if (!right.path || right.path === autoPath('R')) right.path = id ? `${id}-R` : '';
  if (!left.name || left.name === autoName('左路')) left.name = id ? `${id} 左路` : '';
  if (!right.name || right.name === autoName('右路')) right.name = id ? `${id} 右路` : '';
  return {
    ...form,
    aisle_id: id,
    left: syncAutoPlaybackUrl(left, prevLeftPath),
    right: syncAutoPlaybackUrl(right, prevRightPath),
  };
}
