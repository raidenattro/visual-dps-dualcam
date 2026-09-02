import { useCallback, useEffect, useRef, useState } from 'react';
import { apiGet, apiPost, apiPut, thumbnailUrl } from '../api/client.js';
import {
  meshCells,
  moveLayerRow,
  projectPix,
  rayPlane,
  vertIndex,
  wallById,
  wallPlane,
} from '../lib/dualcamGeom.js';
import { letterboxParams } from '../lib/previewLayout.js';
import { formatUserError } from '../lib/userFacingText.js';
import './AisleAnnotatePage.css';

const CORNERS = [
  { n: 1, title: '顶沿·远端', detail: '货架顶部、画面里更远的那个角' },
  { n: 2, title: '顶沿·近端', detail: '货架顶部、离相机更近的角' },
  { n: 3, title: '底沿·近端', detail: '货架底部、离相机更近的角' },
  { n: 4, title: '底沿·远端', detail: '货架底部、再绕回远端，围成一圈' },
];
const HINT = CORNERS.map((c) => `${c.n} ${c.title}`);
/** 右路画面角序与左路 3D 墙角相反（对齐后同一物理角）。 */
const OPP_CORNER = [1, 0, 3, 2];
const FALLBACK_W = 1280;
const FALLBACK_H = 720;
/** 抽帧按上限高度，再把 image_size 写成返回的真实宽高（不沿用旧标定高度）。 */
const GRAB_HEIGHT = 720;
/** 层线命中 / 开始拖：屏幕像素。未超过拖动阈值的点击不得改线。 */
const LAYER_HIT_PX = 10;
const CORNER_HIT_PX = 16;
const DRAG_START_PX = 8;

function distToSeg(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  if (len2 < 1e-9) return Math.hypot(px - ax, py - ay);
  const t = Math.min(1, Math.max(0, ((px - ax) * dx + (py - ay) * dy) / len2));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/** 墙 1/2 固定配色：左右路同一面墙同色，避免现场把两路画面当成两面墙。 */
const WALL_PALETTE = {
  1: { line: '#ff9f1c', dim: '#b87412', mesh: '#ffc45a', ink: '#1a1204', name: '橙' },
  2: { line: '#3ec8ff', dim: '#1a88b5', mesh: '#7edcff', ink: '#041218', name: '青' },
};

function wallPalette(wallId) {
  return WALL_PALETTE[Number(wallId)] || WALL_PALETTE[1];
}

/** 空心圈：把反解出的 3D 矩形墙角投回该路画面，仅作残差提示，不参与编辑。 */
function drawSolvedCornerHints(c, role, layout, aisle) {
  const sol = aisle?.solved;
  if (!sol?.ok) return;
  const cam = sol.cameras?.[role];
  if (!cam?.C || !cam?.fwd) return;
  const order = role === 'R' ? OPP_CORNER : [0, 1, 2, 3];
  for (const w of sol.walls || []) {
    const pal = wallPalette(w.wall_id);
    order.forEach((ci, i) => {
      const p = (w.corners || [])[ci];
      if (!p) return;
      const xy = mapCalibToCanvas(projectPix(p, cam), layout);
      if (!xy) return;
      c.beginPath();
      c.arc(xy[0], xy[1], 11, 0, Math.PI * 2);
      c.strokeStyle = pal.line;
      c.lineWidth = 1.8;
      c.stroke();
      c.fillStyle = pal.line;
      c.font = '10px sans-serif';
      c.textAlign = 'center';
      c.textBaseline = 'bottom';
      c.fillText(String(i + 1), xy[0], xy[1] - 13);
    });
  }
}

function viewSize(state, v) {
  const s = state?.views?.[v]?.image_size;
  if (Array.isArray(s) && s.length >= 2) {
    const w = Number(s[0]);
    const h = Number(s[1]);
    if (w > 1 && h > 1) return [w, h];
  }
  return [FALLBACK_W, FALLBACK_H];
}

function emptyWalls() {
  return [
    { wall_id: 1, width: 2.2, height: 2.0, base: 0, quad: [], shelf_code: '', n_layers: 4, n_cols: 4 },
    { wall_id: 2, width: 2.2, height: 2.0, base: 0, quad: [], shelf_code: '', n_layers: 4, n_cols: 4 },
  ];
}

/** 本墙货格行列：每面墙自己的数字，缺省时回退到已生成 mesh。 */
function wallGrid(wall, mesh) {
  const layers = Number(wall?.n_layers) || Number(mesh?.n_layers || mesh?.rows) || 4;
  const cols = Number(wall?.n_cols) || Number(mesh?.cols) || 4;
  return {
    n_layers: Math.min(8, Math.max(1, layers)),
    n_cols: Math.min(8, Math.max(1, cols)),
  };
}

function meshForWall(aisle, wallId) {
  return (aisle?.slot_meshes || []).find((m) => Number(m.wall_id) === Number(wallId));
}

/** 该路画面已标完四角的墙，才画对应货格。 */
function viewHasWallQuad(aisle, role, wallId) {
  return Boolean(viewWallQuad(aisle, role, wallId));
}

function viewWallQuad(aisle, role, wallId) {
  const wall = (aisle?.views?.[role]?.walls || []).find((w) => Number(w.wall_id) === Number(wallId));
  const quad = wall?.quad || [];
  return quad.length >= 4 ? quad : null;
}

function camOf(aisle, role) {
  return aisle?.solved?.ok ? aisle.solved?.cameras?.[role] || null : null;
}

/**
 * 底图 object-fit:contain：叠层跟照片内容框对齐（均匀缩放 + 黑边），禁止 kx≠ky 铺满 pane。
 * 标定像素先铺到底图自然尺寸（同图只是均匀缩放），再 contain 进 CSS 盒子。
 */
function overlayLayout(canvas, IW, IH) {
  const rect = canvas.getBoundingClientRect();
  const dw = Math.max(2, Math.floor(rect.width));
  const dh = Math.max(2, Math.floor(rect.height));
  const img = canvas.parentElement?.querySelector('img.bg');
  const nw = Number(img?.naturalWidth) || 0;
  const nh = Number(img?.naturalHeight) || 0;
  const srcW = nw > 1 ? nw : IW;
  const srcH = nh > 1 ? nh : IH;
  const box = letterboxParams(srcW, srcH, dw, dh);
  return {
    dw,
    dh,
    srcW,
    srcH,
    sx: srcW / Math.max(IW, 1e-6),
    sy: srcH / Math.max(IH, 1e-6),
    ...box,
  };
}

function mapCalibToCanvas(uv, layout) {
  if (!uv || uv.length < 2 || !layout) return null;
  return [
    uv[0] * layout.sx * layout.scale + layout.padX,
    uv[1] * layout.sy * layout.scale + layout.padY,
  ];
}

function unmapCanvasToCalib(mx, my, layout) {
  const s = Math.max(layout.scale, 1e-9);
  return [
    (mx - layout.padX) / s / layout.sx,
    (my - layout.padY) / s / layout.sy,
  ];
}

/** 墙名标签：只在 2D 四角附近摆字，层线不用这个。 */
function lerpQuad(quad, ty, tz) {
  if (!quad || quad.length < 4) return null;
  const [c0, c1, c2, c3] = quad;
  return [
    (1 - ty) * (1 - tz) * c0[0] + (1 - ty) * tz * c1[0] + ty * tz * c2[0] + ty * (1 - tz) * c3[0],
    (1 - ty) * (1 - tz) * c0[1] + (1 - ty) * tz * c1[1] + ty * tz * c2[1] + ty * (1 - tz) * c3[1],
  ];
}

function pointInPoly(u, v, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i, i += 1) {
    const xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
    if ((yi > v) !== (yj > v) && u < ((xj - xi) * (v - yi)) / ((yj - yi) || 1e-9) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

function shelfCodeOf(aisle, wallId) {
  const wall = (aisle?.views?.L?.walls || []).find((w) => Number(w.wall_id) === Number(wallId));
  return String(wall?.shelf_code || meshForWall(aisle, wallId)?.shelf_code || '').trim();
}

/** 左右路是否已有静止帧（本地 frameAt 或摄像头 last_frame_at）。 */
function hasGrabbedFrames(camL, camR, frameAt, cameras) {
  if (!camL || !camR) return false;
  if (frameAt[camL] && frameAt[camR]) return true;
  const lCam = cameras.find((c) => c.id === camL);
  const rCam = cameras.find((c) => c.id === camR);
  return Boolean(lCam?.last_frame_at && rCam?.last_frame_at);
}

function annotateSteps(aisle, { grouped, frameAt, camL, camR, dirty, cameras = [] }) {
  const req = requiredWallIds(aisle);
  const quadsOk = req.every((id) => viewHasWallQuad(aisle, 'L', id) && viewHasWallQuad(aisle, 'R', id));
  const meshesOk = req.every((id) => meshForWall(aisle, id));
  const shelfOk = req.every((id) => shelfCodeOf(aisle, id));
  const items = [
    { key: 'bind', label: '左右路已绑定', done: grouped },
    { key: 'grab', label: '抽帧', done: hasGrabbedFrames(camL, camR, frameAt, cameras) },
    { key: 'quad', label: '左右路都标满四角', done: quadsOk },
    { key: 'solve', label: '反解并对齐', done: Boolean(aisle.solved?.ok) },
    { key: 'layer', label: '拖层线对齐层板', done: Boolean(aisle.solved?.ok && meshesOk) },
    { key: 'commit', label: '保存墙标定', done: Boolean(aisle.solved?.ok && meshesOk && !dirty) },
    { key: 'shelf', label: '填货架号', done: shelfOk },
  ];
  return { items, next: items.find((s) => !s.done) || null };
}

function emptyAisle(id) {
  return {
    aisle_id: id,
    aisle: 2.0,
    contact_m: 0,
    prior: { camH: 2.84, camDist: 1.56, pitch: 45, yaw: 0 },
    cameras: { L: { camera_id: '', role: 'L' }, R: { camera_id: '', role: 'R' } },
    aabb: { x: [-1.35, 1.35], y: [0.5, 1.65], z: [-0.12, 2.5] },
    views: {
      L: { name: 'L', image_size: [FALLBACK_W, FALLBACK_H], prior: { camH: 2.84, camDist: 1.56, pitch: 45, yaw: 0 }, walls: emptyWalls() },
      R: { name: 'R', image_size: [FALLBACK_W, FALLBACK_H], prior: { camH: 2.84, camDist: 1.56, pitch: 45, yaw: 0 }, walls: emptyWalls() },
    },
    slot_meshes: [],
    required_wall_ids: [1],
    solved: { ok: false },
  };
}

function requiredWallIds(state) {
  const raw = state?.required_wall_ids;
  if (Array.isArray(raw) && raw.length) {
    return [...new Set(raw.map(Number).filter((n) => n >= 1))];
  }
  const fromMesh = [
    ...new Set((state?.slot_meshes || []).map((m) => Number(m.wall_id)).filter((n) => n >= 1)),
  ];
  return fromMesh.length ? fromMesh : [1];
}

export default function AisleAnnotatePage() {
  const [cameras, setCameras] = useState([]);
  const [aisleId, setAisleId] = useState('');
  const [aisleList, setAisleList] = useState([]);
  const [camL, setCamL] = useState('');
  const [camR, setCamR] = useState('');
  const [state, setState] = useState(() => emptyAisle('aisle-1'));
  const [msg, setMsg] = useState('');
  const [toast, setToast] = useState(null);
  const [dirty, setDirty] = useState(false);
  const [shard, setShard] = useState(null);
  const [activeView, setActiveView] = useState('L');
  const [activeWall, setActiveWall] = useState(0);
  const [selected, setSelected] = useState(null);
  const [boxIdEdit, setBoxIdEdit] = useState('');
  const [frameAt, setFrameAt] = useState({});
  const [grabbing, setGrabbing] = useState(false);
  const dragRef = useRef(null);
  const didDragRef = useRef(false);
  const skipClickRef = useRef(false);
  const stateRef = useRef(state);
  stateRef.current = state;
  const editGen = useRef(0);
  const loadedIdRef = useRef(aisleId);
  const toastTimer = useRef(null);
  const cvs = { L: useRef(null), R: useRef(null) };

  const grouped = Boolean(camL && camR && camL !== camR);
  const wallsL = state.views?.L?.walls || emptyWalls();
  const wallsR = state.views?.R?.walls || emptyWalls();
  const activeWallId = wallsL[activeWall]?.wall_id || 1;
  const activeMesh = meshForWall(state, activeWallId);
  const { n_layers: nLayers, n_cols: nCols } = wallGrid(wallsL[activeWall], activeMesh);
  const curQuad = state.views?.[activeView]?.walls?.[activeWall]?.quad || [];
  const steps = annotateSteps(state, { grouped, frameAt, camL, camR, dirty, cameras });
  const calibDone = Boolean(state.solved?.ok)
    && requiredWallIds(state).every((id) => meshForWall(state, id) && shelfCodeOf(state, id))
    && !dirty;
  const selectedCell = selected
    ? (state.slot_meshes || []).flatMap((mesh) => meshCells(mesh).map((cell) => ({ mesh, cell }))).find(
        ({ mesh, cell }) => `${mesh.wall_id}:${cell.slot_key || `r${cell.row}c${cell.col}`}` === selected,
      )?.cell
    : null;

  const showToast = (text, kind = 'ok') => {
    setMsg(text);
    setToast({ text, kind });
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), 4500);
  };

  const markDirty = (next) => {
    editGen.current += 1;
    if (next) {
      stateRef.current = next;
      setState(next);
    }
    setDirty(true);
  };

  const applyAisle = (aisle) => {
    setState(aisle);
    stateRef.current = aisle;
    loadedIdRef.current = aisle.aisle_id || aisleId;
    setDirty(false);
    setShard(aisle.logical_shard);
    const L = aisle.cameras?.L?.camera_id || '';
    const R = aisle.cameras?.R?.camera_id || '';
    setCamL(L);
    setCamR(R);
  };

  useEffect(() => {
    apiGet('/api/cameras?probe=0').then((d) => setCameras(d.items || [])).catch(() => {});
    apiGet('/api/aisles')
      .then(async (d) => {
        const items = (d.items || []).filter((a) => a.camera_l && a.camera_r);
        setAisleList(items);
        const first = items.find((a) => a.camera_l && a.camera_r) || items[0];
        if (!first?.aisle_id) return;
        setAisleId(first.aisle_id);
        const one = await apiGet(`/api/aisles/${encodeURIComponent(first.aisle_id)}`);
        if (one.status === 'success' && one.aisle) applyAisle(one.aisle);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const extra = {};
    for (const id of [camL, camR]) {
      if (!id) continue;
      const cam = cameras.find((c) => c.id === id);
      if (cam?.last_frame_at) extra[id] = cam.last_frame_at;
    }
    if (!Object.keys(extra).length) return;
    setFrameAt((prev) => ({ ...extra, ...prev }));
  }, [cameras, camL, camR]);

  const grabStills = async (lId = camL, rId = camR, aisleState = state) => {
    const targets = [
      ['L', lId],
      ['R', rId],
    ].filter(([, id]) => id);
    if (!targets.length) {
      setMsg('请先选择左右路摄像头再抽帧');
      return;
    }
    setGrabbing(true);
    try {
      const stamps = {};
      const sizes = {};
      const errs = [];
      await Promise.all(
        targets.map(async ([role, id]) => {
          try {
            const data = await apiPost(
              `/api/cameras/${encodeURIComponent(id)}/capture?height=${GRAB_HEIGHT}`,
              {},
            );
            if (data?.error || data?.status !== 'success') {
              errs.push(`${role === 'L' ? '左路' : '右路'}：${formatUserError(data?.error)}`);
              return;
            }
            stamps[role] = data.last_frame_at ?? Date.now() / 1000;
            const w = Number(data.width);
            const h = Number(data.height);
            if (w > 1 && h > 1) sizes[role] = { width: w, height: h };
          } catch (e) {
            errs.push(`${role === 'L' ? '左路' : '右路'}：${formatUserError(e.message)}`);
          }
        }),
      );
      if (Object.keys(stamps).length) {
        const byCam = {};
        targets.forEach(([role, id]) => {
          if (stamps[role]) byCam[id] = stamps[role];
        });
        setFrameAt((prev) => ({ ...prev, ...byCam }));
        setCameras((prev) =>
          prev.map((c) => {
            if (!byCam[c.id]) return c;
            return { ...c, last_frame_at: byCam[c.id], has_thumbnail: true };
          }),
        );
      }
      const aid = aisleState?.aisle_id || aisleId;
      if (aid && Object.keys(sizes).length) {
        const sized = await apiPost(`/api/aisles/${encodeURIComponent(aid)}/capture-sizes`, {
          ...sizes,
          views: aisleState?.views || stateRef.current?.views,
        });
        if (sized?.status === 'success' && sized.aisle) {
          applyAisle(sized.aisle);
          const lsz = sized.aisle.views?.L?.image_size;
          const rsz = sized.aisle.views?.R?.image_size;
          const dim = (s) => (Array.isArray(s) ? `${s[0]}×${s[1]}` : '');
          if (sized.size_changed) {
            showToast(
              sized.aisle.solved?.ok
                ? `已按实际画面更新标定像素（左 ${dim(lsz)} · 右 ${dim(rsz)}），反解已同步缩放。`
                : `已按实际画面更新标定像素（左 ${dim(lsz)} · 右 ${dim(rsz)}）。若已反解请再点一次反解。`,
              'ok',
            );
          } else if (!errs.length) {
            setMsg(`已抽取静止画面（左 ${dim(lsz)} · 右 ${dim(rsz)}），可在图上标墙四角`);
          }
        } else if (sized?.error) {
          errs.push(formatUserError(sized.error));
        }
      } else if (!errs.length) {
        setMsg('已抽取静止画面，可在图上标墙四角');
      }
      if (errs.length) setMsg(errs.join('\n'));
    } finally {
      setGrabbing(false);
    }
  };

  const autoGrabKeyRef = useRef('');
  useEffect(() => {
    if (!grouped || !aisleId || grabbing) return;
    const key = `${aisleId}:${camL}:${camR}`;
    if (hasGrabbedFrames(camL, camR, frameAt, cameras)) {
      autoGrabKeyRef.current = key;
      return;
    }
    if (autoGrabKeyRef.current === key) return;
    autoGrabKeyRef.current = key;
    grabStills(camL, camR, stateRef.current);
  }, [aisleId, camL, camR, grouped, grabbing, frameAt, cameras]);

  const loadAisle = async (id, { quiet = false } = {}) => {
    try {
      const d = await apiGet(`/api/aisles/${encodeURIComponent(id)}`);
      if (d.status !== 'success' || !d.aisle) {
        if (!quiet) setMsg(d.error || '巷道不存在');
        return;
      }
      applyAisle(d.aisle);
      if (!quiet) setMsg('已打开巷道标定，点「抽帧」取左右路静止画面');
    } catch (e) {
      if (!quiet) setMsg(formatUserError(e.message));
    }
  };

  const camNameOf = (id) => {
    const cam = cameras.find((c) => c.id === id);
    return cam?.name || id || '未绑定';
  };

  const persistAisle = async (next = stateRef.current) => {
    const snap = editGen.current;
    const payload = structuredClone(next);
    payload.aisle_id = aisleId;
    const d = await apiPut(`/api/aisles/${encodeURIComponent(aisleId)}`, payload);
    if (d.status !== 'success') {
      showToast(d.error || '保存失败', 'err');
      return null;
    }
    if (editGen.current !== snap) {
      setDirty(true);
      return d.aisle;
    }
    setState(d.aisle);
    stateRef.current = d.aisle;
    setDirty(false);
    return d.aisle;
  };

  const save = async (next = stateRef.current) => {
    const snap = editGen.current;
    const aisle = await persistAisle(next);
    if (aisle && editGen.current === snap) showToast('已保存');
    return aisle;
  };

  const ensureWallMesh = async (aisle, wallId) => {
    const wall = (aisle.views?.L?.walls || []).find((w) => Number(w.wall_id) === Number(wallId));
    const grid = wallGrid(wall, meshForWall(aisle, wallId));
    if (!aisle.solved?.ok) return aisle;
    if (!viewHasWallQuad(aisle, 'L', wallId) || !viewHasWallQuad(aisle, 'R', wallId)) return aisle;
    const mesh = meshForWall(aisle, wallId);
    const same = Boolean(
      mesh
      && Number(mesh.n_layers || mesh.rows) === grid.n_layers
      && Number(mesh.cols) === grid.n_cols,
    );
    if (same) return aisle;
    const d = await apiPost(`/api/aisles/${encodeURIComponent(aisle.aisle_id || aisleId)}/mesh`, {
      wall_id: wallId,
      n_layers: grid.n_layers,
      n_cols: grid.n_cols,
      contact_m: aisle.contact_m,
      shelf_code: wall?.shelf_code || '',
    });
    if (d.status !== 'success') {
      showToast(d.error || `墙${wallId} 层线写入失败`, 'err');
      return aisle;
    }
    return d.aisle;
  };

  const saveCurrentWall = async () => {
    const wall = wallsL[activeWall];
    const wallId = wall?.wall_id || 1;
    const grid = wallGrid(wall, meshForWall(stateRef.current, wallId));
    const payload = structuredClone(stateRef.current);
    payload.aisle_id = aisleId;
    const snap = editGen.current;
    const aisle = await persistAisle(payload);
    if (!aisle) return;
    if (!aisle.solved?.ok) {
      showToast(`已保存墙${wallId}。左右路四角都齐后，到右下点「反解并对齐」`, 'warn');
      return;
    }
    if (!viewHasWallQuad(aisle, 'L', wallId) || !viewHasWallQuad(aisle, 'R', wallId)) {
      showToast(`已保存墙${wallId}。请把左路和右路都标满 4 个角，再反解`, 'warn');
      return;
    }
    const next = await ensureWallMesh(aisle, wallId);
    if (editGen.current === snap) {
      setState(next);
      stateRef.current = next;
    }
    const shelf = shelfCodeOf(next, wallId);
    showToast(
      shelf
        ? `墙${wallId} 标定已保存（${grid.n_layers} 层 × ${grid.n_cols} 列）`
        : `墙${wallId} 层线已保存。请填写货架号后再保存一次`,
      shelf ? 'ok' : 'warn',
    );
  };

  const solve = async () => {
    const d = await apiPost(`/api/aisles/${encodeURIComponent(aisleId)}/solve`, stateRef.current);
    if (d.status !== 'success') {
      showToast(d.error || '反解失败。请确认四角顺序是 1顶远→2顶近→3底近→4底远', 'err');
      if (d.aisle) setState(d.aisle);
      return;
    }
    let aisle = d.aisle;
    for (const wid of requiredWallIds(aisle)) {
      aisle = await ensureWallMesh(aisle, wid);
    }
    setState(aisle);
    stateRef.current = aisle;
    setDirty(false);
    const a = aisle.solved?.per_view?.L?.resid_px ?? aisle.solved?.cameras?.L?.resid_px;
    const b = aisle.solved?.per_view?.R?.resid_px ?? aisle.solved?.cameras?.R?.resid_px;
    showToast(
      `反解成功（左 ${a ?? '?'} px · 右 ${b ?? '?'} px）。空心圈是墙角投回画面，套不进圈十几像素是正常的。`,
    );
  };

  const draw = useCallback(() => {
    for (const v of ['L', 'R']) {
      const canvas = cvs[v].current;
      if (!canvas) continue;
      const [IW, IH] = viewSize(state, v);
      const layout = overlayLayout(canvas, IW, IH);
      if (canvas.width !== layout.dw || canvas.height !== layout.dh) {
        canvas.width = layout.dw;
        canvas.height = layout.dh;
      }
      const c = canvas.getContext('2d');
      c.clearRect(0, 0, layout.dw, layout.dh);
      const walls = (state.views?.[v]?.walls || []);
      walls.forEach((wall, wi) => {
        if (!wall.quad?.length) return;
        const pal = wallPalette(wall.wall_id);
        const on = v === activeView && wi === activeWall;
        c.strokeStyle = on ? pal.line : pal.dim;
        c.lineWidth = on ? 2.6 : 1.6;
        c.beginPath();
        wall.quad.forEach((p, i) => {
          const xy = mapCalibToCanvas(p, layout);
          if (!xy) return;
          if (i === 0) c.moveTo(xy[0], xy[1]);
          else c.lineTo(xy[0], xy[1]);
        });
        if (wall.quad.length === 4) c.closePath();
        c.stroke();
        wall.quad.forEach((p, i) => {
          const xy = mapCalibToCanvas(p, layout);
          if (!xy) return;
          c.beginPath();
          c.arc(xy[0], xy[1], 8, 0, Math.PI * 2);
          c.fillStyle = on ? pal.line : pal.dim;
          c.fill();
          c.fillStyle = pal.ink;
          c.font = 'bold 11px sans-serif';
          c.textAlign = 'center';
          c.textBaseline = 'middle';
          c.fillText(String(i + 1), xy[0], xy[1]);
        });
        if (wall.quad.length >= 4) {
          const tag = mapCalibToCanvas(lerpQuad(wall.quad, 0.08, 0.12), layout);
          if (tag) {
            c.font = 'bold 12px sans-serif';
            c.textAlign = 'left';
            c.textBaseline = 'middle';
            c.fillStyle = pal.line;
            c.fillText(`墙${wall.wall_id}`, tag[0], tag[1]);
          }
        }
      });
      const cam = camOf(state, v);
      if (state.solved?.ok && cam) {
        for (const mesh of state.slot_meshes || []) {
          const pal = wallPalette(mesh.wall_id);
          const onWall = Number(mesh.wall_id) === Number(state.views?.L?.walls?.[activeWall]?.wall_id);
          const rows = Number(mesh.rows) || 0;
          const cols = Number(mesh.cols) || 0;
          if (rows < 1 || cols < 1 || !mesh.vertices) continue;
          const toUv = (r, col) => projectPix(mesh.vertices[vertIndex(rows, cols, r, col)], cam);
          const strokeUv = (pts, color, width) => {
            const mapped = pts.map((p) => mapCalibToCanvas(p, layout)).filter(Boolean);
            if (mapped.length < 2) return;
            c.beginPath();
            mapped.forEach((p, i) => {
              if (i === 0) c.moveTo(p[0], p[1]);
              else c.lineTo(p[0], p[1]);
            });
            c.strokeStyle = color;
            c.lineWidth = width;
            c.stroke();
          };
          for (let j = 0; j <= cols; j++) {
            const pts = [];
            for (let i = 0; i <= rows; i++) pts.push(toUv(i, j));
            strokeUv(pts, onWall ? pal.dim : pal.dim, onWall ? 1.2 : 0.9);
          }
          for (let i = 0; i <= rows; i++) {
            const pts = [];
            for (let j = 0; j <= cols; j++) pts.push(toUv(i, j));
            const inner = i > 0 && i < rows;
            strokeUv(
              pts,
              onWall ? (inner ? pal.mesh : pal.line) : pal.dim,
              onWall ? (inner ? 2.4 : 1.4) : 1.1,
            );
          }
          for (const cell of meshCells(mesh)) {
            const a = projectPix(cell.corners[0], cam);
            const b = projectPix(cell.corners[2], cam);
            if (!a || !b) continue;
            const pa = mapCalibToCanvas(a, layout);
            const pb = mapCalibToCanvas(b, layout);
            if (!pa || !pb) continue;
            const cw = Math.abs(pb[0] - pa[0]);
            const ch = Math.abs(pb[1] - pa[1]);
            const key = `${mesh.wall_id}:${cell.slot_key || cell.box_id}`;
            const onSel = selected === key;
            if (!onSel && !onWall) continue;
            if (!onSel && (cw < 28 || ch < 16)) continue;
            const mx = (pa[0] + pb[0]) / 2;
            const my = (pa[1] + pb[1]) / 2;
            const label = String(cell.box_id || '');
            c.font = onSel ? 'bold 11px sans-serif' : '10px sans-serif';
            c.textAlign = 'center';
            c.textBaseline = 'middle';
            const tw = c.measureText(label).width;
            if (!onSel && tw + 8 > cw) continue;
            c.fillStyle = 'rgba(8, 12, 16, 0.62)';
            c.fillRect(mx - tw / 2 - 3, my - 7, tw + 6, 14);
            c.fillStyle = onSel ? '#ffe08a' : '#f4f8fb';
            c.fillText(label, mx, my);
          }
        }
      }
      drawSolvedCornerHints(c, v, layout, state);
    }
  }, [state, activeView, activeWall, selected]);

  useEffect(() => {
    draw();
    const nodes = [cvs.L.current, cvs.R.current].filter(Boolean);
    if (!nodes.length) return undefined;
    const obs = new ResizeObserver(() => draw());
    nodes.forEach((n) => obs.observe(n));
    return () => obs.disconnect();
  }, [draw]);

  const evtToImg = (e, v, aisle = stateRef.current) => {
    const canvas = cvs[v].current;
    const r = canvas.getBoundingClientRect();
    const [IW, IH] = viewSize(aisle, v);
    const layout = overlayLayout(canvas, IW, IH);
    const mx = (e.clientX - r.left) * (layout.dw / Math.max(r.width, 1e-6));
    const my = (e.clientY - r.top) * (layout.dh / Math.max(r.height, 1e-6));
    return unmapCanvasToCalib(mx, my, layout);
  };

  /** 层线命中：与实验仓 dualcam_annot 一样，在标定像素里测 3D 投影线段。 */
  const hitLayerRow = (e, v, aisle = stateRef.current) => {
    if (!aisle.solved?.ok) return null;
    const cam = camOf(aisle, v);
    const canvas = cvs[v].current;
    if (!cam || !canvas) return null;
    const wallId = (aisle.views?.L?.walls || [])[activeWall]?.wall_id;
    const mesh = meshForWall(aisle, wallId);
    if (!mesh?.vertices) return null;
    const [IW, IH] = viewSize(aisle, v);
    const layout = overlayLayout(canvas, IW, IH);
    const [iu, iv] = evtToImg(e, v, aisle);
    const hitPx = LAYER_HIT_PX / Math.max(layout.scale * ((layout.sx + layout.sy) / 2), 1e-6);
    const rows = Number(mesh.rows) || 0;
    const cols = Number(mesh.cols) || 0;
    let best = null;
    for (let i = 1; i < rows; i++) {
      const a = projectPix(mesh.vertices[vertIndex(rows, cols, i, 0)], cam);
      const b = projectPix(mesh.vertices[vertIndex(rows, cols, i, cols)], cam);
      if (!a || !b) continue;
      const d = distToSeg(iu, iv, a[0], a[1], b[0], b[1]);
      if (d <= hitPx && (!best || d < best.dist)) {
        best = { v, row: i, wallId: Number(mesh.wall_id), dist: d };
      }
    }
    return best;
  };

  const hitWallCorner = (e, v, aisle = stateRef.current) => {
    const canvas = cvs[v].current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const [IW, IH] = viewSize(aisle, v);
    const layout = overlayLayout(canvas, IW, IH);
    const mx = (e.clientX - rect.left) * (layout.dw / Math.max(rect.width, 1e-6));
    const my = (e.clientY - rect.top) * (layout.dh / Math.max(rect.height, 1e-6));
    let best = null;
    (aisle.views?.[v]?.walls || []).forEach((w, wi) => {
      (w.quad || []).forEach((p, i) => {
        const xy = mapCalibToCanvas(p, layout);
        if (!xy) return;
        const d = Math.hypot(xy[0] - mx, xy[1] - my);
        if (d <= CORNER_HIT_PX && (!best || d < best.dist)) {
          best = { kind: 'corner', v, wi, i, dist: d };
        }
      });
    });
    return best;
  };

  const setLayerCursor = (e, v) => {
    const canvas = cvs[v].current;
    if (!canvas) return;
    const drag = dragRef.current;
    if (drag?.kind === 'corner') {
      canvas.style.cursor = 'grabbing';
      return;
    }
    if (drag) {
      canvas.style.cursor = 'ns-resize';
      return;
    }
    if (hitWallCorner(e, v)) {
      canvas.style.cursor = 'grab';
      return;
    }
    canvas.style.cursor = hitLayerRow(e, v) ? 'ns-resize' : 'crosshair';
  };

  const onCanvasClick = (e, v) => {
    if (skipClickRef.current || didDragRef.current) {
      skipClickRef.current = false;
      didDragRef.current = false;
      return;
    }
    setActiveView(v);
    const aisle = stateRef.current;
    const walls = aisle.views[v].walls;
    const w = walls[activeWall];
    if (w.quad.length < 4) {
      const [u, vv] = evtToImg(e, v, aisle);
      const next = structuredClone(aisle);
      next.views[v].walls[activeWall].quad.push([u, vv]);
      markDirty(next);
      return;
    }
    if (!aisle.solved?.ok) return;
    const [u, vv] = evtToImg(e, v, aisle);
    const cam = camOf(aisle, v);
    for (const mesh of aisle.slot_meshes || []) {
      if (Number(mesh.wall_id) !== Number(wallsL[activeWall]?.wall_id)) continue;
      if (!cam) continue;
      for (const cell of meshCells(mesh)) {
        const pts = (cell.corners || []).map((p) => projectPix(p, cam)).filter(Boolean);
        if (pts.length < 4 || !pointInPoly(u, vv, pts)) continue;
        const key = `${mesh.wall_id}:${cell.slot_key || `r${cell.row}c${cell.col}`}`;
        setSelected(key);
        setBoxIdEdit(cell.box_id);
        return;
      }
    }
  };

  const onCanvasDown = (e, v) => {
    if (e.button != null && e.button !== 0) return;
    setActiveView(v);
    const aisle = stateRef.current;
    const corner = hitWallCorner(e, v, aisle);
    const layer = hitLayerRow(e, v, aisle);
    // 角点优先：标定后外沿层线会贴在四角上，否则永远拖到层线
    if (corner && (!layer || corner.dist <= layer.dist + 2)) {
      setActiveWall(corner.wi);
      dragRef.current = {
        kind: 'corner',
        v,
        wi: corner.wi,
        i: corner.i,
        x0: e.clientX,
        y0: e.clientY,
      };
      didDragRef.current = false;
      skipClickRef.current = true;
      try {
        e.currentTarget.setPointerCapture(e.pointerId);
      } catch {
        /* 非主指针时部分浏览器会拒绝 capture */
      }
      e.preventDefault();
      return;
    }
    if (!layer) return;
    dragRef.current = {
      kind: 'layer',
      ...layer,
      x0: e.clientX,
      y0: e.clientY,
    };
    didDragRef.current = false;
    skipClickRef.current = false;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* 非主指针时部分浏览器会拒绝 capture */
    }
    e.preventDefault();
  };

  const onCanvasMove = (e, v) => {
    const drag = dragRef.current;
    if (!drag || drag.v !== v) {
      setLayerCursor(e, v);
      return;
    }
    const moved = Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0);
    if (!didDragRef.current && moved < DRAG_START_PX) return;
    const aisle = stateRef.current;
    const [u, vv] = evtToImg(e, v, aisle);
    didDragRef.current = true;
    skipClickRef.current = true;
    if (drag.kind === 'corner') {
      const next = structuredClone(aisle);
      const wall = next.views?.[v]?.walls?.[drag.wi];
      if (!wall?.quad?.[drag.i]) return;
      wall.quad[drag.i] = [u, vv];
      markDirty(next);
      const canvas = cvs[v].current;
      if (canvas) canvas.style.cursor = 'grabbing';
      return;
    }
    const wall = wallById(aisle.solved, drag.wallId);
    const mesh = meshForWall(aisle, drag.wallId);
    const cam = camOf(aisle, v);
    if (!wall || !mesh || !cam) return;
    const plane = wallPlane(wall.corners);
    const hit = rayPlane(u, vv, cam, plane.p0, plane.n);
    if (!hit) return;
    const nextMesh = moveLayerRow(mesh, wall.corners, drag.row, hit[1]);
    const next = structuredClone(aisle);
    next.slot_meshes = (aisle.slot_meshes || []).map((m) => (
      Number(m.wall_id) === Number(drag.wallId) ? nextMesh : m
    ));
    markDirty(next);
    const canvas = cvs[v].current;
    if (canvas) canvas.style.cursor = 'ns-resize';
  };

  const onCanvasUp = (e) => {
    const drag = dragRef.current;
    if (!drag) return;
    const canvas = cvs[drag.v]?.current;
    if (e && canvas?.hasPointerCapture?.(e.pointerId)) {
      canvas.releasePointerCapture(e.pointerId);
    }
    if (canvas) canvas.style.cursor = 'crosshair';
    dragRef.current = null;
    if (!didDragRef.current) return;
    if (drag.kind === 'corner') {
      const wallId = stateRef.current?.views?.L?.walls?.[drag.wi]?.wall_id;
      showToast(
        `墙${wallId ?? ''} 四角已改，空心圈仍是上次反解。请再点「反解并对齐」`,
        'warn',
      );
      return;
    }
    showToast(`墙${drag.wallId} 层线已调整，尚未保存。请点底部「保存墙${drag.wallId} 标定」`, 'warn');
  };

  const applyBoxId = () => {
    if (!selected) return;
    const [wallId, slotKey] = selected.split(':');
    const next = structuredClone(state);
    const mesh = next.slot_meshes.find((m) => String(m.wall_id) === String(wallId));
    if (!mesh) return;
    mesh.cell_ids = mesh.cell_ids || {};
    mesh.cell_ids[slotKey] = boxIdEdit.trim() || slotKey;
    markDirty(next);
    save(next);
  };

  const deleteCell = () => {
    if (!selected) return;
    const [wallId, slotKey] = selected.split(':');
    const next = structuredClone(state);
    const mesh = next.slot_meshes.find((m) => String(m.wall_id) === String(wallId));
    if (!mesh) return;
    mesh.deleted = [...new Set([...(mesh.deleted || []), slotKey])];
    setSelected(null);
    markDirty(next);
    save(next);
  };

  const setWallField = (i, key, val, persist = false) => {
    const next = structuredClone(state);
    next.views.L.walls[i][key] = val;
    next.views.R.walls[i][key] = val;
    if (key === 'shelf_code') {
      const wallId = next.views.L.walls[i].wall_id;
      next.slot_meshes = (next.slot_meshes || []).map((m) => (
        m.wall_id === wallId ? { ...m, shelf_code: val } : m
      ));
    }
    markDirty(next);
    if (persist) save(next);
  };

  const patchCurrentQuad = (quad) => {
    const next = structuredClone(state);
    const wall = next.views?.[activeView]?.walls?.[activeWall];
    if (!wall) return;
    wall.quad = quad;
    if (next.solved?.ok) next.solved = { ...next.solved, ok: false };
    setSelected(null);
    markDirty(next);
  };

  const resetQuad = () => {
    if (!curQuad.length) return;
    patchCurrentQuad([]);
    setMsg(
      `${activeView === 'L' ? '左路' : '右路'} 墙${activeWallId} 四角已清空，请按 1→2→3→4 重新点`,
    );
  };

  const undoCorner = () => {
    if (!curQuad.length) return;
    patchCurrentQuad(curQuad.slice(0, -1));
  };

  const toggleRequiredWall = (wallId, on) => {
    const cur = requiredWallIds(state);
    const nextIds = on
      ? [...new Set([...cur, wallId])]
      : cur.filter((id) => id !== wallId);
    if (!nextIds.length) {
      setMsg('至少勾选一面拣货墙。现场只有一面货架时只勾那一面即可。');
      return;
    }
    const next = { ...state, required_wall_ids: nextIds };
    markDirty(next);
    save(next);
  };

  return (
    <div className="aisle-page">
      <aside className="aisle-panel">
        <h1>巷道双路标注</h1>
        <p className="muted">选巷道后抽帧、点四角。墙面尺寸在右侧画面下方。</p>
        <ol className="aisle-steps">
          {steps.items.map((s) => (
            <li key={s.key} className={s.done ? 'done' : steps.next?.key === s.key ? 'next' : ''}>
              {s.label}
            </li>
          ))}
        </ol>
        {steps.next && (
          <p className="hint aisle-next">
            下一步：
            {steps.next.key === 'solve' && '到右下点「反解并对齐」，随后会自动画出层线'}
            {steps.next.key === 'layer' && '按住画面里的层线上下拖，对齐货架层板'}
            {steps.next.key === 'commit' && '层线已改，点底部绿色按钮保存，画面上方会提示「已保存」'}
            {steps.next.key === 'shelf' && '填写货架号，再点底部「保存墙标定」'}
            {steps.next.key === 'bind' && '请到总览「添加巷道」同时配置左右路摄像头'}
            {steps.next.key === 'grab' && '点「抽帧」取左右路静止画面'}
            {steps.next.key === 'quad' && `在${activeView === 'L' ? '左路' : '右路'}按 1顶远→2顶近→3底近→4底远 点四角`}
          </p>
        )}
        <div className="aisle-field">
          <span className="aisle-field-label">巷道</span>
          {aisleList.length ? (
            <select
              value={aisleList.some((a) => a.aisle_id === aisleId) ? aisleId : aisleList[0].aisle_id}
              onChange={(e) => {
                const id = e.target.value;
                setAisleId(id);
                if (id) loadAisle(id);
              }}
            >
              {aisleList.map((a) => (
                <option key={a.aisle_id} value={a.aisle_id}>
                  {a.aisle_id}
                </option>
              ))}
            </select>
          ) : (
            <p className="muted">暂无巷道，请先到总览添加。</p>
          )}
        </div>
        <p className="aisle-cam-line">
          左路 {camNameOf(camL)}
          {' · '}
          右路 {camNameOf(camR)}
        </p>
        <div className="btns">
          <button type="button" className="pri" onClick={() => grabStills()} disabled={grabbing || !grouped}>
            {grabbing ? '抽帧中…' : '抽帧'}
          </button>
        </div>
        {shard !== null && shard !== undefined && (
          <p className="ok">logical shard = {shard}（L/R 共用）</p>
        )}

        <section className="aisle-section">
        <p className="group-title">当前墙</p>
        <p className="muted">左右路画面里，橙色=墙1，青色=墙2；同色就是同一面货架。</p>
        <div className="aisle-wall-tabs">
          {wallsL.map((w, i) => (
            <button
              key={w.wall_id}
              type="button"
              className={`${i === activeWall ? 'on' : ''} wall-${w.wall_id}`}
              onClick={() => setActiveWall(i)}
            >
              墙{w.wall_id} · {wallPalette(w.wall_id).name} · L {wallsL[i].quad.length}/4 · R {wallsR[i].quad.length}/4
            </button>
          ))}
        </div>
        <label className="aisle-field">
          <span className="aisle-field-label">
            货架号
            <span className="aisle-field-extra">墙{activeWallId}</span>
          </span>
          <input
            value={wallsL[activeWall]?.shelf_code || ''}
            placeholder="必填"
            onChange={(e) => setWallField(activeWall, 'shelf_code', e.target.value)}
            onBlur={(e) => setWallField(activeWall, 'shelf_code', e.target.value, true)}
          />
        </label>
        <div className="aisle-field-pair">
          <div className="aisle-field">
            <span className="aisle-field-label">层数</span>
            <input
              type="number"
              min={1}
              max={8}
              value={nLayers}
              onChange={(e) => setWallField(activeWall, 'n_layers', Number(e.target.value))}
            />
          </div>
          <div className="aisle-field">
            <span className="aisle-field-label">列数</span>
            <input
              type="number"
              min={1}
              max={8}
              value={nCols}
              onChange={(e) => setWallField(activeWall, 'n_cols', Number(e.target.value))}
            />
          </div>
        </div>
        <p className="group-title">四角顺序</p>
        <p className="muted">
          面向货架绕一圈，先顶后底。正在标{activeView === 'L' ? '左路' : '右路'} · 墙{activeWallId}。
        </p>
        <ol className="aisle-corner-steps">
          {CORNERS.map((c, i) => (
            <li key={c.n} className={i === curQuad.length ? 'next' : i < curQuad.length ? 'done' : ''}>
              <span className="aisle-corner-n">{c.n}</span>
              <strong>{c.title}</strong>
            </li>
          ))}
        </ol>
        <div className="btns">
          <button type="button" onClick={undoCorner} disabled={!curQuad.length}>
            撤销上一点
          </button>
          <button type="button" onClick={resetQuad} disabled={!curQuad.length}>
            重置四角 1–4
          </button>
        </div>
        <p className="hint">
          {curQuad.length < 4
            ? `下一步：在${activeView === 'L' ? '左路' : '右路'}点「${HINT[curQuad.length]}」`
            : state.solved?.ok && activeMesh
              ? '四角已齐。按住层线上下拖，对齐货架层板；点货位可改编号'
              : '四角已齐。左右路都标满后，到右下点「反解并对齐」生成层线'}
        </p>
        {selected && (
          <div className="cell-edit">
            <p className="group-title">货位编号</p>
            <p className="muted">
              {selectedCell
                ? `第 ${selectedCell.row + 1} 层 · 第 ${selectedCell.col + 1} 列`
                : '点击画面中的货位进行编辑'}
            </p>
            <label className="aisle-field">
              <span className="aisle-field-label">编号</span>
              <input value={boxIdEdit} onChange={(e) => setBoxIdEdit(e.target.value)} />
            </label>
            <div className="btns">
              <button type="button" className="pri" onClick={applyBoxId}>
                应用编号
              </button>
              <button type="button" onClick={deleteCell}>
                删除货位
              </button>
            </div>
          </div>
        )}
        </section>
        <div className="btns aisle-save-bar">
          <button type="button" className="pri" onClick={saveCurrentWall} disabled={grabbing}>
            {dirty ? `保存墙${activeWallId} 标定（有未保存）` : `保存墙${activeWallId} 标定`}
          </button>
        </div>
        <p className="muted">会写入本墙四角、层列、货架号，以及你拖过的层线位置。另一面墙原样保留。</p>
        <p className={`msg ${dirty ? 'warn' : ''}`}>{dirty ? '有未保存的改动' : msg}</p>
      </aside>
      <div className="aisle-stage">
      {(toast || calibDone || dirty || msg) && (
        <div className={`aisle-status ${toast?.kind || (calibDone ? 'ok' : dirty ? 'warn' : '')}`}>
          {toast?.text
            || (calibDone ? '巷道标定已完成，层线已保存，可以去开推理。' : '')
            || (dirty ? `有未保存的改动，请点左下「保存墙${activeWallId} 标定」` : '')
            || msg}
        </div>
      )}
      <div className="aisle-views">
        {['L', 'R'].map((v) => (
          <div
            key={v}
            className={`pane ${activeView === v ? 'on' : ''}`}
            style={{ '--pane-ar': `${viewSize(state, v)[0]} / ${viewSize(state, v)[1]}` }}
          >
            <span className="tag" onClick={() => setActiveView(v)}>
              {v === 'L' ? '左路' : '右路'}
              {v === activeView ? ' · 正在标' : ''}
            </span>
            {(() => {
              const camId = v === 'L' ? camL : camR;
              const stamp = frameAt[camId] || cameras.find((c) => c.id === camId)?.last_frame_at || 0;
              if (!camId) {
                return <div className="bg empty">请先选择摄像头并成组</div>;
              }
              if (!stamp) {
                return (
                  <div className="bg empty">
                    {grabbing ? '正在抽取画面…' : '点「抽帧」获取静止画面'}
                  </div>
                );
              }
              return (
                <img
                  alt=""
                  src={thumbnailUrl(camId, stamp)}
                  className="bg"
                  draggable={false}
                  onLoad={() => draw()}
                />
              );
            })()}
            <canvas
              ref={cvs[v]}
              className="ov"
              onClick={(e) => onCanvasClick(e, v)}
              onPointerDown={(e) => onCanvasDown(e, v)}
              onPointerMove={(e) => onCanvasMove(e, v)}
              onPointerUp={onCanvasUp}
              onPointerCancel={onCanvasUp}
            />
          </div>
        ))}
      </div>
      <div className="aisle-dock">
        <div className="aisle-dock-block">
          <p className="group-title">拣货墙面</p>
          <p className="muted">现场只有一面货架时只勾那一面。点表格行可切换当前墙。</p>
          {wallsL.map((w) => (
            <label key={`req-${w.wall_id}`} className="aisle-check">
              <input
                type="checkbox"
                checked={requiredWallIds(state).includes(w.wall_id)}
                onChange={(e) => toggleRequiredWall(w.wall_id, e.target.checked)}
              />
              墙{w.wall_id} 参与拣货
              <span className={`aisle-wall-dot wall-${w.wall_id}`} />
            </label>
          ))}
          <table>
            <thead>
              <tr>
                <th>墙</th>
                <th>宽 (m)</th>
                <th>高 (m)</th>
                <th>底沿 (m)</th>
                <th>L</th>
                <th>R</th>
              </tr>
            </thead>
            <tbody>
              {wallsL.map((w, i) => (
                <tr key={w.wall_id} className={i === activeWall ? 'active' : ''} onClick={() => setActiveWall(i)}>
                  <td>
                    <span className={`aisle-wall-dot wall-${w.wall_id}`} />
                    墙{w.wall_id}
                  </td>
                  <td>
                    <input type="number" step="0.01" value={w.width} onChange={(e) => setWallField(i, 'width', Number(e.target.value))} />
                  </td>
                  <td>
                    <input type="number" step="0.01" value={w.height} onChange={(e) => setWallField(i, 'height', Number(e.target.value))} />
                  </td>
                  <td>
                    <input type="number" step="0.01" value={w.base} onChange={(e) => setWallField(i, 'base', Number(e.target.value))} />
                  </td>
                  <td>{wallsL[i].quad.length}/4</td>
                  <td>{wallsR[i].quad.length}/4</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="aisle-dock-block">
          <p className="group-title">巷道几何</p>
          <div className="aisle-field-pair">
            <div className="aisle-field">
              <span className="aisle-field-label">巷道净宽 (m)</span>
              <input type="number" step="0.01" value={state.aisle} onChange={(e) => setState({ ...state, aisle: Number(e.target.value) })} />
            </div>
            <div className="aisle-field">
              <span className="aisle-field-label">报警阈值 (m)</span>
              <input
                type="number"
                step="0.01"
                value={state.contact_m}
                onChange={(e) => setState({ ...state, contact_m: Number(e.target.value) })}
              />
            </div>
          </div>
          <div className="aisle-field">
            <span className="aisle-field-label">{activeView === 'L' ? '左路' : '右路'} 标定宽 × 高 (px)</span>
            <p className="muted">抽帧后自动写成静止图真实像素，勿再手填 16:9 默认值。</p>
            <div className="aisle-field-pair">
              <input
                type="number"
                min={320}
                value={viewSize(state, activeView)[0]}
                onChange={(e) => {
                  const next = { ...state, views: { ...state.views } };
                  const view = { ...(next.views[activeView] || {}) };
                  const h = viewSize(state, activeView)[1];
                  view.image_size = [Number(e.target.value), h];
                  next.views[activeView] = view;
                  setState(next);
                }}
              />
              <input
                type="number"
                min={180}
                value={viewSize(state, activeView)[1]}
                onChange={(e) => {
                  const next = { ...state, views: { ...state.views } };
                  const view = { ...(next.views[activeView] || {}) };
                  const w = viewSize(state, activeView)[0];
                  view.image_size = [w, Number(e.target.value)];
                  next.views[activeView] = view;
                  setState(next);
                }}
              />
            </div>
          </div>
          <div className="aisle-field-pair">
            <div className="aisle-field">
              <span className="aisle-field-label">{activeView === 'L' ? '左路' : '右路'} 相机高度 (m)</span>
              <input
                type="number"
                step="0.01"
                value={state.views?.[activeView]?.prior?.camH ?? state.prior?.camH ?? 2.84}
                onChange={(e) => {
                  const next = { ...state, views: { ...state.views } };
                  const view = { ...(next.views[activeView] || {}) };
                  view.prior = { ...(view.prior || state.prior || {}), camH: Number(e.target.value) };
                  next.views[activeView] = view;
                  setState(next);
                }}
              />
            </div>
            <div className="aisle-field">
              <span className="aisle-field-label">{activeView === 'L' ? '左路' : '右路'} 距巷道 (m)</span>
              <input
                type="number"
                step="0.01"
                value={state.views?.[activeView]?.prior?.camDist ?? state.prior?.camDist ?? 1.56}
                onChange={(e) => {
                  const next = { ...state, views: { ...state.views } };
                  const view = { ...(next.views[activeView] || {}) };
                  view.prior = { ...(view.prior || state.prior || {}), camDist: Number(e.target.value) };
                  next.views[activeView] = view;
                  setState(next);
                }}
              />
            </div>
          </div>
          <div className="aisle-field">
            <span className="aisle-field-label">AABB X / Y / Z (m)</span>
            <div className="aisle-field-pair">
              <input type="number" step="0.01" value={(state.aabb?.x || [-1.35, 1.35])[0]} onChange={(e) => setState({ ...state, aabb: { ...(state.aabb || {}), x: [Number(e.target.value), (state.aabb?.x || [-1.35, 1.35])[1]] } })} />
              <input type="number" step="0.01" value={(state.aabb?.x || [-1.35, 1.35])[1]} onChange={(e) => setState({ ...state, aabb: { ...(state.aabb || {}), x: [(state.aabb?.x || [-1.35, 1.35])[0], Number(e.target.value)] } })} />
            </div>
            <div className="aisle-field-pair">
              <input type="number" step="0.01" value={(state.aabb?.y || [0.5, 1.65])[0]} onChange={(e) => setState({ ...state, aabb: { ...(state.aabb || {}), y: [Number(e.target.value), (state.aabb?.y || [0.5, 1.65])[1]] } })} />
              <input type="number" step="0.01" value={(state.aabb?.y || [0.5, 1.65])[1]} onChange={(e) => setState({ ...state, aabb: { ...(state.aabb || {}), y: [(state.aabb?.y || [0.5, 1.65])[0], Number(e.target.value)] } })} />
            </div>
            <div className="aisle-field-pair">
              <input type="number" step="0.01" value={(state.aabb?.z || [-0.12, 2.5])[0]} onChange={(e) => setState({ ...state, aabb: { ...(state.aabb || {}), z: [Number(e.target.value), (state.aabb?.z || [-0.12, 2.5])[1]] } })} />
              <input type="number" step="0.01" value={(state.aabb?.z || [-0.12, 2.5])[1]} onChange={(e) => setState({ ...state, aabb: { ...(state.aabb || {}), z: [(state.aabb?.z || [-0.12, 2.5])[0], Number(e.target.value)] } })} />
            </div>
          </div>
          <p className="hint">
            {state.solved?.ok
              ? '空心圈=反解墙角投回画面（墙色）；实心点=你标的。模型按面宽×面高的竖直面拟合，残差十几像素套不进圈是正常的，不用死磕。层线画在四角内，按住分层线对齐后再保存。'
              : '左右路四角都齐后点此。成功后会画出空心圈提示反投影，并自动生成层线。'}
          </p>
          <div className="btns">
            <button type="button" className="pri" onClick={solve} disabled={!grouped}>
              反解并对齐
            </button>
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}
