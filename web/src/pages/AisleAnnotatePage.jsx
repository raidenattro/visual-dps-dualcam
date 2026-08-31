import { useCallback, useEffect, useRef, useState } from 'react';
import { apiGet, apiPost, apiPut, cameraStreamUrl } from '../api/client.js';
import {
  meshCells,
  moveLayerRow,
  projectPix,
  vertIndex,
  wallById,
} from '../lib/dualcamGeom.js';
import './AisleAnnotatePage.css';

const HINT = ['① 顶沿·远端', '② 顶沿·近端', '③ 底沿·近端', '④ 底沿·远端'];
const IW = 1280;
const IH = 720;

function emptyWalls() {
  return [
    { wall_id: 1, width: 2.2, height: 2.0, base: 0, quad: [] },
    { wall_id: 2, width: 2.2, height: 2.0, base: 0, quad: [] },
  ];
}

function emptyAisle(id) {
  return {
    aisle_id: id,
    aisle: 2.0,
    contact_m: 0,
    prior: { camH: 2.84, camDist: 1.56, pitch: 45, yaw: 0 },
    cameras: { L: { camera_id: '', role: 'L' }, R: { camera_id: '', role: 'R' } },
    views: {
      L: { name: 'L', image_size: [IW, IH], walls: emptyWalls() },
      R: { name: 'R', image_size: [IW, IH], walls: emptyWalls() },
    },
    slot_meshes: [],
    solved: { ok: false },
  };
}

export default function AisleAnnotatePage() {
  const [cameras, setCameras] = useState([]);
  const [aisleId, setAisleId] = useState('aisle-1');
  const [camL, setCamL] = useState('');
  const [camR, setCamR] = useState('');
  const [sameGroup, setSameGroup] = useState(true);
  const [state, setState] = useState(() => emptyAisle('aisle-1'));
  const [msg, setMsg] = useState('');
  const [shard, setShard] = useState(null);
  const [activeView, setActiveView] = useState('L');
  const [activeWall, setActiveWall] = useState(0);
  const [nLayers, setNLayers] = useState(4);
  const [nCols, setNCols] = useState(4);
  const [selected, setSelected] = useState(null);
  const [boxIdEdit, setBoxIdEdit] = useState('');
  const dragRef = useRef(null);
  const cvs = { L: useRef(null), R: useRef(null) };

  const grouped = sameGroup && camL && camR && camL !== camR;
  const wallsL = state.views?.L?.walls || emptyWalls();
  const wallsR = state.views?.R?.walls || emptyWalls();

  useEffect(() => {
    apiGet('/api/cameras?probe=0').then((d) => setCameras(d.items || [])).catch(() => {});
  }, []);

  const loadAisle = useCallback(async (id) => {
    const d = await apiGet(`/api/aisles/${encodeURIComponent(id)}`);
    if (d.status === 'success' && d.aisle) {
      setState(d.aisle);
      setShard(d.aisle.logical_shard);
      const L = d.aisle.cameras?.L?.camera_id || '';
      const R = d.aisle.cameras?.R?.camera_id || '';
      if (L) setCamL(L);
      if (R) setCamR(R);
      setSameGroup(Boolean(L && R));
    }
  }, []);

  const bind = async () => {
    if (!sameGroup) {
      setMsg('必须勾选同一组才能标定和开推理');
      return;
    }
    const d = await apiPut(`/api/aisles/${encodeURIComponent(aisleId)}/group`, {
      camera_l: camL,
      camera_r: camR,
    });
    if (d.status !== 'success') {
      setMsg(d.error || '成组失败');
      return;
    }
    setState(d.aisle);
    setShard(d.aisle.logical_shard);
    setMsg(`已成组，logical shard=${d.aisle.logical_shard}（左右路同一 worker）`);
  };

  const save = async (next = state) => {
    const d = await apiPut(`/api/aisles/${encodeURIComponent(aisleId)}`, next);
    if (d.status === 'success') {
      setState(d.aisle);
      setMsg('已保存');
    } else setMsg(d.error || '保存失败');
  };

  const solve = async () => {
    const d = await apiPost(`/api/aisles/${encodeURIComponent(aisleId)}/solve`, state);
    if (d.status !== 'success') {
      setMsg(d.error || '反解失败');
      if (d.aisle) setState(d.aisle);
      return;
    }
    setState(d.aisle);
    setMsg(`反解成功 对齐残差 ${d.aisle.solved?.align_rms_m ?? '?'} m`);
  };

  const genMesh = async () => {
    const wallId = wallsL[activeWall]?.wall_id || 1;
    const d = await apiPost(`/api/aisles/${encodeURIComponent(aisleId)}/mesh`, {
      wall_id: wallId,
      n_layers: nLayers,
      n_cols: nCols,
      contact_m: state.contact_m,
    });
    if (d.status !== 'success') {
      setMsg(d.error || '生成层线失败');
      return;
    }
    setState(d.aisle);
    setMsg('已生成本墙层线，拖水平分格线对齐隔板');
  };

  const draw = useCallback(() => {
    for (const v of ['L', 'R']) {
      const canvas = cvs[v].current;
      if (!canvas) continue;
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(2, Math.floor(rect.width));
      const h = Math.max(2, Math.floor((w * IH) / IW));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      const c = canvas.getContext('2d');
      c.clearRect(0, 0, w, h);
      const k = w / IW;
      const walls = (state.views?.[v]?.walls || []);
      walls.forEach((wall, wi) => {
        if (!wall.quad?.length) return;
        const on = v === activeView && wi === activeWall;
        c.strokeStyle = on ? '#5cf08a' : '#4a8a5f';
        c.lineWidth = on ? 2.2 : 1.4;
        c.beginPath();
        wall.quad.forEach((p, i) => {
          if (i === 0) c.moveTo(p[0] * k, p[1] * k);
          else c.lineTo(p[0] * k, p[1] * k);
        });
        if (wall.quad.length === 4) c.closePath();
        c.stroke();
        wall.quad.forEach((p, i) => {
          c.beginPath();
          c.arc(p[0] * k, p[1] * k, 7, 0, Math.PI * 2);
          c.fillStyle = on ? '#5cf08a' : '#4a8a5f';
          c.fill();
          c.fillStyle = '#111';
          c.font = '11px sans-serif';
          c.textAlign = 'center';
          c.textBaseline = 'middle';
          c.fillText(String(i + 1), p[0] * k, p[1] * k);
        });
      });
      const cam = state.solved?.ok ? state.solved.cameras?.[v] : null;
      if (cam) {
        for (const mesh of state.slot_meshes || []) {
          const rows = mesh.rows;
          const cols = mesh.cols;
          for (let i = 0; i <= rows; i++) {
            c.beginPath();
            let started = false;
            for (let j = 0; j <= cols; j++) {
              const uv = projectPix(mesh.vertices[vertIndex(rows, cols, i, j)], cam);
              if (!uv) continue;
              if (!started) {
                c.moveTo(uv[0] * k, uv[1] * k);
                started = true;
              } else c.lineTo(uv[0] * k, uv[1] * k);
            }
            c.strokeStyle = i > 0 && i < rows ? '#7ec8ff' : '#5cf08a';
            c.lineWidth = i > 0 && i < rows ? 2.4 : 1.2;
            c.stroke();
          }
          for (const cell of meshCells(mesh)) {
            const a = projectPix(cell.corners[0], cam);
            const b = projectPix(cell.corners[2], cam);
            if (!a || !b) continue;
            const key = `${mesh.wall_id}:${cell.slot_key || cell.box_id}`;
            c.fillStyle = selected === key ? '#ffe08a' : '#7ec8ff';
            c.font = '10px sans-serif';
            c.textAlign = 'center';
            c.textBaseline = 'middle';
            c.fillText(cell.box_id, ((a[0] + b[0]) / 2) * k, ((a[1] + b[1]) / 2) * k);
          }
        }
      }
    }
  }, [state, activeView, activeWall, selected]);

  useEffect(() => {
    draw();
  }, [draw]);

  const evtToImg = (e, v) => {
    const canvas = cvs[v].current;
    const r = canvas.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * IW, ((e.clientY - r.top) / r.height) * IH];
  };

  const onCanvasClick = (e, v) => {
    setActiveView(v);
    const walls = state.views[v].walls;
    const w = walls[activeWall];
    if (w.quad.length < 4) {
      const [u, vv] = evtToImg(e, v);
      const next = structuredClone(state);
      next.views[v].walls[activeWall].quad.push([u, vv]);
      setState(next);
      return;
    }
    if (!state.solved?.ok) return;
    const cam = state.solved.cameras[v];
    const [u, vv] = evtToImg(e, v);
    for (const mesh of state.slot_meshes || []) {
      for (const cell of meshCells(mesh)) {
        const pts = cell.corners.map((p) => projectPix(p, cam)).filter(Boolean);
        if (pts.length < 4) continue;
        const xs = pts.map((p) => p[0]);
        const ys = pts.map((p) => p[1]);
        if (u >= Math.min(...xs) && u <= Math.max(...xs) && vv >= Math.min(...ys) && vv <= Math.max(...ys)) {
          const key = `${mesh.wall_id}:${cell.slot_key || `r${cell.row}c${cell.col}`}`;
          setSelected(key);
          setBoxIdEdit(cell.box_id);
          return;
        }
      }
    }
  };

  const onCanvasDown = (e, v) => {
    if (!state.solved?.ok) return;
    const cam = state.solved.cameras[v];
    const [u, vv] = evtToImg(e, v);
    const wall = wallById(state.solved, wallsL[activeWall]?.wall_id);
    const mesh = (state.slot_meshes || []).find((m) => m.wall_id === wall?.wall_id);
    if (!mesh || !cam) return;
    for (let i = 1; i < mesh.rows; i++) {
      const a = projectPix(mesh.vertices[vertIndex(mesh.rows, mesh.cols, i, 0)], cam);
      const b = projectPix(mesh.vertices[vertIndex(mesh.rows, mesh.cols, i, mesh.cols)], cam);
      if (!a || !b) continue;
      const midY = (a[1] + b[1]) / 2;
      if (Math.abs(vv - midY) < 18) {
        dragRef.current = { v, row: i, wallId: mesh.wall_id };
        return;
      }
    }
  };

  const onCanvasMove = (e, v) => {
    const drag = dragRef.current;
    if (!drag || drag.v !== v) return;
    const cam = state.solved.cameras[v];
    const wall = wallById(state.solved, drag.wallId);
    const mesh = (state.slot_meshes || []).find((m) => m.wall_id === drag.wallId);
    if (!cam || !wall || !mesh) return;
    const [, vv] = evtToImg(e, v);
    const yTop = Math.max(...wall.corners.map((p) => p[1]));
    const yBot = Math.min(...wall.corners.map((p) => p[1]));
    const t = Math.min(1, Math.max(0, (e.nativeEvent.offsetY / cvs[v].current.height)));
    const y = yTop - t * (yTop - yBot);
    const nextMesh = moveLayerRow(mesh, wall.corners, drag.row, y);
    const next = structuredClone(state);
    next.slot_meshes = (state.slot_meshes || []).map((m) => (m.wall_id === drag.wallId ? nextMesh : m));
    setState(next);
  };

  const onCanvasUp = () => {
    if (dragRef.current) {
      dragRef.current = null;
      save(state);
    }
  };

  const applyBoxId = () => {
    if (!selected) return;
    const [wallId, slotKey] = selected.split(':');
    const next = structuredClone(state);
    const mesh = next.slot_meshes.find((m) => String(m.wall_id) === String(wallId));
    if (!mesh) return;
    mesh.cell_ids = mesh.cell_ids || {};
    mesh.cell_ids[slotKey] = boxIdEdit.trim() || slotKey;
    setState(next);
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
    setState(next);
    save(next);
  };

  const setWallField = (i, key, val) => {
    const next = structuredClone(state);
    next.views.L.walls[i][key] = val;
    next.views.R.walls[i][key] = val;
    setState(next);
  };

  const camOptions = cameras.map((c) => (
    <option key={c.id} value={c.id}>
      {c.name || c.id}
    </option>
  ));

  return (
    <div className="aisle-page">
      <aside className="aisle-panel">
        <h1>巷道双路标注</h1>
        <p className="muted">先勾选同一组并指定左右路，才能反解、切格和开推理。成组后两路分到同一 worker。</p>
        <label>
          巷道编号
          <input value={aisleId} onChange={(e) => setAisleId(e.target.value)} />
        </label>
        <label className="check">
          <input type="checkbox" checked={sameGroup} onChange={(e) => setSameGroup(e.target.checked)} />
          两台摄像头为同一组
        </label>
        <label>
          左路
          <select value={camL} onChange={(e) => setCamL(e.target.value)} disabled={!sameGroup}>
            <option value="">选择摄像头</option>
            {camOptions}
          </select>
        </label>
        <label>
          右路
          <select value={camR} onChange={(e) => setCamR(e.target.value)} disabled={!sameGroup}>
            <option value="">选择摄像头</option>
            {camOptions}
          </select>
        </label>
        <div className="btns">
          <button type="button" className="pri" onClick={bind} disabled={!grouped}>
            保存同一组
          </button>
          <button type="button" onClick={() => loadAisle(aisleId)}>
            加载
          </button>
        </div>
        {shard !== null && shard !== undefined && (
          <p className="ok">logical shard = {shard}（L/R 共用）</p>
        )}

        <hr />
        <table>
          <thead>
            <tr>
              <th>墙</th>
              <th>宽</th>
              <th>高</th>
              <th>底沿</th>
              <th>L</th>
              <th>R</th>
            </tr>
          </thead>
          <tbody>
            {wallsL.map((w, i) => (
              <tr key={w.wall_id} className={i === activeWall ? 'active' : ''} onClick={() => setActiveWall(i)}>
                <td>墙{w.wall_id}</td>
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
        <p className="hint">
          {activeView === 'L' ? '左路' : '右路'} 墙{wallsL[activeWall]?.wall_id}：
          {(state.views?.[activeView]?.walls?.[activeWall]?.quad || []).length < 4
            ? `点 ${HINT[(state.views?.[activeView]?.walls?.[activeWall]?.quad || []).length]}`
            : '点「反解并对齐」后生成层线'}
        </p>
        <label>
          巷道净宽 m
          <input type="number" step="0.01" value={state.aisle} onChange={(e) => setState({ ...state, aisle: Number(e.target.value) })} />
        </label>
        <label>
          报警阈值 m
          <input
            type="number"
            step="0.01"
            value={state.contact_m}
            onChange={(e) => setState({ ...state, contact_m: Number(e.target.value) })}
          />
        </label>
        <div className="btns">
          <button type="button" className="pri" onClick={solve} disabled={!grouped}>
            1. 反解并对齐
          </button>
          <button type="button" onClick={() => save()}>
            保存标定
          </button>
        </div>
        <label>
          层数
          <input type="number" min={1} max={8} value={nLayers} onChange={(e) => setNLayers(Number(e.target.value))} />
        </label>
        <label>
          列数
          <input type="number" min={1} max={8} value={nCols} onChange={(e) => setNCols(Number(e.target.value))} />
        </label>
        <div className="btns">
          <button type="button" className="pri" onClick={genMesh} disabled={!state.solved?.ok}>
            2. 生成本墙层线
          </button>
        </div>
        {selected && (
          <div className="cell-edit">
            <div className="group-title">选中货格</div>
            <label>
              box_id
              <input value={boxIdEdit} onChange={(e) => setBoxIdEdit(e.target.value)} />
            </label>
            <div className="btns">
              <button type="button" className="pri" onClick={applyBoxId}>
                应用编号
              </button>
              <button type="button" onClick={deleteCell}>
                删除此格
              </button>
            </div>
          </div>
        )}
        <p className="msg">{msg}</p>
      </aside>
      <div className="aisle-views">
        {['L', 'R'].map((v) => (
          <div key={v} className={`pane ${activeView === v ? 'on' : ''}`}>
            <span className="tag">{v === 'L' ? '左路' : '右路'}</span>
            {(v === 'L' ? camL : camR) ? (
              <img alt="" src={cameraStreamUrl(v === 'L' ? camL : camR)} className="bg" />
            ) : (
              <div className="bg empty">请先选择摄像头并成组</div>
            )}
            <canvas
              ref={cvs[v]}
              className="ov"
              onClick={(e) => onCanvasClick(e, v)}
              onMouseDown={(e) => onCanvasDown(e, v)}
              onMouseMove={(e) => onCanvasMove(e, v)}
              onMouseUp={onCanvasUp}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
