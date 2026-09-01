import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import AisleScene3D from '../components/AisleScene3D.jsx';
import { apiGet, cameraStreamUrl, openCameraLiveStream, thumbnailUrl } from '../api/client.js';
import {
  AISLE_INFER_LABEL,
  aisleInferOn,
  aisleInferStatus,
  startAisleInference,
  stopAisleInference,
} from '../lib/aisleInference.js';
import { COCO_LINES, SKELETON_CONF, scaleInferPoint } from '../lib/cocoSkeleton.js';
import { formatInferenceMessage, formatUserError } from '../lib/userFacingText.js';
import './AisleLivePage.css';

function inferStatusOf(cam) {
  return cam?.inference?.status || 'stopped';
}

function SkeletonOverlay({ skeletons, inferW, inferH }) {
  const lines = useMemo(() => {
    if (!Array.isArray(skeletons) || !skeletons.length) return [];
    const fw = inferW > 0 ? inferW : 1;
    const fh = inferH > 0 ? inferH : 1;
    const out = [];
    skeletons.forEach((person, pi) => {
      const pts = person?.keypoints;
      if (!Array.isArray(pts) || !pts.length) return;
      COCO_LINES.forEach((pair, li) => {
        const p1 = pts[pair[0]];
        const p2 = pts[pair[1]];
        if (!p1 || !p2 || p1[2] <= SKELETON_CONF || p2[2] <= SKELETON_CONF) return;
        const [x1, y1] = scaleInferPoint(p1[0], p1[1], inferW, inferH, fw, fh);
        const [x2, y2] = scaleInferPoint(p2[0], p2[1], inferW, inferH, fw, fh);
        out.push(
          <line key={`${pi}-${li}`} className="aisle-skel-line" x1={x1} y1={y1} x2={x2} y2={y2} />,
        );
      });
    });
    return out;
  }, [skeletons, inferW, inferH]);

  if (!lines.length) return null;
  const vw = inferW > 0 ? inferW : 1280;
  const vh = inferH > 0 ? inferH : 720;
  return (
    <svg className="aisle-live-skel" viewBox={`0 0 ${vw} ${vh}`} preserveAspectRatio="xMidYMid meet">
      {lines}
    </svg>
  );
}

function CamPane({ cam, label, overlay, status }) {
  const live = Boolean(cam?.id && (cam.online || inferStatusOf(cam) === 'running' || inferStatusOf(cam) === 'starting'));
  const hasSkel = Array.isArray(overlay?.skeletons) && overlay.skeletons.length > 0;
  const starting = status === 'starting' && !hasSkel;
  const waitingPose = live && !hasSkel && !starting && status === 'running';
  return (
    <div className="aisle-live-pane">
      <span className="aisle-live-tag">{label}</span>
      {starting ? <div className="aisle-live-banner">骨架推理启动中</div> : null}
      {waitingPose ? <div className="aisle-live-banner dim">等待姿态数据…</div> : null}
      <div className="aisle-live-stage">
        {cam?.id && live ? (
          <>
            <img alt="" src={cameraStreamUrl(cam.id, 480)} className="aisle-live-frame" />
            <SkeletonOverlay
              skeletons={overlay?.skeletons || []}
              inferW={overlay?.inferW || 0}
              inferH={overlay?.inferH || 0}
            />
          </>
        ) : cam?.id && cam.has_thumbnail ? (
          <img alt="" src={thumbnailUrl(cam.id, cam.last_frame_at)} className="aisle-live-frame" />
        ) : (
          <div className="aisle-live-empty">{cam?.id ? '暂无画面' : '未绑定'}</div>
        )}
      </div>
    </div>
  );
}

function emptyOverlay() {
  return { skeletons: [], inferW: 0, inferH: 0, persons3d: [], collisions: [], alarms: [] };
}

function aisleFingerprint(a) {
  if (!a) return '';
  const meshes = a.slot_meshes || [];
  const walls = a.solved?.walls || [];
  return `${a.aisle_id}|${walls.length}|${meshes.map((m) => `${m.wall_id}:${m.rows}:${m.cols}`).join(',')}`;
}

export default function AisleLivePage() {
  const { aisleId } = useParams();
  const navigate = useNavigate();
  const [aisle, setAisle] = useState(null);
  const [cameras, setCameras] = useState([]);
  const [msg, setMsg] = useState('');
  const [busy, setBusy] = useState(false);
  const [overlayL, setOverlayL] = useState(emptyOverlay);
  const [overlayR, setOverlayR] = useState(emptyOverlay);

  const load = useCallback(async () => {
    try {
      const [a, c] = await Promise.all([
        apiGet(`/api/aisles/${encodeURIComponent(aisleId)}`),
        apiGet('/api/cameras?probe=0'),
      ]);
      if (a.status !== 'success' || !a.aisle) {
        setMsg(a.error || '巷道不存在');
        return;
      }
      setAisle((prev) => (aisleFingerprint(prev) === aisleFingerprint(a.aisle) ? prev : a.aisle));
      setCameras(c.items || []);
    } catch (e) {
      setMsg(formatUserError(e.message));
    }
  }, [aisleId]);

  useEffect(() => {
    load();
    const t = setInterval(load, 15000);
    return () => clearInterval(t);
  }, [load]);

  const idL = aisle?.cameras?.L?.camera_id || '';
  const idR = aisle?.cameras?.R?.camera_id || '';
  const camL = cameras.find((c) => c.id === idL);
  const camR = cameras.find((c) => c.id === idR);
  const inferOn = aisleInferOn(camL, camR);
  const inferStatusRaw = aisleInferStatus(camL, camR);
  const inferLabel = AISLE_INFER_LABEL[inferStatusRaw] || AISLE_INFER_LABEL.stopped;

  useEffect(() => {
    if (inferStatusRaw !== 'starting') return undefined;
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [inferStatusRaw, load]);

  useEffect(() => {
    const closers = [];
    const bind = (cameraId, setOverlay) => {
      if (!cameraId || (inferStatusRaw !== 'running' && inferStatusRaw !== 'starting')) {
        setOverlay(emptyOverlay());
        return;
      }
      closers.push(openCameraLiveStream(cameraId, {
        onFrame: (data) => {
          if (!data || typeof data !== 'object') return;
          setOverlay((prev) => {
            const p3 = Array.isArray(data.persons_3d) ? data.persons_3d : [];
            const cols = Array.isArray(data.collisions) ? data.collisions : [];
            const alarms = Array.isArray(data.alarm_collisions) ? data.alarm_collisions : [];
            const stereoIn = p3.filter((p) => p?.xyz && p.preview !== true);
            const prevStereo = (prev.persons3d || []).some((p) => p?.xyz && p.preview !== true);
            // 单路预览帧不要盖掉已经配上的立体骨架
            const persons3d = stereoIn.length ? stereoIn : (p3.length && !prevStereo ? p3 : prev.persons3d);
            const got3d = stereoIn.length > 0 || (p3.length > 0 && !prevStereo);
            return {
              skeletons: Array.isArray(data.skeletons) ? data.skeletons : [],
              inferW: Number(data.infer_width) || prev.inferW || 0,
              inferH: Number(data.infer_height) || prev.inferH || 0,
              persons3d,
              collisions: got3d || cols.length ? cols : prev.collisions,
              alarms: got3d || alarms.length ? alarms : prev.alarms,
            };
          });
        },
      }));
    };
    bind(idL, setOverlayL);
    bind(idR, setOverlayR);
    return () => {
      closers.forEach((fn) => fn?.());
    };
  }, [idL, idR, inferStatusRaw]);

  const persons3d = (() => {
    const a = overlayL.persons3d || [];
    const b = overlayR.persons3d || [];
    const pick = (list) => list.filter((p) => p?.xyz && p.preview !== true);
    const sa = pick(a);
    if (sa.length) return sa;
    const sb = pick(b);
    if (sb.length) return sb;
    return a.length ? a : b;
  })();
  const collisions = overlayL.collisions?.length ? overlayL.collisions : overlayR.collisions;
  const alarms = overlayL.alarms?.length ? overlayL.alarms : overlayR.alarms;
  const hasLiveSkel = (overlayL.skeletons?.length || 0) + (overlayR.skeletons?.length || 0) > 0;
  const inferStatus = hasLiveSkel && inferStatusRaw === 'starting' ? 'running' : inferStatusRaw;
  const inferLabelUi = AISLE_INFER_LABEL[inferStatus] || inferLabel;
  const hudText = (alarms || []).length
    ? `告警 ${alarms.join(' · ')}`
    : persons3d.length
      ? (persons3d.some((p) => p.preview) ? '单路预览 · 等待对侧配对成立体' : '立体骨架')
      : inferStatus === 'running'
        ? '等待 3D 姿态…'
        : '';

  const toggleInfer = async (on) => {
    if (!idL || !idR) {
      setMsg('请先到巷道标注绑定左右路');
      return;
    }
    setBusy(true);
    if (on) setMsg('骨架推理启动中');
    try {
      const r = on ? await startAisleInference(idL, idR) : await stopAisleInference(idL, idR);
      if (!r.ok) setMsg(r.error || (on ? '启动失败' : '停止失败'));
      else setMsg(on ? '左右路检测已启动' : '本巷道检测已停止');
      await load();
    } catch (e) {
      setMsg(formatUserError(e.message));
    } finally {
      setBusy(false);
    }
  };

  if (!aisle && msg) {
    return (
      <div className="page aisle-live-page">
        <p className="aisle-live-msg">{msg}</p>
        <button type="button" onClick={() => navigate('/')}>返回总览</button>
      </div>
    );
  }

  const btnBusy = busy || (inferStatus === 'starting' && !hasLiveSkel);
  const btnText = btnBusy
    ? '骨架推理启动中'
    : inferOn
      ? '停止本巷道检测'
      : '启动本巷道检测';

  return (
    <div className="aisle-live-page">
      <div className="aisle-live-app">
        <CamPane
          cam={camL}
          label={`左路 · ${camL?.name || idL || '未绑定'}`}
          overlay={overlayL}
          status={inferStatus}
        />
        <div className="aisle-live-3d">
          <span className="aisle-live-tag">巷道 3D · 拖拽旋转</span>
          {hudText ? (
            <div className={`aisle-live-hud${(alarms || []).length ? ' alarm' : persons3d.some((p) => p.preview) ? ' preview' : ''}`}>
              {hudText}
            </div>
          ) : null}
          {inferStatus === 'starting' && !persons3d.length && !hasLiveSkel ? (
            <div className="aisle-live-banner">骨架推理启动中</div>
          ) : null}
          {aisle?.solved?.ok ? (
            <AisleScene3D
              aisle={aisle}
              persons3d={persons3d}
              collisions={collisions}
              alarms={alarms}
            />
          ) : (
            <div className="aisle-live-empty">尚未反解。请先到标注页完成四角与反解。</div>
          )}
        </div>
        <CamPane
          cam={camR}
          label={`右路 · ${camR?.name || idR || '未绑定'}`}
          overlay={overlayR}
          status={inferStatus}
        />
        <div className="aisle-live-bar">
          <strong>{aisle?.aisle_id || aisleId}</strong>
          <span className={`aisle-live-status ${inferStatus}`}>{inferLabelUi}</span>
          <button
            type="button"
            className="pri"
            disabled={busy || !idL || !idR || (inferStatus === 'starting' && !hasLiveSkel)}
            onClick={() => toggleInfer(!inferOn)}
          >
            {btnText}
          </button>
          <Link to="/aisle">去标注</Link>
          <Link to="/">返回总览</Link>
          <span className="aisle-live-hint">
            左右路同一 worker。一次启动两台，不要分开开检测。
          </span>
          {msg ? <span className="aisle-live-msg">{msg}</span> : null}
          {camL?.inference?.message ? (
            <span className="aisle-live-hint">{formatInferenceMessage(camL.inference.message)}</span>
          ) : null}
        </div>
      </div>
    </div>
  );
}
