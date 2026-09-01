import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import AisleScene3D from '../components/AisleScene3D.jsx';
import { apiGet, cameraPlaybackUrl, cameraStreamUrl, openCameraLiveStream, thumbnailUrl } from '../api/client.js';
import { usePreviewStream } from '../hooks/usePreviewStream.js';
import {
  AISLE_INFER_LABEL,
  aisleInferOn,
  aisleInferStatus,
  startAisleInference,
  stopAisleInference,
} from '../lib/aisleInference.js';
import { COCO_LINES, SKELETON_CONF, scaleInferPoint } from '../lib/cocoSkeleton.js';
import { resolveCameraModelLabel } from '../lib/cameraSettings.js';
import { formatInferenceMessage, formatStreamError, formatUserError } from '../lib/userFacingText.js';
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

function CamPane({ cam, label, overlay, status, playback }) {
  const videoRef = useRef(null);
  const imgRef = useRef(null);
  const [hasMedia, setHasMedia] = useState(false);
  useEffect(() => {
    setHasMedia(false);
  }, [cam?.id]);

  const waitingPb = Boolean(cam?.id && playback === undefined);
  const rtcOk = Boolean(cam?.id && playback?.formats?.webrtc?.available);
  const inferLive = inferStatusOf(cam) === 'running' || inferStatusOf(cam) === 'starting';
  const useMjpeg = Boolean(cam?.id && !waitingPb && !rtcOk && (cam.online || inferLive));
  const format = rtcOk ? 'webrtc' : 'mjpeg';
  const enabled = Boolean(cam?.id && !waitingPb && (rtcOk || useMjpeg));
  const mjpegSrc = useMjpeg ? cameraStreamUrl(cam.id, 480) : '';
  const { streamError } = usePreviewStream({
    format,
    playback,
    mjpegSrc,
    videoRef,
    imgRef,
    enabled,
  });

  const streamHint = formatStreamError(streamError)
    || (!hasMedia ? formatStreamError(cam?.stream_error) : '');
  const hasSkel = Array.isArray(overlay?.skeletons) && overlay.skeletons.length > 0;
  const starting = status === 'starting' && !hasSkel;
  const waitingPose = (hasMedia || inferLive) && !hasSkel && !starting && status === 'running';
  const emptyText = !cam?.id ? '未绑定' : (streamHint || '暂无画面');
  const showThumb = Boolean(cam?.id && cam.has_thumbnail && !hasMedia);
  const showEmpty = Boolean(
    cam?.id && !hasMedia && !cam.has_thumbnail && !waitingPb && !enabled,
  );

  return (
    <div className="aisle-live-pane">
      <span className="aisle-live-tag">{label}</span>
      {starting ? <div className="aisle-live-banner">骨架推理启动中</div> : null}
      {waitingPose ? <div className="aisle-live-banner dim">等待姿态数据…</div> : null}
      {streamHint && !starting && !hasMedia ? (
        <div className="aisle-live-banner stream-err" title={streamHint}>{streamHint}</div>
      ) : null}
      <div className="aisle-live-stage">
        {showThumb ? (
          <img alt="" src={thumbnailUrl(cam.id, cam.last_frame_at)} className="aisle-live-frame aisle-live-thumb" />
        ) : null}
        {cam?.id ? (
          <>
            <video
              ref={videoRef}
              className={`aisle-live-frame${rtcOk ? '' : ' is-hidden'}`}
              autoPlay
              muted
              playsInline
              onPlaying={() => setHasMedia(true)}
            />
            <img
              ref={imgRef}
              alt=""
              className={`aisle-live-frame${useMjpeg ? '' : ' is-hidden'}`}
              onLoad={() => setHasMedia(true)}
              onError={() => setHasMedia(false)}
            />
            <SkeletonOverlay
              skeletons={overlay?.skeletons || []}
              inferW={overlay?.inferW || 0}
              inferH={overlay?.inferH || 0}
            />
          </>
        ) : (
          <div className="aisle-live-empty">未绑定</div>
        )}
        {showEmpty ? <div className="aisle-live-empty">{emptyText}</div> : null}
      </div>
    </div>
  );
}

function emptyOverlay() {
  return { skeletons: [], inferW: 0, inferH: 0, persons3d: [], collisions: [], alarms: [], ts: 0 };
}

function newerOverlay(a, b) {
  return (Number(a?.ts) || 0) >= (Number(b?.ts) || 0) ? a : b;
}

function stereoPersons(list) {
  return (list || []).filter((p) => p?.xyz && p.preview !== true);
}

function mergeAisleOverlay(prev, data) {
  const hasP3 = Array.isArray(data.persons_3d);
  const p3 = hasP3 ? data.persons_3d : [];
  const stereoIn = stereoPersons(p3);
  let persons3d = hasP3 ? (stereoIn.length ? stereoIn : p3) : (prev.persons3d || []);
  const hasCols = Array.isArray(data.collisions);
  const hasAlarms = Array.isArray(data.alarm_collisions);
  const cols = hasCols ? data.collisions : prev.collisions;
  const alarms = hasAlarms ? data.alarm_collisions : prev.alarms;
  if (hasAlarms && !alarms.length && Array.isArray(persons3d) && persons3d.length) {
    persons3d = persons3d.map((p) => (
      p?.wrist_alarm ? { ...p, wrist_alarm: { 9: false, 10: false } } : p
    ));
  }
  return {
    skeletons: Array.isArray(data.skeletons) ? data.skeletons : [],
    inferW: Number(data.infer_width) || prev.inferW || 0,
    inferH: Number(data.infer_height) || prev.inferH || 0,
    persons3d,
    collisions: cols,
    alarms,
    ts: Number(data.ts) || prev.ts || 0,
  };
}

/** 任一 overlay 还有 3D 就画；两路都空才清。避免一路空 pose 把预览骨架闪掉。 */
function pickOverlay3d(overlayL, overlayR) {
  const a = overlayL.persons3d || [];
  const b = overlayR.persons3d || [];
  if (stereoPersons(a).length) return overlayL;
  if (stereoPersons(b).length) return overlayR;
  if (a.length) return overlayL;
  if (b.length) return overlayR;
  return newerOverlay(overlayL, overlayR);
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
  const [playbackById, setPlaybackById] = useState({});

  const load = useCallback(async () => {
    try {
      const [a, c] = await Promise.all([
        apiGet(`/api/aisles/${encodeURIComponent(aisleId)}`),
        apiGet('/api/cameras?probe=false'),
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
  }, [load]);

  const idL = aisle?.cameras?.L?.camera_id || '';
  const idR = aisle?.cameras?.R?.camera_id || '';
  const camL = cameras.find((c) => c.id === idL);
  const camR = cameras.find((c) => c.id === idR);
  const inferOn = aisleInferOn(camL, camR);
  const inferStatusRaw = aisleInferStatus(camL, camR);
  const inferLabel = AISLE_INFER_LABEL[inferStatusRaw] || AISLE_INFER_LABEL.stopped;

  const refreshCameraMeta = useCallback(async () => {
    const ids = [idL, idR].filter(Boolean);
    if (!ids.length) return;
    try {
      const rows = await Promise.all(
        ids.map((cid) => apiGet(`/api/cameras/${encodeURIComponent(cid)}?settings=0`)),
      );
      setCameras((prev) => {
        const next = [...prev];
        for (const data of rows) {
          const cam = data?.camera;
          if (!cam?.id) continue;
          const i = next.findIndex((c) => c.id === cam.id);
          if (i >= 0) next[i] = { ...next[i], ...cam };
          else next.push(cam);
        }
        return next;
      });
    } catch {
      /* 与原监控页一致：刷新失败忽略 */
    }
  }, [idL, idR]);

  useEffect(() => {
    let cancelled = false;
    const ids = [idL, idR].filter(Boolean);
    ids.forEach((cid) => {
      apiGet(cameraPlaybackUrl(cid))
        .then((pb) => {
          if (cancelled) return;
          setPlaybackById((prev) => ({
            ...prev,
            [cid]: pb?.status === 'success' ? pb : null,
          }));
        })
        .catch(() => {
          if (!cancelled) {
            setPlaybackById((prev) => ({ ...prev, [cid]: null }));
          }
        });
    });
    return () => {
      cancelled = true;
    };
  }, [idL, idR]);

  useEffect(() => {
    if (inferStatusRaw !== 'starting') return undefined;
    const t = setInterval(refreshCameraMeta, 2000);
    return () => clearInterval(t);
  }, [inferStatusRaw, refreshCameraMeta]);

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
          setOverlay((prev) => mergeAisleOverlay(prev, data));
        },
      }));
    };
    bind(idL, setOverlayL);
    bind(idR, setOverlayR);
    return () => {
      closers.forEach((fn) => fn?.());
    };
  }, [idL, idR, inferStatusRaw]);

  const hasLiveSkel = (overlayL.skeletons?.length || 0) + (overlayR.skeletons?.length || 0) > 0;
  const src3d = pickOverlay3d(overlayL, overlayR);
  const raw3d = src3d.persons3d || [];
  const stereo = stereoPersons(raw3d);
  const persons3d = stereo.length ? stereo : raw3d;
  const collisions = src3d.collisions || [];
  const alarms = src3d.alarms || [];
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
      await refreshCameraMeta();
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
  const modelL = resolveCameraModelLabel(camL);
  const modelR = resolveCameraModelLabel(camR);
  const modelHint = camL || camR
    ? (modelL === modelR ? `使用模型 ${modelL}` : `左 ${modelL} · 右 ${modelR}`)
    : '';
  const inferHint = formatInferenceMessage(camL?.inference?.message)
    || formatInferenceMessage(camR?.inference?.message);

  return (
    <div className="aisle-live-page">
      <div className="aisle-live-app">
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
          {modelHint ? <span className="aisle-live-model">{modelHint}</span> : null}
          {inferHint ? <span className="aisle-live-hint">{inferHint}</span> : null}
          <Link to="/aisle">去标注</Link>
          <Link to="/">返回总览</Link>
          <span className="aisle-live-hint">
            左右路同一 worker。一次启动两台，不要分开开检测。
          </span>
          {msg ? <span className="aisle-live-msg">{msg}</span> : null}
        </div>
        <div className="aisle-live-cams">
          <CamPane
            cam={camL}
            label={`左路 · ${camL?.name || idL || '未绑定'}`}
            overlay={overlayL}
            status={inferStatus}
            playback={idL ? playbackById[idL] : null}
          />
          <CamPane
            cam={camR}
            label={`右路 · ${camR?.name || idR || '未绑定'}`}
            overlay={overlayR}
            status={inferStatus}
            playback={idR ? playbackById[idR] : null}
          />
        </div>
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
      </div>
    </div>
  );
}
