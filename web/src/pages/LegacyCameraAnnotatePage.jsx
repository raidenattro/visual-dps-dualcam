import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import AnnotateControls from '../components/AnnotateControls.jsx';
import { useAnnotateTool } from '../features/annotate/useAnnotateTool.js';
import { apiGet } from '../api/client.js';
import { cameraMonitorPath } from '../lib/aisleNavigation.js';
import { formatUserError } from '../lib/userFacingText.js';
import '../pages/AnnotatePage.css';
import './LegacyCameraAnnotatePage.css';

function groupedIdSet(aisles) {
  const ids = new Set();
  for (const a of aisles || []) {
    if (a.camera_l) ids.add(String(a.camera_l));
    if (a.camera_r) ids.add(String(a.camera_r));
  }
  return ids;
}

export default function LegacyCameraAnnotatePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const cameraId = String(searchParams.get('camera') || '').trim();

  const [cameras, setCameras] = useState([]);
  const [aisles, setAisles] = useState([]);
  const [camDetail, setCamDetail] = useState(null);
  const [loadErr, setLoadErr] = useState('');

  const canvasRef = useRef(null);
  const grouped = useMemo(() => groupedIdSet(aisles), [aisles]);
  const legacyCameras = useMemo(
    () => (cameras || []).filter((c) => c.id && !grouped.has(String(c.id))),
    [cameras, grouped],
  );
  const isGrouped = Boolean(cameraId && grouped.has(cameraId));

  const fixedCamera = useMemo(() => {
    if (!camDetail?.id || isGrouped) return null;
    return {
      id: camDetail.id,
      name: camDetail.name || camDetail.id,
      url: camDetail.url || '',
      last_frame_at: camDetail.last_frame_at,
      has_thumbnail: camDetail.has_thumbnail,
    };
  }, [camDetail, isGrouped]);

  const canvasActive = Boolean(fixedCamera?.id);
  const tool = useAnnotateTool(canvasRef, {
    fixedCamera,
    embedded: true,
    canvasActive,
  });

  const loadLists = useCallback(async () => {
    try {
      const [c, a] = await Promise.all([
        apiGet('/api/cameras?probe=false'),
        apiGet('/api/aisles'),
      ]);
      setCameras(c.items || []);
      setAisles(a.items || []);
    } catch (e) {
      setLoadErr(formatUserError(e.message));
    }
  }, []);

  useEffect(() => {
    loadLists();
  }, [loadLists]);

  useEffect(() => {
    if (!cameraId) {
      setCamDetail(null);
      return;
    }
    let cancelled = false;
    setLoadErr('');
    (async () => {
      try {
        const data = await apiGet(
          `/api/cameras/${encodeURIComponent(cameraId)}?settings=0&probe=false`,
        );
        if (cancelled) return;
        if (data.error || !data.camera) {
          setCamDetail(null);
          setLoadErr(formatUserError(data.error) || '摄像头不存在');
          return;
        }
        setCamDetail(data.camera);
      } catch (e) {
        if (!cancelled) {
          setCamDetail(null);
          setLoadErr(formatUserError(e.message));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [cameraId]);

  const onSelectCamera = (id) => {
    const next = String(id || '').trim();
    if (next) setSearchParams({ camera: next });
    else setSearchParams({});
  };

  return (
    <div className="legacy-annotate-root">
      <header className="legacy-annotate-header">
        <h1 className="page-title">单路 2D 货框标注</h1>
        <p className="legacy-annotate-lead">
          进页自动恢复上次抓帧与标注；需更新画面时点「抓帧」→ 生成货位 → 保存
        </p>
      </header>

      <div className="legacy-annotate-page">
        <aside className="legacy-annotate-sidebar">
          <div className="legacy-annotate-field">
            <label htmlFor="legacy-cam-select">摄像头</label>
            <select
              id="legacy-cam-select"
              value={cameraId}
              onChange={(e) => onSelectCamera(e.target.value)}
            >
              <option value="">请选择未成组单路…</option>
              {legacyCameras.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name || c.id} ({c.path || c.id})
                </option>
              ))}
            </select>
          </div>

          {isGrouped ? (
            <p className="legacy-annotate-msg err">已编入巷道，请用「巷道标注」。</p>
          ) : null}
          {loadErr ? <p className="legacy-annotate-msg err">{loadErr}</p> : null}

          {fixedCamera ? (
            <div className="legacy-annotate-op-block">
              <div className="group-title">货架与保存</div>
              <label className="legacy-annotate-field-inline">
                货架名称
                <input
                  type="text"
                  value={tool.cameraName}
                  onChange={(e) => tool.setCameraName(e.target.value)}
                />
              </label>
              <div className="legacy-annotate-actions">
                <button type="button" className="legacy-btn primary" onClick={() => tool.captureFrame()}>
                  抓帧
                </button>
                <button
                  type="button"
                  className="legacy-btn primary"
                  disabled={!tool.canSave}
                  onClick={() => tool.saveAnnotation()}
                >
                  保存
                </button>
                <Link className="legacy-btn ghost" to={cameraMonitorPath({ cameraId: fixedCamera.id })}>
                  监控
                </Link>
              </div>
            </div>
          ) : null}

          {canvasActive ? <AnnotateControls tool={tool} embedded /> : null}

          <button type="button" className="legacy-btn ghost block" onClick={() => navigate('/')}>
            返回总览
          </button>
        </aside>

        <div className="legacy-annotate-stage">
          {!cameraId ? (
            <p className="legacy-annotate-placeholder">请选择单路摄像头</p>
          ) : (
            <>
              <div className="legacy-annotate-canvas-host">
                <canvas ref={canvasRef} className="legacy-annotate-canvas" />
              </div>
              {tool.statusHtml ? (
                <div
                  className={`legacy-annotate-status ${tool.statusClass || ''}`}
                  dangerouslySetInnerHTML={{ __html: tool.statusHtml }}
                />
              ) : null}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
