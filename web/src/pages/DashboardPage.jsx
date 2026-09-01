import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import CameraSetupDrawer from '../components/CameraSetupDrawer';
import InferenceToggle from '../components/InferenceToggle';
import { confirmDeleteCamera } from '../lib/confirmDelete';
import { apiDelete, apiGet, apiPost, apiPut, formatDuration, thumbnailUrl } from '../api/client';
import {
  STREAM_CONFIG_SAVED_HINT,
  formatUserError,
} from '../lib/userFacingText';
import { aisleInferOn, aisleInferStatus, AISLE_INFER_LABEL, startAisleInference, stopAisleInference } from '../lib/aisleInference';
import { cameraToForm, emptyCameraForm, formToCameraPayload } from '../lib/cameraStreamForm';
import './DashboardPage.css';

const POLL_MS = 30000;

export default function DashboardPage() {
  const navigate = useNavigate();
  const [cameras, setCameras] = useState([]);
  const [aisles, setAisles] = useState([]);
  const [listLoading, setListLoading] = useState(true);
  const [probing, setProbing] = useState(false);
  const [msg, setMsg] = useState('');
  const [msgErr, setMsgErr] = useState(false);
  const [refreshingId, setRefreshingId] = useState(null);
  const [inferLoadingId, setInferLoadingId] = useState(null);
  const [batchInferAction, setBatchInferAction] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerMode, setDrawerMode] = useState('edit');
  const [setupCamera, setSetupCamera] = useState(null);
  const [form, setForm] = useState(emptyCameraForm());
  const [saving, setSaving] = useState(false);
  const [configHint, setConfigHint] = useState('');
  const [globalSettings, setGlobalSettings] = useState({});

  const applyConfigHint = (data) => {
    if (data?.reload_hint || data?.mediamtx?.reload_hint) {
      setConfigHint(STREAM_CONFIG_SAVED_HINT);
    }
  };

  const applyCameraItems = useCallback((items) => {
    if (!Array.isArray(items)) return false;
    const now = Date.now() / 1000;
    const mapped = items.map((item) => ({
      ...item,
      _syncedAt: now,
      _displayActivity: item.activity_seconds ?? 0,
    }));
    setCameras(mapped);
    setSetupCamera((prev) => {
      if (!prev) return prev;
      return mapped.find((c) => c.id === prev.id) || prev;
    });
    setMsg(`共 ${items.length} 路摄像头 · 上次更新 ${new Date().toLocaleTimeString()}`);
    setMsgErr(false);
    return true;
  }, []);

  const loadCameras = useCallback(async ({ probe = false } = {}) => {
    if (probe) setProbing(true);
    try {
      const qs = probe ? '' : '?probe=false';
      const data = await apiGet(`/api/cameras${qs}`);
      if (data.status !== 'success' || !Array.isArray(data.items)) {
        setMsg(formatUserError(data.error) || '加载失败');
        setMsgErr(true);
        return false;
      }
      applyCameraItems(data.items);
      try {
        const ad = await apiGet('/api/aisles');
        if (ad.status === 'success' && Array.isArray(ad.items)) setAisles(ad.items);
      } catch {
        /* 巷道列表失败时仍展示单路卡片 */
      }
      return true;
    } catch (e) {
      setMsg(formatUserError(e.message) || '无法连接服务器');
      setMsgErr(true);
      return false;
    } finally {
      if (probe) {
        setProbing(false);
      } else {
        setListLoading(false);
      }
    }
  }, [applyCameraItems]);

  const refreshListFastThenProbe = useCallback(async () => {
    await loadCameras({ probe: false });
    void loadCameras({ probe: true });
  }, [loadCameras]);

  const refreshCamerasAfterMutation = useCallback(
    async (mutationData) => {
      if (applyCameraItems(mutationData?.items)) {
        setListLoading(false);
        void loadCameras({ probe: true });
        return;
      }
      await refreshListFastThenProbe();
    },
    [applyCameraItems, refreshListFastThenProbe],
  );

  useEffect(() => {
    let cancelled = false;
    (async () => {
      await loadCameras({ probe: false });
      if (!cancelled) void loadCameras({ probe: true });
    })();
    const poll = setInterval(() => loadCameras({ probe: false }), POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(poll);
    };
  }, [loadCameras]);

  useEffect(() => {
    const hasStarting = cameras.some((c) => c.inference?.status === 'starting');
    if (!hasStarting) return undefined;
    const fast = setInterval(() => loadCameras({ probe: false }), 5000);
    return () => clearInterval(fast);
  }, [cameras, loadCameras]);

  useEffect(() => {
    const tick = setInterval(() => {
      const now = Date.now() / 1000;
      setCameras((prev) =>
        prev.map((cam) => {
          if (!cam.online || !cam._syncedAt) {
            return { ...cam, _displayActivity: cam.activity_seconds };
          }
          const elapsed = Math.floor(now - cam._syncedAt);
          return { ...cam, _displayActivity: (cam.activity_seconds || 0) + elapsed };
        }),
      );
    }, 1000);
    return () => clearInterval(tick);
  }, []);

  const openCreate = () => {
    setDrawerMode('create');
    setSetupCamera(null);
    setForm(emptyCameraForm());
    setDrawerOpen(true);
    loadGlobalSettings();
  };

  const loadGlobalSettings = useCallback(async () => {
    try {
      const data = await apiGet('/api/settings');
      if (data.items) setGlobalSettings(data.items);
    } catch {
      /* ignore */
    }
  }, []);

  const openSetup = async (cam) => {
    setDrawerMode('edit');
    setSetupCamera(cam);
    setForm(cameraToForm(cam));
    setDrawerOpen(true);
    let settings = { ...(cam.settings || {}) };
    let fullCam = cam;
    try {
      const detail = await apiGet(`/api/cameras/${encodeURIComponent(cam.id)}`);
      if (detail?.camera) {
        fullCam = detail.camera;
        setSetupCamera((prev) => ({ ...prev, ...fullCam }));
        settings = { ...(fullCam.settings || {}) };
        if (fullCam.global_defaults && typeof fullCam.global_defaults === 'object') {
          setGlobalSettings(fullCam.global_defaults);
        } else {
          await loadGlobalSettings();
        }
      } else {
        await loadGlobalSettings();
      }
    } catch {
      await loadGlobalSettings();
    }
    setForm({ ...cameraToForm(fullCam), settings });
  };

  const closeDrawer = () => {
    setDrawerOpen(false);
    setSetupCamera(null);
  };

  useEffect(() => {
    if (!drawerOpen) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') closeDrawer();
    };
    document.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = '';
    };
  }, [drawerOpen]);

  const onFormChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const saveFromDrawer = async () => {
    const payload = formToCameraPayload(form);
    setSaving(true);
    try {
      const data =
        drawerMode === 'create'
          ? await apiPost('/api/cameras', payload)
          : await apiPut(`/api/cameras/${encodeURIComponent(setupCamera.id)}`, payload);
      if (data.error) {
        alert(formatUserError(data.error));
        return;
      }
      applyConfigHint(data);
      closeDrawer();
      await refreshCamerasAfterMutation(data);
    } catch (err) {
      alert(formatUserError(err.message) || '保存失败');
    } finally {
      setSaving(false);
    }
  };

  const deleteFromDrawer = async () => {
    if (!setupCamera) return;
    if (!confirmDeleteCamera(setupCamera.name)) return;
    setSaving(true);
    try {
      const data = await apiDelete(`/api/cameras/${encodeURIComponent(setupCamera.id)}`);
      if (data.error) {
        alert(formatUserError(data.error));
        return;
      }
      applyConfigHint(data);
      closeDrawer();
      await refreshCamerasAfterMutation(data);
    } catch (err) {
      alert(formatUserError(err.message) || '删除失败');
    } finally {
      setSaving(false);
    }
  };

  const startInference = async (cam) => {
    const group = aisles.find((a) => a.camera_l === cam.id || a.camera_r === cam.id);
    if (group?.camera_l && group?.camera_r) {
      await toggleAisleInference(group, true);
      return;
    }
    setInferLoadingId(cam.id);
    try {
      const data = await apiPost(`/api/cameras/${encodeURIComponent(cam.id)}/inference/start`, {});
      if (data.error) {
        alert(formatUserError(data.error));
        return;
      }
      await loadCameras();
    } catch (e) {
      alert(formatUserError(e.message) || '启动检测失败');
    } finally {
      setInferLoadingId(null);
    }
  };

  const toggleInference = async (cam, turnOn) => {
    if (turnOn) await startInference(cam);
    else await stopInference(cam);
  };

  const stopInference = async (cam) => {
    const group = aisles.find((a) => a.camera_l === cam.id || a.camera_r === cam.id);
    if (group?.camera_l && group?.camera_r) {
      await toggleAisleInference(group, false);
      return;
    }
    setInferLoadingId(cam.id);
    try {
      const data = await apiPost(`/api/cameras/${encodeURIComponent(cam.id)}/inference/stop`, {});
      if (data.error) {
        alert(formatUserError(data.error));
        return;
      }
      await loadCameras();
    } catch (e) {
      alert(formatUserError(e.message) || '停止检测失败');
    } finally {
      setInferLoadingId(null);
    }
  };

  const toggleAisleInference = async (aisle, turnOn) => {
    setInferLoadingId(aisle.aisle_id);
    if (turnOn) {
      setMsg('骨架推理启动中');
      setMsgErr(false);
    }
    try {
      const r = turnOn
        ? await startAisleInference(aisle.camera_l, aisle.camera_r)
        : await stopAisleInference(aisle.camera_l, aisle.camera_r);
      if (!r.ok) {
        alert(formatUserError(r.error) || (turnOn ? '启动失败' : '停止失败'));
        return;
      }
      await loadCameras({ probe: false });
    } catch (e) {
      alert(formatUserError(e.message) || (turnOn ? '启动失败' : '停止失败'));
    } finally {
      setInferLoadingId(null);
    }
  };

  const startAllInference = async () => {
    if (!window.confirm('确认启动全部已启用摄像头的智能检测？')) return;
    setBatchInferAction('start');
    setMsg('正在批量启动智能检测…');
    setMsgErr(false);
    try {
      const data = await apiPost('/api/inference/start-all', {});
      if (data.error) {
        setMsg(formatUserError(data.error));
        setMsgErr(true);
        return;
      }
      const failed = Number(data.failed) || 0;
      setMsg(
        `批量启动完成：成功 ${data.started ?? 0} 路，跳过 ${data.skipped ?? 0} 路，失败 ${failed} 路`,
      );
      setMsgErr(failed > 0);
      await loadCameras({ probe: false });
    } catch (e) {
      setMsg(formatUserError(e.message) || '批量启动失败');
      setMsgErr(true);
    } finally {
      setBatchInferAction(null);
    }
  };

  const stopAllInference = async () => {
    if (!window.confirm('确认停止全部摄像头的智能检测？')) return;
    setBatchInferAction('stop');
    setMsg('正在批量停止智能检测…');
    setMsgErr(false);
    try {
      const data = await apiPost('/api/inference/stop-all', {});
      if (data.error) {
        setMsg(formatUserError(data.error));
        setMsgErr(true);
        return;
      }
      const failed = Number(data.failed) || 0;
      setMsg(
        `批量停止完成：已停 ${data.stopped ?? 0} 路，跳过 ${data.skipped ?? 0} 路，失败 ${failed} 路`,
      );
      setMsgErr(failed > 0);
      await loadCameras({ probe: false });
    } catch (e) {
      setMsg(formatUserError(e.message) || '批量停止失败');
      setMsgErr(true);
    } finally {
      setBatchInferAction(null);
    }
  };

  const batchInferBusy = Boolean(batchInferAction);

  const openMonitor = (cam) => {
    const group = aisles.find((a) => a.camera_l === cam.id || a.camera_r === cam.id);
    if (group?.aisle_id) {
      navigate(`/live/${encodeURIComponent(group.aisle_id)}`);
      return;
    }
    navigate(`/monitor?camera=${encodeURIComponent(cam.id)}`);
  };

  const groupedIds = new Set(
    aisles.flatMap((a) => [a.camera_l, a.camera_r].filter(Boolean)),
  );
  const ungrouped = cameras.filter((c) => !groupedIds.has(c.id));
  const aisleCards = aisles.filter((a) => a.camera_l && a.camera_r);

  const captureFrame = async (cam) => {
    setRefreshingId(cam.id);
    try {
      const data = await apiPost(`/api/cameras/${encodeURIComponent(cam.id)}/capture`, {});
      if (data?.error || data?.status !== 'success') {
        alert(formatUserError(data?.error) || '抓帧失败');
        return;
      }
      const patch = {
        has_thumbnail: true,
        last_frame_at: data.last_frame_at ?? Date.now() / 1000,
        _thumbNonce: Date.now(),
        online: data.online ?? cam.online,
        activity_seconds: data.activity_seconds ?? cam.activity_seconds,
      };
      setCameras((prev) => prev.map((c) => (c.id === cam.id ? { ...c, ...patch } : c)));
      setSetupCamera((prev) => (prev?.id === cam.id ? { ...prev, ...patch } : prev));
    } catch (e) {
      alert(formatUserError(e.message) || '抓帧失败');
    } finally {
      setRefreshingId(null);
    }
  };

  const drawerActionLoading = inferLoadingId === setupCamera?.id || refreshingId === setupCamera?.id;
  const inferToggleDisabled = Boolean(inferLoadingId) || batchInferBusy;
  const drawerCamera = setupCamera
    ? cameras.find((c) => c.id === setupCamera.id) || setupCamera
    : null;

  return (
    <div className="page dashboard-page">
        <h1 className="page-title">摄像头总览</h1>
        <p className="dash-lead">成组巷道点进即是「左路 · 3D · 右路」。未编组的仍单路预览。</p>

        <div className="toolbar">
          <span className={`msg ${msgErr ? 'err' : ''}`}>
            {listLoading ? '加载列表…' : msg}
            {probing && !listLoading ? ' · 正在探测在线状态' : ''}
          </span>
          <div className="toolbar-actions">
            <button
              type="button"
              className="btn-batch"
              title="启动全部已启用摄像头的智能检测"
              disabled={batchInferBusy || listLoading || !cameras.length}
              onClick={startAllInference}
            >
              {batchInferAction === 'start' ? '启动中…' : '全部启动检测'}
            </button>
            <button
              type="button"
              className="btn-batch btn-batch-stop"
              title="停止全部摄像头的智能检测"
              disabled={batchInferBusy || listLoading || !cameras.length}
              onClick={stopAllInference}
            >
              {batchInferAction === 'stop' ? '停止中…' : '全部停止检测'}
            </button>
            <button
              type="button"
              className="btn-icon btn-icon-primary"
              title="添加摄像头"
              aria-label="添加摄像头"
              onClick={openCreate}
            >
              +
            </button>
            <button
              type="button"
              className="btn-icon"
              title="刷新列表"
              aria-label="刷新列表"
              disabled={listLoading || probing}
              onClick={() => refreshListFastThenProbe()}
            >
              ↻
            </button>
          </div>
        </div>

        {configHint && <div className="config-hint">{configHint}</div>}

        <div className="grid">
          {listLoading ? (
            <div className="empty grid-status">加载中…</div>
          ) : !cameras.length ? (
            <div className="empty">暂无摄像头，点击「添加摄像头」开始配置。</div>
          ) : (
            <>
              {aisleCards.map((aisle) => {
                const left = cameras.find((c) => c.id === aisle.camera_l);
                const right = cameras.find((c) => c.id === aisle.camera_r);
                const on = aisleInferOn(left, right);
                const inferSt = aisleInferStatus(left, right);
                const inferText = inferLoadingId === aisle.aisle_id && !on
                  ? AISLE_INFER_LABEL.starting
                  : (AISLE_INFER_LABEL[inferSt] || AISLE_INFER_LABEL.stopped);
                return (
                  <article className="card aisle-card" key={aisle.aisle_id}>
                    <div
                      className="card-preview card-preview-link"
                      role="button"
                      tabIndex={0}
                      title="进入巷道检测（左路 + 3D + 右路）"
                      onClick={() => navigate(`/live/${encodeURIComponent(aisle.aisle_id)}`)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          navigate(`/live/${encodeURIComponent(aisle.aisle_id)}`);
                        }
                      }}
                    >
                      <div className="aisle-thumbs">
                        <figure>
                          <span>左路</span>
                          {left?.has_thumbnail ? (
                            <img
                              src={thumbnailUrl(left.id, left.last_frame_at, left._thumbNonce)}
                              alt={left.name}
                            />
                          ) : (
                            <div className="card-preview-empty">暂无画面</div>
                          )}
                        </figure>
                        <div className="aisle-thumb-mid" aria-hidden="true">
                          <span>3D</span>
                        </div>
                        <figure>
                          <span>右路</span>
                          {right?.has_thumbnail ? (
                            <img
                              src={thumbnailUrl(right.id, right.last_frame_at, right._thumbNonce)}
                              alt={right.name}
                            />
                          ) : (
                            <div className="card-preview-empty">暂无画面</div>
                          )}
                        </figure>
                      </div>
                      <div className="card-actions" onClick={(e) => e.stopPropagation()}>
                        <InferenceToggle
                          on={on}
                          loading={inferLoadingId === aisle.aisle_id}
                          disabled={inferToggleDisabled}
                          title={on ? '停止本巷道两路检测' : '同时启动本巷道两路检测'}
                          onToggle={(turnOn) => toggleAisleInference(aisle, turnOn)}
                        />
                        <button
                          type="button"
                          className="btn-icon"
                          title="设置左路"
                          onClick={() => left && openSetup(left)}
                        >
                          ⚙
                        </button>
                      </div>
                      <div className="card-body">
                        <h2 className="card-title">巷道 {aisle.aisle_id}</h2>
                        <div className="card-status">
                          <span className={left?.online || right?.online ? 'st-online' : 'st-offline'}>
                            {left?.name || aisle.camera_l} / {right?.name || aisle.camera_r}
                          </span>
                          <span className="card-status-sep">·</span>
                          <span className={`card-infer ${inferSt}`}>
                            {inferText}
                          </span>
                        </div>
                        <div className="card-url">
                          {aisle.solved ? '已反解 · 点开查看 3D 骨架' : '未反解 · 请先巷道标注'}
                        </div>
                      </div>
                    </div>
                  </article>
                );
              })}
              {ungrouped.map((cam) => (
                <article className="card" key={cam.id}>
                  <div
                    className="card-preview card-preview-link"
                    role="button"
                    tabIndex={0}
                    title="未编入巷道，仅单路预览"
                    onClick={() => openMonitor(cam)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        openMonitor(cam);
                      }
                    }}
                  >
                    {cam.has_thumbnail ? (
                      <img
                        src={thumbnailUrl(cam.id, cam.last_frame_at, cam._thumbNonce)}
                        alt={cam.name}
                      />
                    ) : (
                      <div className="card-preview-empty">暂无画面</div>
                    )}
                    <div className="card-actions" onClick={(e) => e.stopPropagation()}>
                      <button
                        type="button"
                        className="btn-icon"
                        title="抓帧"
                        disabled={refreshingId === cam.id}
                        onClick={() => captureFrame(cam)}
                      >
                        ↻
                      </button>
                      <button
                        type="button"
                        className="btn-icon"
                        title="设置"
                        onClick={() => openSetup(cam)}
                      >
                        ⚙
                      </button>
                    </div>
                    <div className="card-body">
                      <h2 className="card-title">{cam.name}</h2>
                      <div className="card-status">
                        <span className={cam.online ? 'st-online' : 'st-offline'}>
                          {cam.online ? '在线' : '离线'}
                        </span>
                        <span className="card-status-sep">·</span>
                        <span className="card-activity">{formatDuration(cam._displayActivity)}</span>
                        <span className="card-status-sep">·</span>
                        <span className="card-infer stopped">未编入巷道</span>
                      </div>
                      <div className="card-url" title={cam.url}>
                        请到巷道标注绑定左右路后，才能开 3D 检测
                      </div>
                    </div>
                  </div>
                </article>
              ))}
            </>
          )}
        </div>

      <CameraSetupDrawer
        open={drawerOpen}
        mode={drawerMode}
        camera={drawerCamera}
        form={form}
        onChange={onFormChange}
        globalDefaults={globalSettings}
        effectiveSettings={drawerCamera?.effective_settings || {}}
        onClose={closeDrawer}
        onSave={saveFromDrawer}
        onDelete={deleteFromDrawer}
        onCapture={() => drawerCamera && captureFrame(drawerCamera)}
        onToggleInference={(turnOn) => drawerCamera && toggleInference(drawerCamera, turnOn)}
        saving={saving}
        actionLoading={drawerActionLoading}
      />
    </div>
  );
}
