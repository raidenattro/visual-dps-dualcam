import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import CameraSetupDrawer from '../components/CameraSetupDrawer';
import AisleCreateDrawer from '../components/AisleCreateDrawer';
import InferenceToggle from '../components/InferenceToggle';
import { confirmDeleteAisle, confirmDeleteCamera } from '../lib/confirmDelete';
import { apiDelete, apiGet, apiPost, apiPut, formatDuration, thumbnailUrl } from '../api/client';
import {
  STREAM_CONFIG_SAVED_HINT,
  formatStreamError,
  formatUserError,
} from '../lib/userFacingText';
import { aisleInferOn, aisleInferStatus, AISLE_INFER_LABEL, startAisleInference, stopAisleInference } from '../lib/aisleInference';
import { applyFormFields, cameraToForm, emptyAisleCreateForm, emptyCameraForm, formToCameraPayload } from '../lib/cameraStreamForm';
import './DashboardPage.css';

const POLL_MS = 30000;

function streamHintOf(cam) {
  if (!cam || cam.online) return '';
  return formatStreamError(cam.stream_error);
}

function aisleStreamHint(left, right) {
  const parts = [];
  const le = streamHintOf(left);
  const re = streamHintOf(right);
  if (le) parts.push(`左路 ${le}`);
  if (re) parts.push(`右路 ${re}`);
  return parts.join(' · ');
}

/** 创建接口返回的是整份 aisle JSON，总览卡片要的是 list 那套 camera_l / camera_r */
function aisleCardFromMutation(data) {
  const a = data?.aisle;
  if (!a?.aisle_id) return null;
  const cams = a.cameras || {};
  const idOf = (role, fallback) => {
    const fromRole = cams[role]?.camera_id;
    if (fromRole) return fromRole;
    if (typeof fallback === 'string') return fallback;
    return fallback?.id || '';
  };
  return {
    aisle_id: a.aisle_id,
    camera_l: idOf('L', data.camera_l),
    camera_r: idOf('R', data.camera_r),
    solved: Boolean(a.solved?.ok),
    mesh_walls: Array.isArray(a.slot_meshes) ? a.slot_meshes.length : Number(a.mesh_walls) || 0,
    logical_shard: a.logical_shard,
  };
}

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
  const [inferLoadingAction, setInferLoadingAction] = useState(null);
  const [batchInferAction, setBatchInferAction] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerMode, setDrawerMode] = useState('edit');
  const [setupCamera, setSetupCamera] = useState(null);
  const [setupAisle, setSetupAisle] = useState(null);
  const [form, setForm] = useState(emptyCameraForm());
  const [aisleForm, setAisleForm] = useState(emptyAisleCreateForm());
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
    setCameras((prev) => {
      const prevById = new Map(prev.map((c) => [c.id, c]));
      return items.map((item) => {
        const old = prevById.get(item.id);
        const hasListStatus = typeof item.has_thumbnail === 'boolean' || typeof item.online === 'boolean';
        if (!old || hasListStatus) {
          return {
            ...item,
            _syncedAt: now,
            _displayActivity: item.activity_seconds ?? old?._displayActivity ?? 0,
            _thumbNonce: item._thumbNonce ?? old?._thumbNonce,
            has_thumbnail: item.has_thumbnail ?? old?.has_thumbnail,
            last_frame_at: item.last_frame_at ?? old?.last_frame_at,
          };
        }
        return {
          ...old,
          ...item,
          has_thumbnail: old.has_thumbnail,
          last_frame_at: old.last_frame_at,
          online: old.online,
          activity_seconds: old.activity_seconds,
          inference: old.inference,
          stream_error: old.stream_error,
          _thumbNonce: old._thumbNonce,
          _syncedAt: now,
          _displayActivity: old._displayActivity ?? old.activity_seconds ?? 0,
        };
      });
    });
    setSetupCamera((prev) => {
      if (!prev) return prev;
      const next = items.find((c) => c.id === prev.id);
      return next ? { ...prev, ...next } : prev;
    });
    setMsg(`共 ${items.length} 路摄像头 · 上次更新 ${new Date().toLocaleTimeString()}`);
    setMsgErr(false);
    return true;
  }, []);

  const loadAisles = useCallback(async () => {
    try {
      const ad = await apiGet('/api/aisles');
      if (ad.status === 'success' && Array.isArray(ad.items)) setAisles(ad.items);
    } catch {
      /* 巷道列表失败时仍展示单路卡片 */
    }
  }, []);

  const applyAislesFromMutation = useCallback((data) => {
    const card = aisleCardFromMutation(data);
    if (card) {
      const drop = new Set([card.aisle_id, data?.renamed_from].filter(Boolean));
      setAisles((prev) => [...prev.filter((a) => !drop.has(a.aisle_id)), card]);
      return;
    }
    if (data?.aisle_id && !data?.aisle) {
      setAisles((prev) => prev.filter((a) => a.aisle_id !== data.aisle_id));
    }
  }, []);

  const loadCameras = useCallback(async ({ probe = false } = {}) => {
    if (probe) setProbing(true);
    try {
      const qs = probe ? '' : '?probe=false';
      const [data] = await Promise.all([
        apiGet(`/api/cameras${qs}`),
        loadAisles(),
      ]);
      if (data.status !== 'success' || !Array.isArray(data.items)) {
        setMsg(formatUserError(data.error) || '加载失败');
        setMsgErr(true);
        return false;
      }
      applyCameraItems(data.items);
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
  }, [applyCameraItems, loadAisles]);

  const refreshListFastThenProbe = useCallback(async () => {
    await loadCameras({ probe: false });
    void loadCameras({ probe: true });
  }, [loadCameras]);

  const refreshCamerasAfterMutation = useCallback(
    async (mutationData) => {
      applyAislesFromMutation(mutationData);
      // 创建/保存返回的 items 是裸 camera_ips，没有缩略图和在线状态，不能整表替换。
      await refreshListFastThenProbe();
    },
    [applyAislesFromMutation, refreshListFastThenProbe],
  );

  useEffect(() => {
    if (!configHint) return undefined;
    const t = setTimeout(() => setConfigHint(''), 4000);
    return () => clearTimeout(t);
  }, [configHint]);

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
    setSetupAisle(null);
    setAisleForm(emptyAisleCreateForm());
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

  const openAisleSetup = async (aisle) => {
    const left = cameras.find((c) => c.id === aisle.camera_l);
    const right = cameras.find((c) => c.id === aisle.camera_r);
    setDrawerMode('aisle');
    setSetupAisle(aisle);
    setSetupCamera(null);
    setAisleForm({
      aisle_id: aisle.aisle_id,
      left: cameraToForm(left),
      right: cameraToForm(right),
    });
    setDrawerOpen(true);
    loadGlobalSettings();
    try {
      const [ld, rd] = await Promise.all([
        aisle.camera_l
          ? apiGet(`/api/cameras/${encodeURIComponent(aisle.camera_l)}`)
          : Promise.resolve(null),
        aisle.camera_r
          ? apiGet(`/api/cameras/${encodeURIComponent(aisle.camera_r)}`)
          : Promise.resolve(null),
      ]);
      setAisleForm((prev) => ({
        ...prev,
        left: cameraToForm(ld?.camera || left),
        right: cameraToForm(rd?.camera || right),
      }));
    } catch {
      /* 列表里的表单仍可用 */
    }
  };

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
    setSetupAisle(null);
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
    setForm((prev) => applyFormFields(prev, field, value));
  };

  const saveFromDrawer = async () => {
    if (drawerMode === 'create') {
      const aid = String(aisleForm.aisle_id || '').trim();
      if (!aid) {
        alert('请填写巷道编号');
        return;
      }
      setSaving(true);
      try {
        const data = await apiPost('/api/aisles', {
          aisle_id: aid,
          camera_l: formToCameraPayload(aisleForm.left),
          camera_r: formToCameraPayload(aisleForm.right),
        });
        if (data.error) {
          alert(formatUserError(data.error));
          return;
        }
        applyConfigHint(data);
        closeDrawer();
        await refreshCamerasAfterMutation(data);
      } catch (err) {
        alert(formatUserError(err.message) || '创建巷道失败');
      } finally {
        setSaving(false);
      }
      return;
    }
    if (drawerMode === 'aisle') {
      if (!setupAisle?.camera_l || !setupAisle?.camera_r) {
        alert('巷道未绑定左右路');
        return;
      }
      const nextAisleId = String(aisleForm.aisle_id || '').trim();
      if (!nextAisleId) {
        alert('请填写巷道编号');
        return;
      }
      setSaving(true);
      try {
        const data = await apiPut(
          `/api/aisles/${encodeURIComponent(setupAisle.aisle_id)}/cameras`,
          {
            aisle_id: String(aisleForm.aisle_id || '').trim(),
            camera_l: formToCameraPayload(aisleForm.left),
            camera_r: formToCameraPayload(aisleForm.right),
          },
        );
        if (data.error) {
          alert(formatUserError(data.error) || '保存失败');
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
      return;
    }
    const payload = formToCameraPayload(form);
    setSaving(true);
    try {
      const data = await apiPut(`/api/cameras/${encodeURIComponent(setupCamera.id)}`, payload);
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
    if (drawerMode === 'aisle') {
      if (!setupAisle || !confirmDeleteAisle(setupAisle.aisle_id)) return;
      setSaving(true);
      try {
        const data = await apiDelete(`/api/aisles/${encodeURIComponent(setupAisle.aisle_id)}`);
        if (data.error) {
          alert(formatUserError(data.error));
          return;
        }
        applyConfigHint(data);
        closeDrawer();
        await refreshCamerasAfterMutation(data);
      } catch (err) {
        alert(formatUserError(err.message) || '删除巷道失败');
      } finally {
        setSaving(false);
      }
      return;
    }
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
    setInferLoadingAction(turnOn ? 'start' : 'stop');
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
      setInferLoadingAction(null);
    }
  };

  const startAllInference = async () => {
    const targets = aisles.filter((a) => a.camera_l && a.camera_r);
    if (!targets.length) {
      alert('没有已成组巷道可启动。未编入巷道的摄像头不会开推理。');
      return;
    }
    if (!window.confirm(`确认启动全部 ${targets.length} 条巷道的智能检测？将按左右路成对启动。`)) {
      return;
    }
    setBatchInferAction('start');
    setMsg('正在按巷道启动智能检测…');
    setMsgErr(false);
    try {
      let started = 0;
      let skipped = 0;
      let failed = 0;
      const errors = [];
      const isLive = (st) => st === 'running' || st === 'starting';
      for (const aisle of targets) {
        const left = cameras.find((c) => c.id === aisle.camera_l);
        const right = cameras.find((c) => c.id === aisle.camera_r);
        if (isLive(left?.inference?.status) && isLive(right?.inference?.status)) {
          skipped += 1;
          continue;
        }
        const r = await startAisleInference(aisle.camera_l, aisle.camera_r);
        if (!r.ok) {
          failed += 1;
          if (r.error) errors.push(`${aisle.aisle_id}：${r.error}`);
        } else {
          started += 1;
        }
      }
      const detail = errors.length ? `。${errors[0]}` : '';
      setMsg(
        `按巷道启动完成：成功 ${started} 条，跳过 ${skipped} 条，失败 ${failed} 条${detail}`,
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
        alert(formatStreamError(data?.error) || formatUserError(data?.error) || '抓帧失败');
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
  const aisleSetupLeft = cameras.find((c) => c.id === setupAisle?.camera_l) || null;
  const aisleSetupRight = cameras.find((c) => c.id === setupAisle?.camera_r) || null;

  return (
    <div className="page dashboard-page">
        <h1 className="page-title">摄像头总览</h1>

        <div className="toolbar">
          <span className={`msg ${msgErr ? 'err' : ''}`}>
            {listLoading ? '加载列表…' : msg}
            {probing && !listLoading ? ' · 正在探测在线状态' : ''}
          </span>
          <div className="toolbar-actions">
            <button
              type="button"
              className="btn-batch"
              title="按巷道成对启动智能检测（未编入巷道的不启动）"
              disabled={batchInferBusy || listLoading || !aisleCards.length}
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
              title="添加巷道"
              aria-label="添加巷道"
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
            <div className="empty">暂无巷道，点击「添加巷道」同时配置左右路摄像头。</div>
          ) : (
            <>
              {aisleCards.map((aisle) => {
                const left = cameras.find((c) => c.id === aisle.camera_l);
                const right = cameras.find((c) => c.id === aisle.camera_r);
                const thumb = (left?.has_thumbnail && left) || (right?.has_thumbnail && right) || left || right;
                const on = aisleInferOn(left, right);
                const inferSt = aisleInferStatus(left, right);
                const inferText = inferLoadingId === aisle.aisle_id
                  ? (inferLoadingAction === 'stop'
                    ? AISLE_INFER_LABEL.stopped
                    : AISLE_INFER_LABEL.starting)
                  : (AISLE_INFER_LABEL[inferSt] || AISLE_INFER_LABEL.stopped);
                const online = Boolean(left?.online || right?.online);
                const activity = Math.max(Number(left?._displayActivity) || 0, Number(right?._displayActivity) || 0);
                const streamHint = aisleStreamHint(left, right);
                return (
                  <article
                    className={`card${on || inferSt === 'starting' ? ' is-inferring' : ''}`}
                    key={aisle.aisle_id}
                  >
                    <div
                      className="card-preview card-preview-link"
                      role="button"
                      tabIndex={0}
                      title={`进入 ${aisle.aisle_id}`}
                      onClick={() => navigate(`/live/${encodeURIComponent(aisle.aisle_id)}`)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          navigate(`/live/${encodeURIComponent(aisle.aisle_id)}`);
                        }
                      }}
                    >
                      {thumb?.has_thumbnail ? (
                        <img
                          src={thumbnailUrl(thumb.id, thumb.last_frame_at, thumb._thumbNonce)}
                          alt={aisle.aisle_id}
                        />
                      ) : (
                        <div className="card-preview-empty">{streamHint || '暂无画面'}</div>
                      )}
                      <div className="card-actions" onClick={(e) => e.stopPropagation()}>
                        <InferenceToggle
                          on={on}
                          loading={inferLoadingId === aisle.aisle_id}
                          disabled={inferToggleDisabled}
                          title={on ? '停止检测' : '开启检测'}
                          onToggle={(turnOn) => toggleAisleInference(aisle, turnOn)}
                        />
                        <button
                          type="button"
                          className="btn-icon"
                          title="抓帧"
                          disabled={!thumb || refreshingId === thumb.id}
                          onClick={() => thumb && captureFrame(thumb)}
                        >
                          ↻
                        </button>
                        <button
                          type="button"
                          className="btn-icon"
                          title="修改巷道"
                          onClick={() => openAisleSetup(aisle)}
                        >
                          ⚙
                        </button>
                      </div>
                      <div className="card-body">
                        <h2 className="card-title">{aisle.aisle_id}</h2>
                        <div className="card-status">
                          <span className={online ? 'st-online' : 'st-offline'}>
                            {online ? '在线' : '离线'}
                          </span>
                          <span className="card-status-sep">·</span>
                          <span className="card-activity">{formatDuration(activity)}</span>
                          <span className="card-status-sep">·</span>
                          <span className={`card-infer ${inferSt}`}>{inferText}</span>
                        </div>
                        <div className={`card-url${streamHint ? ' is-stream-err' : ''}`} title={streamHint || undefined}>
                          {streamHint
                            || `${left?.name || aisle.camera_l}${right ? ` · ${right.name || aisle.camera_r}` : ''}`}
                        </div>
                      </div>
                    </div>
                  </article>
                );
              })}
              {ungrouped.map((cam) => {
                const camHint = streamHintOf(cam);
                return (
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
                      <div className="card-preview-empty">{camHint || '暂无画面'}</div>
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
                      <div className={`card-url${camHint ? ' is-stream-err' : ''}`} title={camHint || cam.url}>
                        {camHint || '未编入巷道'}
                      </div>
                    </div>
                  </div>
                </article>
                );
              })}
            </>
          )}
        </div>

      <AisleCreateDrawer
        open={drawerOpen && (drawerMode === 'create' || drawerMode === 'aisle')}
        mode={drawerMode === 'aisle' ? 'aisle' : 'create'}
        form={aisleForm}
        onChange={setAisleForm}
        onClose={closeDrawer}
        onSave={saveFromDrawer}
        onDelete={deleteFromDrawer}
        saving={saving}
        camL={aisleSetupLeft}
        camR={aisleSetupRight}
        inferOn={aisleInferOn(aisleSetupLeft, aisleSetupRight)}
        inferLoading={inferLoadingId === setupAisle?.aisle_id}
        onToggleInference={
          setupAisle ? (turnOn) => toggleAisleInference(setupAisle, turnOn) : undefined
        }
        onCapture={(side) => {
          const cam = side === 'L' ? aisleSetupLeft : aisleSetupRight;
          if (cam) captureFrame(cam);
        }}
        capturingId={refreshingId}
      />
      <CameraSetupDrawer
        open={drawerOpen && drawerMode === 'edit'}
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
