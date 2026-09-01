import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import InferenceToggle from './InferenceToggle';
import { InferenceModelOverrideCard } from './InferenceModelFields';
import {
  CAMERA_OVERRIDE_FIELDS,
  DEFAULT_RTM_DET,
  formatSettingDisplayValue,
  settingsFieldTooltip,
} from '../lib/cameraSettings';
import FieldHint from './FieldHint';
import {
  CAMERA_SOURCE_TYPES,
  DEFAULT_SOURCE_TYPE,
  defaultPlaybackUrl,
  sourceTypeLabel,
} from '../lib/cameraStreamForm';
import { apiGet, apiPut, formatDuration, thumbnailUrl } from '../api/client';
import './CameraSetupDrawer.css';

function DetailRow({ label, value, mono, title }) {
  return (
    <div className="detail-row" title={title}>
      <span className="detail-label">{label}</span>
      <span className={`detail-value ${mono ? 'mono' : ''}`}>{value ?? '—'}</span>
    </div>
  );
}

export default function CameraSetupDrawer({
  open,
  mode,
  camera,
  form,
  globalDefaults = {},
  effectiveSettings = {},
  onChange,
  onClose,
  onSave,
  onDelete,
  onCapture,
  onToggleInference,
  saving,
  actionLoading,
}) {
  if (!open) return null;

  const isCreate = mode === 'create';
  const infer = camera?.inference;
  const inferStatus = infer?.status || 'stopped';
  const settings = form.settings || {};
  const inferOn = inferStatus === 'running' || inferStatus === 'starting';

  const setSettings = (next) => onChange('settings', next);

  const toggleCustom = (key, enabled) => {
    if (enabled) {
      const fallback = globalDefaults[key];
      setSettings({
        ...settings,
        [key]:
          settings[key] ??
          fallback ??
          (key === 'debug-info.enabled'
            ? false
            : key === 'pipeline_log.enabled'
              ? false
              : key === 'models.backend'
              ? 'rtmpose_t'
              : ''),
      });
    } else {
      const next = { ...settings };
      delete next[key];
      setSettings(next);
    }
  };

  const setOverrideValue = (key, raw) => {
    const field = CAMERA_OVERRIDE_FIELDS.find((f) => f.key === key);
    let val = raw;
    if (field?.type === 'number') val = Number(raw);
    if (field?.type === 'boolean') val = Boolean(raw);
    if (field?.type === 'select') val = String(raw);
    setSettings({ ...settings, [key]: val });
  };

  return (
    <div className="drawer-root" role="presentation">
      <button type="button" className="drawer-backdrop" aria-label="关闭" onClick={onClose} />
      <aside className="drawer-panel" role="dialog" aria-labelledby="drawer-title">
        <header className="drawer-header">
          <div>
            <h2 id="drawer-title">{isCreate ? '添加摄像头' : '摄像头设置'}</h2>
            {!isCreate && <p className="drawer-subtitle">{camera?.name}</p>}
          </div>
          <button type="button" className="drawer-close" onClick={onClose} aria-label="关闭">
            ×
          </button>
        </header>

        <div className="drawer-body">
          {!isCreate && camera && (
            <>
              <section className="drawer-section">
                <h3>实时状态</h3>
                <div className="drawer-preview">
                  <div className="drawer-preview-actions">
                    <button
                      type="button"
                      className="drawer-preview-capture"
                      title="抓取预览帧"
                      disabled={actionLoading}
                      onClick={onCapture}
                    >
                      ↻
                    </button>
                  </div>
                  {camera.has_thumbnail ? (
                    <img src={thumbnailUrl(camera.id, camera.last_frame_at)} alt="" />
                  ) : (
                    <div className="drawer-preview-empty">暂无预览</div>
                  )}
                </div>
                <DetailRow
                  label="在线"
                  value={camera.online ? '在线' : '离线'}
                />
                <DetailRow
                  label="本次在线"
                  value={formatDuration(camera._displayActivity ?? camera.activity_seconds)}
                  title="自本次检测到在线起累计，离线后清零（非历史总时长）"
                />
                <div className="detail-row detail-row--switch">
                  <span className="detail-label">智能检测</span>
                  <div className="detail-row-control">
                    <InferenceToggle
                      on={inferOn}
                      loading={actionLoading}
                      disabled={actionLoading}
                      title={inferOn ? '关闭智能检测' : '开启智能检测'}
                      onToggle={onToggleInference}
                    />
                  </div>
                </div>
                <div className="detail-row detail-row--switch">
                  <span className="detail-label">启用该路摄像头</span>
                  <div className="detail-row-control">
                    <InferenceToggle
                      on={form.enabled}
                      title={form.enabled ? '关闭该路摄像头' : '启用该路摄像头'}
                      onToggle={(turnOn) => onChange('enabled', turnOn)}
                    />
                  </div>
                </div>
              </section>
            </>
          )}

          <section className="drawer-section">
            <h3>视频流设置</h3>
            <form
              id="camera-setup-form"
              className="drawer-form"
              onSubmit={(e) => {
                e.preventDefault();
                onSave();
              }}
            >
              <label>
                通道编号
                <input
                  value={form.path}
                  onChange={(e) => onChange('path', e.target.value)}
                  placeholder="如 cam、warehouse_1"
                  disabled={!isCreate}
                  required
                />
              </label>
              <label>
                显示名称
                <input
                  value={form.name}
                  onChange={(e) => onChange('name', e.target.value)}
                  required
                />
              </label>
              <label>
                流类型
                <select
                  value={form.source_type || DEFAULT_SOURCE_TYPE}
                  onChange={(e) => {
                    const next = e.target.value;
                    onChange('source_type', next);
                    if (next === 'publisher') {
                      const play = defaultPlaybackUrl(form.path);
                      if (play && !form.url) onChange('url', play);
                    }
                    if (next === 'external') {
                      onChange('pull_url', '');
                    }
                  }}
                >
                  {CAMERA_SOURCE_TYPES.map((t) => (
                    <option key={t.value} value={t.value}>
                      {t.label}
                    </option>
                  ))}
                </select>
              </label>
              <p className="drawer-field-hint">
                {CAMERA_SOURCE_TYPES.find((t) => t.value === (form.source_type || DEFAULT_SOURCE_TYPE))?.hint}
              </p>
              {form.source_type === 'rtsp_pull' ? (
                <>
                  <label>
                    上游拉流地址
                    <input
                      value={form.pull_url || ''}
                      onChange={(e) => onChange('pull_url', e.target.value)}
                      placeholder="rtsp://192.168.1.100:554/stream1"
                      required
                    />
                  </label>
                  <label>
                    本机播放地址
                    <input
                      value={form.url || ''}
                      onChange={(e) => onChange('url', e.target.value)}
                      placeholder={defaultPlaybackUrl(form.path) || 'rtsp://127.0.0.1:8554/cam1'}
                    />
                  </label>
                  <p className="drawer-field-hint">
                    推理/监控使用「本机播放地址」（通常为本机 MediaMTX）。留空则按通道编号自动生成。
                  </p>
                </>
              ) : null}
              {form.source_type === 'publisher' ? (
                <label>
                  本机播放地址
                  <input
                    value={form.url || defaultPlaybackUrl(form.path)}
                    onChange={(e) => onChange('url', e.target.value)}
                    placeholder={defaultPlaybackUrl(form.path) || 'rtsp://127.0.0.1:8554/cam1'}
                  />
                </label>
              ) : null}
              {form.source_type === 'external' ? (
                <label>
                  视频流地址 (RTSP)
                  <input
                    value={form.url || ''}
                    onChange={(e) => onChange('url', e.target.value)}
                    placeholder="rtsp://192.168.1.10:554/live"
                    required
                  />
                </label>
              ) : null}
              {!isCreate && form.source_type ? (
                <DetailRow label="当前流类型" value={sourceTypeLabel(form.source_type)} />
              ) : null}
              {isCreate ? (
                <div className="detail-row detail-row--switch drawer-form-enabled-row">
                  <span className="detail-label">启用该路摄像头</span>
                  <div className="detail-row-control">
                    <InferenceToggle
                      on={form.enabled}
                      title={form.enabled ? '关闭该路摄像头' : '启用该路摄像头'}
                      onToggle={(turnOn) => onChange('enabled', turnOn)}
                    />
                  </div>
                </div>
              ) : null}
            </form>
          </section>

          {!isCreate && camera?.id ? <DualcamGeomSection cameraId={camera.id} /> : null}

          {!isCreate && (
            <section className="drawer-section drawer-section-inference">
              <div className="drawer-section-head">
                <h3>检测参数</h3>
                <p className="drawer-section-desc">
                  未开启「自定义」时沿用全局默认；保存后请重新启动该路智能检测。
                </p>
              </div>
              <div className="drawer-settings-grid">
                <InferenceModelOverrideCard
                  settings={settings}
                  globalDefaults={globalDefaults}
                  effectiveSettings={effectiveSettings}
                  onEnableCustom={() => {
                    setSettings({
                      ...settings,
                      'models.backend':
                        settings['models.backend'] ??
                        effectiveSettings['models.backend'] ??
                        globalDefaults['models.backend'] ??
                        'rtmpose_t',
                      'models.det':
                        settings['models.det'] ??
                        effectiveSettings['models.det'] ??
                        globalDefaults['models.det'] ??
                        DEFAULT_RTM_DET,
                    });
                  }}
                  onDisableCustom={() => {
                    const next = { ...settings };
                    delete next['models.backend'];
                    delete next['models.det'];
                    setSettings(next);
                  }}
                  onSetValue={(key, raw) => {
                    setSettings({ ...settings, [key]: String(raw) });
                  }}
                />
                {CAMERA_OVERRIDE_FIELDS.map((field) => {
                  const customized = Object.prototype.hasOwnProperty.call(settings, field.key);
                  const globalVal = globalDefaults[field.key];
                  const displayGlobal = formatSettingDisplayValue(field, globalVal);
                  const currentVal = customized
                    ? (settings[field.key] ?? effectiveSettings[field.key] ?? globalVal)
                    : globalVal;
                  const isWide = field.type === 'select' || Boolean(field.hint);
                  return (
                    <div
                      key={field.key}
                      className={`drawer-param-card${customized ? ' is-custom' : ''}${isWide ? ' drawer-param-card--wide' : ''}`}
                    >
                      <div className="drawer-param-top">
                        <span className="drawer-param-label">
                          {field.label}
                          <FieldHint text={settingsFieldTooltip(field)} />
                        </span>
                        <label className="drawer-param-custom">
                          <input
                            type="checkbox"
                            checked={customized}
                            onChange={(e) => toggleCustom(field.key, e.target.checked)}
                          />
                          <span className="drawer-param-custom-track" aria-hidden />
                          <span className="drawer-param-custom-text">自定义</span>
                        </label>
                      </div>
                      <div className="drawer-param-control">
                        {field.type === 'boolean' ? (
                          <label className="drawer-param-bool">
                            <input
                              type="checkbox"
                              className="drawer-param-bool-input"
                              disabled={!customized}
                              checked={Boolean(currentVal)}
                              onChange={(e) => setOverrideValue(field.key, e.target.checked)}
                            />
                            <span className="drawer-param-bool-track" aria-hidden />
                            <span className="drawer-param-bool-text">
                              {Boolean(currentVal) ? '开启' : '关闭'}
                            </span>
                          </label>
                        ) : field.type === 'select' ? (
                          <select
                            className="drawer-param-input"
                            disabled={!customized}
                            value={String(currentVal ?? field.options[0]?.value ?? '')}
                            onChange={(e) => setOverrideValue(field.key, e.target.value)}
                          >
                            {field.options.map((opt) => (
                              <option key={opt.value} value={opt.value}>
                                {opt.shortLabel || opt.label}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="number"
                            className="drawer-param-input"
                            disabled={!customized}
                            min={field.min}
                            max={field.max}
                            value={currentVal ?? ''}
                            onChange={(e) => setOverrideValue(field.key, e.target.value)}
                          />
                        )}
                      </div>
                      <div className="drawer-param-foot">
                        <span className="drawer-param-default">
                          全局默认 <strong>{displayGlobal}</strong>
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}
        </div>

        <footer className="drawer-footer">
          {!isCreate && (
            <button type="button" className="btn-danger" disabled={saving} onClick={onDelete}>
              删除
            </button>
          )}
          <div className="drawer-footer-right">
            <button type="button" className="secondary" onClick={onClose}>
              取消
            </button>
            <button type="submit" form="camera-setup-form" disabled={saving}>
              {saving ? '保存中…' : isCreate ? '创建' : '保存'}
            </button>
          </div>
        </footer>
      </aside>
    </div>
  );
}

function DualcamGeomSection({ cameraId }) {
  const [info, setInfo] = useState(null);
  const [msg, setMsg] = useState('');
  const [savingGeom, setSavingGeom] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setInfo(null);
    setMsg('');
    if (!cameraId) return undefined;
    apiGet(`/api/aisles/by-camera/${encodeURIComponent(cameraId)}`)
      .then((d) => {
        if (!cancelled && d?.status === 'success') setInfo(d);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [cameraId]);

  if (!info) {
    return (
      <section className="drawer-section">
        <div className="drawer-section-head">
          <h3>双路 3D 几何</h3>
          <p className="drawer-section-desc">
            该路尚未编入巷道。请先到 <Link to="/aisle">巷道标注</Link> 绑定左右路，否则禁止开推理。
          </p>
        </div>
      </section>
    );
  }

  const aisle = info.aisle || {};
  const role = info.role;
  const view = aisle.views?.[role] || {};
  const size = view.image_size || [1280, 720];
  const prior = view.prior || aisle.prior || {};

  const patchView = (patch) => {
    setInfo((prev) => {
      const nextAisle = { ...prev.aisle, views: { ...prev.aisle.views } };
      nextAisle.views[role] = { ...(nextAisle.views[role] || {}), ...patch };
      return { ...prev, aisle: nextAisle };
    });
  };

  const save = async () => {
    setSavingGeom(true);
    setMsg('');
    try {
      const d = await apiPut(`/api/aisles/${encodeURIComponent(aisle.aisle_id)}`, aisle);
      if (d.status !== 'success') {
        setMsg(d.error || '保存失败');
        return;
      }
      setInfo((prev) => ({ ...prev, aisle: d.aisle }));
      setMsg('几何已保存，请重新反解后再开检测');
    } catch (e) {
      setMsg(e.message || '保存失败');
    } finally {
      setSavingGeom(false);
    }
  };

  return (
    <section className="drawer-section">
      <div className="drawer-section-head">
        <h3>双路 3D 几何</h3>
        <p className="drawer-section-desc">
          巷道 {aisle.aisle_id} · 本路 {role === 'L' ? '左路' : '右路'}。宽高是标定像素。
          {' '}
          <Link to="/aisle">打开巷道标注</Link>
        </p>
      </div>
      <div className="drawer-form drawer-settings-grid">
        <label>
          标定宽度 (px)
          <input
            type="number"
            min={320}
            value={size[0]}
            onChange={(e) => patchView({ image_size: [Number(e.target.value), Number(size[1])] })}
          />
        </label>
        <label>
          标定高度 (px)
          <input
            type="number"
            min={180}
            value={size[1]}
            onChange={(e) => patchView({ image_size: [Number(size[0]), Number(e.target.value)] })}
          />
        </label>
        <label>
          相机离地 (m)
          <input
            type="number"
            step="0.01"
            value={prior.camH ?? ''}
            onChange={(e) => patchView({ prior: { ...prior, camH: Number(e.target.value) } })}
          />
        </label>
        <label>
          距巷道 (m)
          <input
            type="number"
            step="0.01"
            value={prior.camDist ?? ''}
            onChange={(e) => patchView({ prior: { ...prior, camDist: Number(e.target.value) } })}
          />
        </label>
        <div className="drawer-geom-actions">
          <button type="button" className="drawer-btn drawer-btn-primary" disabled={savingGeom} onClick={save}>
            {savingGeom ? '保存中…' : '保存本路几何'}
          </button>
        </div>
      </div>
      {msg ? <p className="drawer-field-hint">{msg}</p> : null}
    </section>
  );
}
