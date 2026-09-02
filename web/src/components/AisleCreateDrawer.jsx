import { Link } from 'react-router-dom';
import CameraStreamFields from './CameraStreamFields.jsx';
import InferenceToggle from './InferenceToggle';
import { applyAisleIdToCreateForm, applyFormFields } from '../lib/cameraStreamForm';
import { formatDuration, thumbnailUrl } from '../api/client';
import { formatStreamError } from '../lib/userFacingText';
import './CameraSetupDrawer.css';

function SidePreview({ cam, onCapture, capturing, role }) {
  const err = formatStreamError(!cam?.online ? cam?.stream_error : '');
  return (
    <div className="aisle-side-block">
      <div className="drawer-preview aisle-side-preview">
        <div className="drawer-preview-actions">
          <button
            type="button"
            className="drawer-preview-capture"
            title="抓取预览帧"
            disabled={capturing || !cam}
            onClick={onCapture}
          >
            ↻
          </button>
        </div>
        {cam?.has_thumbnail ? (
          <img src={thumbnailUrl(cam.id, cam.last_frame_at, cam._thumbNonce)} alt="" />
        ) : (
          <div className="drawer-preview-empty">{err || '暂无预览'}</div>
        )}
        {err && cam?.has_thumbnail ? (
          <div className="drawer-stream-hint" title={err}>{err}</div>
        ) : null}
      </div>
      <div className={`aisle-side-status${err ? ' is-err' : ''}`} title={err || undefined}>
        {role} · {cam?.online ? '在线' : (err || '离线')}
      </div>
    </div>
  );
}

/** 总览入口：添加 / 编辑巷道（左右路一起，不是单路 cam 配置） */
export default function AisleCreateDrawer({
  open,
  mode = 'create',
  form,
  onChange,
  onClose,
  onSave,
  onDelete,
  saving,
  camL,
  camR,
  inferOn = false,
  inferLoading = false,
  onToggleInference,
  onCapture,
  capturingId,
}) {
  if (!open) return null;

  const isEdit = mode === 'aisle';
  const setAisleId = (value) => {
    onChange((prev) => applyAisleIdToCreateForm(prev || form, value));
  };
  const setSide = (side, field, value) => {
    onChange((prev) => {
      const cur = prev || form;
      return { ...cur, [side]: applyFormFields(cur[side] || {}, field, value) };
    });
  };
  const online = Boolean(camL?.online || camR?.online);
  const activity = Math.max(
    Number(camL?._displayActivity ?? camL?.activity_seconds) || 0,
    Number(camR?._displayActivity ?? camR?.activity_seconds) || 0,
  );

  return (
    <div className="drawer-root" role="presentation">
      <button type="button" className="drawer-backdrop" aria-label="关闭" onClick={onClose} />
      <aside className="drawer-panel aisle-create-panel" role="dialog" aria-labelledby="aisle-create-title">
        <header className="drawer-header">
          <div>
            <h2 id="aisle-create-title">{isEdit ? '修改巷道' : '添加巷道'}</h2>
            <p className="drawer-subtitle">
              {isEdit
                ? '巷道号、通道号、名称、流类型、视频流地址均可改。'
                : '左右路各配一路视频流，标注时只选这条巷道。'}
            </p>
          </div>
          <button type="button" className="drawer-close" onClick={onClose} aria-label="关闭">
            ×
          </button>
        </header>
        <div className="drawer-body">
          {isEdit ? (
            <section className="drawer-section aisle-status-block">
              <div className="aisle-preview-grid">
                <SidePreview
                  cam={camL}
                  role="左路"
                  capturing={capturingId === camL?.id}
                  onCapture={() => onCapture?.('L')}
                />
                <SidePreview
                  cam={camR}
                  role="右路"
                  capturing={capturingId === camR?.id}
                  onCapture={() => onCapture?.('R')}
                />
              </div>
              <div className="detail-row">
                <span className="detail-label">在线</span>
                <span className="detail-value">{online ? '在线' : '离线'}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">本次在线</span>
                <span className="detail-value">{formatDuration(activity)}</span>
              </div>
              <div className="detail-row detail-row--switch">
                <span className="detail-label">智能检测</span>
                <div className="detail-row-control">
                  <InferenceToggle
                    on={inferOn}
                    loading={inferLoading}
                    disabled={inferLoading || !onToggleInference}
                    title={inferOn ? '停止本巷道检测' : '启动本巷道检测'}
                    onToggle={onToggleInference}
                  />
                </div>
              </div>
            </section>
          ) : null}
          <form
            id="aisle-create-form"
            className="drawer-form"
            onSubmit={(e) => {
              e.preventDefault();
              onSave();
            }}
          >
            <section className="drawer-section aisle-id-block">
              <h3>巷道</h3>
              <label>
                巷道编号
                <input
                  value={form.aisle_id}
                  onChange={(e) => setAisleId(e.target.value)}
                  placeholder="如 aisle-1"
                  required
                />
              </label>
              {!isEdit ? (
                <p className="drawer-field-hint">
                  通道号默认 {form.aisle_id || '编号'}-L / {form.aisle_id || '编号'}-R，可改。
                </p>
              ) : (
                <p className="drawer-field-hint">
                  巷道号只是名称，可改。检测正在跑时改名会先停止该巷道检测。
                </p>
              )}
            </section>
            <div className="aisle-cam-grid">
              <section className="drawer-section aisle-cam-block">
                <h3>左路 {camL?.name ? `· ${camL.name}` : ''}</h3>
                <CameraStreamFields
                  dense
                  showEnabled
                  form={form.left}
                  onChange={(field, value) => setSide('left', field, value)}
                />
              </section>
              <section className="drawer-section aisle-cam-block">
                <h3>右路 {camR?.name ? `· ${camR.name}` : ''}</h3>
                <CameraStreamFields
                  dense
                  showEnabled
                  form={form.right}
                  onChange={(field, value) => setSide('right', field, value)}
                />
              </section>
            </div>
          </form>
          {isEdit && form.aisle_id ? (
            <p className="drawer-field-hint aisle-setup-links">
              <Link to={`/live/${encodeURIComponent(form.aisle_id)}`} onClick={onClose}>
                进入直播
              </Link>
              {' · '}
              <Link to="/aisle" onClick={onClose}>
                巷道标注
              </Link>
            </p>
          ) : null}
        </div>
        <footer className="drawer-footer">
          {isEdit ? (
            <button type="button" className="btn-danger" disabled={saving} onClick={onDelete}>
              删除巷道
            </button>
          ) : null}
          <div className="drawer-footer-right">
            <button type="button" className="secondary" onClick={onClose}>
              取消
            </button>
            <button type="submit" form="aisle-create-form" disabled={saving}>
              {saving ? (isEdit ? '保存中…' : '创建中…') : isEdit ? '保存修改' : '创建巷道'}
            </button>
          </div>
        </footer>
      </aside>
    </div>
  );
}
