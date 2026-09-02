import InferenceToggle from './InferenceToggle';
import {
  CAMERA_SOURCE_TYPES,
  DEFAULT_SOURCE_TYPE,
  defaultPlaybackUrl,
  isAutoPlaybackUrl,
  sourceTypeFormPatch,
} from '../lib/cameraStreamForm';

/** 单路视频流字段：通道号、名称、上游 RTSP（摄像头 IP 入口）、本机播放地址 */
export default function CameraStreamFields({
  form,
  onChange,
  pathLocked = false,
  showEnabled = true,
  dense = false,
}) {
  const sourceType = form.source_type || DEFAULT_SOURCE_TYPE;
  const typeHint = CAMERA_SOURCE_TYPES.find((t) => t.value === sourceType)?.hint;

  return (
    <>
      <label>
        通道编号
        <input
          value={form.path}
          onChange={(e) => {
            const path = e.target.value;
            if (sourceType !== 'external' && isAutoPlaybackUrl(form.url, form.path)) {
              onChange({ path, url: defaultPlaybackUrl(path) });
              return;
            }
            onChange('path', path);
          }}
          placeholder="如 aisle-1-L"
          disabled={pathLocked}
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
          title={typeHint}
          value={sourceType}
          onChange={(e) => onChange(sourceTypeFormPatch(form, e.target.value))}
        >
          {CAMERA_SOURCE_TYPES.map((t) => (
            <option key={t.value} value={t.value}>
              {t.label}
            </option>
          ))}
        </select>
      </label>
      {!dense && typeHint ? <p className="drawer-field-hint">{typeHint}</p> : null}
      {sourceType === 'rtsp_pull' ? (
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
              placeholder="输入通道编号后自动生成"
            />
          </label>
        </>
      ) : null}
      {sourceType === 'publisher' ? (
        <label>
          本机播放地址
          <input
            value={form.url || ''}
            onChange={(e) => onChange('url', e.target.value)}
            placeholder="输入通道编号后自动生成"
          />
        </label>
      ) : null}
      {sourceType === 'external' ? (
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
      {showEnabled ? (
        <div className="detail-row detail-row--switch drawer-form-enabled-row">
          <span className="detail-label">启用该路</span>
          <div className="detail-row-control">
            <InferenceToggle
              on={form.enabled}
              title={form.enabled ? '关闭该路摄像头' : '启用该路摄像头'}
              onToggle={(turnOn) => onChange('enabled', turnOn)}
            />
          </div>
        </div>
      ) : null}
    </>
  );
}
