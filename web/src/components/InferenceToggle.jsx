import './InferenceToggle.css';

/** 智能检测启停开关；可选 label 与状态样式类，用于开关旁显示当前状态文案 */
export default function InferenceToggle({
  on,
  loading,
  disabled,
  onToggle,
  title = '智能检测',
  label = null,
  statusClass = '',
}) {
  const isDisabled = disabled || loading;
  return (
    <label
      className={`infer-toggle${label ? ' infer-toggle--labeled' : ''}${on ? ' on' : ''}${loading ? ' loading' : ''}${statusClass ? ` ${statusClass}` : ''}`}
      title={title}
    >
      <input
        type="checkbox"
        checked={on}
        disabled={isDisabled}
        onChange={() => onToggle(!on)}
        aria-label={label || title}
      />
      <span className="infer-toggle-track" aria-hidden="true" />
      {label ? <span className="infer-toggle-label">{label}</span> : null}
    </label>
  );
}
