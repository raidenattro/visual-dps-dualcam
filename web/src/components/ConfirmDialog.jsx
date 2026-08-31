import { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import './ConfirmDialog.css';

export default function ConfirmDialog({
  open,
  title,
  message,
  step = 1,
  totalSteps = 2,
  cancelLabel = '取消',
  continueLabel = '继续',
  confirmLabel = '确认删除',
  onCancel,
  onContinue,
  onConfirm,
  loading = false,
}) {
  const openedAtRef = useRef(0);

  useEffect(() => {
    if (open) openedAtRef.current = Date.now();
  }, [open]);

  if (!open) return null;

  const isLastStep = step >= totalSteps;

  const handleBackdropClick = () => {
    if (Date.now() - openedAtRef.current < 200) return;
    onCancel();
  };

  return createPortal(
    <div className="confirm-dialog-root" role="presentation">
      <button type="button" className="confirm-dialog-backdrop" aria-label="关闭" onClick={handleBackdropClick} />
      <div
        className="confirm-dialog-panel"
        role="alertdialog"
        aria-labelledby="confirm-dialog-title"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 id="confirm-dialog-title">{title}</h3>
        <p className="confirm-dialog-message">{message}</p>
        {totalSteps > 1 ? (
          <p className="confirm-dialog-step">
            第 {step} / {totalSteps} 步
          </p>
        ) : null}
        <div className="confirm-dialog-actions">
          <button type="button" className="settings-btn-secondary" disabled={loading} onClick={onCancel}>
            {cancelLabel}
          </button>
          {isLastStep ? (
            <button type="button" className="confirm-dialog-danger" disabled={loading} onClick={onConfirm}>
              {loading ? '删除中…' : confirmLabel}
            </button>
          ) : (
            <button type="button" className="settings-btn-primary" disabled={loading} onClick={onContinue}>
              {continueLabel}
            </button>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
