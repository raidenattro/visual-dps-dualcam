import { useCallback, useRef, useState } from 'react';
import './GridPicker.css';

const MAX = 8;

/**
 * 抽屉内拖选 m×n 矩阵（类似 Excel 选区）
 */
export default function GridPicker({ rows = 4, cols = 4, max = MAX, disabled = false, onChange }) {
  const [dragAnchor, setDragAnchor] = useState(null);
  const selectingRef = useRef(false);

  const clamp = (n) => Math.max(1, Math.min(max, n));

  const emitSelection = useCallback(
    (r, c) => {
      const nextRows = clamp(r);
      const nextCols = clamp(c);
      if (nextRows !== rows || nextCols !== cols) {
        onChange?.(nextRows, nextCols);
      }
    },
    [rows, cols, onChange],
  );

  const cellFromEvent = (e) => {
    const cell = e.target.closest('[data-grid-cell]');
    if (!cell) return null;
    const r = Number(cell.dataset.row);
    const c = Number(cell.dataset.col);
    if (!Number.isFinite(r) || !Number.isFinite(c)) return null;
    return { r, c };
  };

  const onPointerDown = (e) => {
    if (disabled) return;
    const hit = cellFromEvent(e);
    if (!hit) return;
    e.preventDefault();
    selectingRef.current = true;
    setDragAnchor(hit);
    emitSelection(hit.r, hit.c);
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };

  const onPointerMove = (e) => {
    if (!selectingRef.current || !dragAnchor) return;
    const hit = cellFromEvent(e);
    if (!hit) return;
    const r = Math.max(dragAnchor.r, hit.r);
    const c = Math.max(dragAnchor.c, hit.c);
    emitSelection(r, c);
  };

  const endSelect = (e) => {
    if (!selectingRef.current) return;
    selectingRef.current = false;
    setDragAnchor(null);
    try {
      e?.currentTarget?.releasePointerCapture?.(e.pointerId);
    } catch {
      /* ignore */
    }
  };

  const previewRows = dragAnchor ? Math.max(rows, dragAnchor.r) : rows;
  const previewCols = dragAnchor ? Math.max(cols, dragAnchor.c) : cols;

  return (
    <div className="grid-picker">
      <div
        className={`grid-picker-matrix${disabled ? ' is-disabled' : ''}`}
        style={{ gridTemplateColumns: `repeat(${max}, 1fr)` }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endSelect}
        onPointerCancel={endSelect}
        role="grid"
        aria-label={`货位矩阵选择，当前 ${rows} 行 ${cols} 列`}
      >
        {Array.from({ length: max }, (_, rowIdx) =>
          Array.from({ length: max }, (_, colIdx) => {
            const r = rowIdx + 1;
            const c = colIdx + 1;
            const selected = r <= previewRows && c <= previewCols;
            return (
              <div
                key={`${r}-${c}`}
                data-grid-cell
                data-row={r}
                data-col={c}
                className={`grid-picker-cell${selected ? ' is-selected' : ''}`}
                role="gridcell"
                aria-selected={selected}
              />
            );
          }),
        )}
      </div>
    </div>
  );
}
