import './ShelfBar.css';

function shelfTabLabel(shelf) {
  const name = String(shelf?.shelf_name || '').trim();
  const code = String(shelf?.shelf_code || '').trim();
  return name || code || '未命名';
}

/** 预览区下方的横向货架 Tab 列表 */
export default function ShelfBar({
  shelves = [],
  selectedCode = '',
  draftLabel = '新货架',
  showDraft = false,
  readOnly = false,
  onSelect,
  onCreate,
  onDelete,
}) {
  if (readOnly && !shelves.length) return null;

  return (
    <div
      className={`monitor-shelf-bar${readOnly ? ' is-readonly' : ''}`}
      role={readOnly ? 'list' : 'tablist'}
      aria-label="货架列表"
      aria-readonly={readOnly || undefined}
    >
      {shelves.map((shelf) => {
        const code = String(shelf.shelf_code || '').trim();
        if (!code) return null;
        const active = !readOnly && code === selectedCode;
        const label = shelfTabLabel(shelf);
        return (
          <div key={code} className={`monitor-shelf-tab-wrap${active ? ' active' : ''}`}>
            {readOnly ? (
              <span className="monitor-shelf-tab monitor-shelf-tab--readonly" title={code}>
                {label}
              </span>
            ) : (
              <button
                type="button"
                role="tab"
                aria-selected={active}
                className={`monitor-shelf-tab${active ? ' active' : ''}`}
                title={code}
                onClick={() => onSelect?.(code)}
              >
                {label}
              </button>
            )}
            {active && onDelete ? (
              <button
                type="button"
                className="monitor-shelf-tab-del"
                aria-label={`删除货架 ${label}`}
                title="删除货架"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(code);
                }}
              >
                ×
              </button>
            ) : null}
          </div>
        );
      })}
      {!readOnly && showDraft ? (
        <button type="button" role="tab" aria-selected className="monitor-shelf-tab active">
          {draftLabel}
        </button>
      ) : null}
      {!readOnly ? (
        <button type="button" className="monitor-shelf-tab monitor-shelf-tab-add" onClick={onCreate}>
          + 新建货架
        </button>
      ) : null}
    </div>
  );
}
