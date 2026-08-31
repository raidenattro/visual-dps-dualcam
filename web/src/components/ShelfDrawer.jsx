import './CameraSetupDrawer.css';

const emptyForm = () => ({
  shelf_id: '',
  shelf_code: '',
  shelf_name: '',
});

export { emptyForm as emptyShelfForm };

/** 新建/编辑货架基本信息 */
export default function ShelfDrawer({ open, mode = 'create', form, onChange, onClose, onConfirm, saving }) {
  if (!open) return null;

  const isCreate = mode === 'create';
  const title = isCreate ? '新建货架' : '编辑货架';

  return (
    <div className="drawer-root" role="presentation">
      <button type="button" className="drawer-backdrop" aria-label="关闭" onClick={onClose} />
      <aside className="drawer-panel" role="dialog" aria-labelledby="shelf-drawer-title">
        <header className="drawer-header">
          <div>
            <h2 id="shelf-drawer-title">{title}</h2>
            <p className="drawer-subtitle">填写后可在画面上标注区域四角与货位网格</p>
          </div>
          <button type="button" className="drawer-close" onClick={onClose} aria-label="关闭">
            ×
          </button>
        </header>

        <div className="drawer-body">
          <form
            id="shelf-form"
            className="drawer-form"
            onSubmit={(e) => {
              e.preventDefault();
              onConfirm();
            }}
          >
            <label>
              货架 ID
              <input
                value={form.shelf_id}
                onChange={(e) => onChange('shelf_id', e.target.value)}
                placeholder="可选，对接外部系统"
                autoComplete="off"
              />
            </label>
            <label>
              编码
              <input
                value={form.shelf_code}
                onChange={(e) => onChange('shelf_code', e.target.value)}
                placeholder="必填，本摄像头内唯一"
                required
                autoComplete="off"
                readOnly={!isCreate}
              />
            </label>
            <label>
              名称
              <input
                value={form.shelf_name}
                onChange={(e) => onChange('shelf_name', e.target.value)}
                placeholder="可选，用于展示"
                autoComplete="off"
              />
            </label>
            <p className="drawer-hint">编码保存后不可修改；名称仅用于 Tab 展示，保存标注时以编码为准。</p>
          </form>
        </div>

        <footer className="drawer-footer">
          <button type="button" className="secondary" onClick={onClose} disabled={saving}>
            取消
          </button>
          <div className="drawer-footer-right">
            <button type="submit" form="shelf-form" disabled={saving}>
              {isCreate ? '创建并开始标注' : '保存'}
            </button>
          </div>
        </footer>
      </aside>
    </div>
  );
}
