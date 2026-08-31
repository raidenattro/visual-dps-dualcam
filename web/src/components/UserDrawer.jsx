import './CameraSetupDrawer.css';

function DetailRow({ label, value, mono }) {
  return (
    <div className="detail-row">
      <span className="detail-label">{label}</span>
      <span className={`detail-value ${mono ? 'mono' : ''}`}>{value ?? '—'}</span>
    </div>
  );
}

function roleLabel(role) {
  return role === 'admin' ? '管理员' : '操作员';
}

export default function UserDrawer({
  open,
  mode,
  user,
  form,
  onChange,
  onClose,
  onSave,
  onSwitchToEdit,
  saving,
  isSelf,
}) {
  if (!open) return null;

  const isCreate = mode === 'create';
  const isView = mode === 'view';
  const title = isCreate ? '新建用户' : isView ? '用户详情' : '编辑用户';

  return (
    <div className="drawer-root" role="presentation">
      <button type="button" className="drawer-backdrop" aria-label="关闭" onClick={onClose} />
      <aside className="drawer-panel" role="dialog" aria-labelledby="user-drawer-title">
        <header className="drawer-header">
          <div>
            <h2 id="user-drawer-title">{title}</h2>
            {!isCreate && user ? <p className="drawer-subtitle">{user.username}</p> : null}
          </div>
          <button type="button" className="drawer-close" onClick={onClose} aria-label="关闭">
            ×
          </button>
        </header>

        <div className="drawer-body">
          {isView && user ? (
            <section className="drawer-section">
              <DetailRow label="用户名" value={user.username} mono />
              <DetailRow label="显示名" value={user.display_name || user.username} />
              <DetailRow label="角色" value={roleLabel(user.role)} />
            </section>
          ) : (
            <section className="drawer-section">
              <form
                id="user-form"
                className="drawer-form"
                onSubmit={(e) => {
                  e.preventDefault();
                  onSave();
                }}
              >
                {isCreate ? (
                  <>
                    <label>
                      用户名
                      <input
                        value={form.username}
                        onChange={(e) => onChange('username', e.target.value)}
                        required
                        autoComplete="off"
                      />
                    </label>
                    <label>
                      密码
                      <input
                        type="password"
                        value={form.password}
                        onChange={(e) => onChange('password', e.target.value)}
                        required
                        minLength={6}
                        autoComplete="new-password"
                      />
                    </label>
                  </>
                ) : (
                  <DetailRow label="用户名" value={form.username} mono />
                )}
                <label>
                  显示名
                  <input
                    value={form.display_name}
                    onChange={(e) => onChange('display_name', e.target.value)}
                    required
                  />
                </label>
                <label>
                  角色
                  <select
                    value={form.role}
                    disabled={!isCreate && isSelf}
                    title={!isCreate && isSelf ? '不能修改自己的角色' : ''}
                    onChange={(e) => onChange('role', e.target.value)}
                  >
                    <option value="operator">操作员</option>
                    <option value="admin">管理员</option>
                  </select>
                </label>
                {!isCreate ? (
                  <label>
                    新密码
                    <input
                      type="password"
                      value={form.password}
                      onChange={(e) => onChange('password', e.target.value)}
                      placeholder="留空则不修改"
                      minLength={6}
                      autoComplete="new-password"
                    />
                  </label>
                ) : null}
              </form>
            </section>
          )}
        </div>

        <footer className="drawer-footer">
          <div className="drawer-footer-right">
            {isView ? (
              <>
                <button type="button" className="secondary" onClick={onClose}>
                  关闭
                </button>
                <button type="button" onClick={onSwitchToEdit}>
                  编辑
                </button>
              </>
            ) : (
              <>
                <button type="button" className="secondary" onClick={onClose}>
                  取消
                </button>
                <button type="submit" form="user-form" disabled={saving}>
                  {saving ? '保存中…' : isCreate ? '创建' : '保存'}
                </button>
              </>
            )}
          </div>
        </footer>
      </aside>
    </div>
  );
}
