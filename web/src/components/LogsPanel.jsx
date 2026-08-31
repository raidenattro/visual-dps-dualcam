import { useCallback, useEffect, useMemo, useState } from 'react';
import { apiGet } from '../api/client';
import { formatUserError } from '../lib/userFacingText';
import './LogsPanel.css';

const PAGE_SIZE = 50;

const AUDIT_COLUMNS = [
  { key: 'ts', label: '时间', sortable: true },
  { key: 'actor', label: '操作者', sortable: true },
  { key: 'action', label: '动作', sortable: true },
  { key: 'resource_id', label: '资源', sortable: true },
  { key: 'result', label: '结果', sortable: true },
];

const EVENT_COLUMNS = [
  { key: 'ts', label: '时间', sortable: true },
  { key: 'event_type', label: '类型', sortable: true },
  { key: 'camera_id', label: '摄像头', sortable: true },
  { key: 'severity', label: '级别', sortable: true },
  { key: 'summary', label: '摘要', sortable: false },
];

const DEFAULT_AUDIT_FILTERS = {
  actor: '',
  action: '',
  result: '',
  resource_id: '',
};

const DEFAULT_EVENT_FILTERS = {
  event_type: '',
  camera_id: '',
  severity: '',
  summary: '',
};

function buildQuery(params) {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && String(v).trim() !== '') {
      qs.set(k, String(v));
    }
  });
  return qs.toString();
}

export default function LogsPanel() {
  const [mode, setMode] = useState('audit');
  const isAudit = mode === 'audit';
  const columns = isAudit ? AUDIT_COLUMNS : EVENT_COLUMNS;
  const defaultFilters = isAudit ? DEFAULT_AUDIT_FILTERS : DEFAULT_EVENT_FILTERS;

  const [filters, setFilters] = useState(defaultFilters);
  const [queryFilters, setQueryFilters] = useState(defaultFilters);
  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState('ts');
  const [sortOrder, setSortOrder] = useState('desc');
  const [data, setData] = useState({ items: [], total: 0 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const totalPages = useMemo(
    () => Math.max(1, Math.ceil((data.total || 0) / PAGE_SIZE)),
    [data.total],
  );

  const fetchLogs = useCallback(async () => {
    setLoading(true);
    setError('');
    const base = isAudit ? '/api/logs/audit' : '/api/logs/events';
    const query = buildQuery({
      page,
      page_size: PAGE_SIZE,
      sort_by: sortBy,
      sort_order: sortOrder,
      ...queryFilters,
    });
    try {
      const res = await apiGet(`${base}?${query}`);
      if (res.status !== 'success') {
        setError(formatUserError(res.error) || '加载失败');
        setData({ items: [], total: 0 });
        return;
      }
      setData({ items: res.items || [], total: res.total || 0 });
    } catch (err) {
      setError(formatUserError(err.message) || '加载失败');
      setData({ items: [], total: 0 });
    } finally {
      setLoading(false);
    }
  }, [isAudit, page, sortBy, sortOrder, queryFilters]);

  useEffect(() => {
    const next = mode === 'audit' ? DEFAULT_AUDIT_FILTERS : DEFAULT_EVENT_FILTERS;
    setFilters({ ...next });
    setQueryFilters({ ...next });
    setPage(1);
    setSortBy('ts');
    setSortOrder('desc');
  }, [mode]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setQueryFilters({ ...filters });
      setPage(1);
    }, 300);
    return () => window.clearTimeout(timer);
  }, [filters]);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  const setFilter = (key, value) => {
    setFilters((f) => ({ ...f, [key]: value }));
  };

  const toggleSort = (key) => {
    if (!columns.find((c) => c.key === key)?.sortable) return;
    if (sortBy === key) {
      setSortOrder((o) => (o === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortBy(key);
      setSortOrder(key === 'ts' ? 'desc' : 'asc');
    }
    setPage(1);
  };

  const sortIndicator = (key) => {
    if (sortBy !== key) return '';
    return sortOrder === 'asc' ? ' ↑' : ' ↓';
  };

  return (
    <section className="settings-panel settings-panel--logs">
      <div className="settings-subtabs">
        <button type="button" className={mode === 'audit' ? 'active' : ''} onClick={() => setMode('audit')}>
          操作审计
        </button>
        <button type="button" className={mode === 'events' ? 'active' : ''} onClick={() => setMode('events')}>
          业务事件
        </button>
      </div>

      <div className="settings-filters settings-filters--inline">
        {isAudit ? (
          <>
            <label>
              操作者
              <input
                value={filters.actor}
                onChange={(e) => setFilter('actor', e.target.value)}
                placeholder="模糊匹配"
              />
            </label>
            <label>
              动作
              <input
                value={filters.action}
                onChange={(e) => setFilter('action', e.target.value)}
                placeholder="如 camera、inference"
              />
            </label>
            <label>
              结果
              <select value={filters.result} onChange={(e) => setFilter('result', e.target.value)}>
                <option value="">全部</option>
                <option value="success">success</option>
                <option value="error">error</option>
              </select>
            </label>
            <label>
              资源 ID
              <input
                value={filters.resource_id}
                onChange={(e) => setFilter('resource_id', e.target.value)}
                placeholder="摄像头 ID 等"
              />
            </label>
          </>
        ) : (
          <>
            <label>
              事件类型
              <input
                value={filters.event_type}
                onChange={(e) => setFilter('event_type', e.target.value)}
                placeholder="模糊匹配"
              />
            </label>
            <label>
              摄像头
              <input
                value={filters.camera_id}
                onChange={(e) => setFilter('camera_id', e.target.value)}
                placeholder="摄像头 ID"
              />
            </label>
            <label>
              级别
              <select value={filters.severity} onChange={(e) => setFilter('severity', e.target.value)}>
                <option value="">全部</option>
                <option value="info">info</option>
                <option value="warn">warn</option>
                <option value="error">error</option>
              </select>
            </label>
            <label>
              摘要
              <input
                value={filters.summary}
                onChange={(e) => setFilter('summary', e.target.value)}
                placeholder="模糊匹配"
              />
            </label>
          </>
        )}
      </div>

      {error ? <p className="settings-msg err">{error}</p> : null}

      <table className="settings-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col.key}>
                  {col.sortable ? (
                    <button
                      type="button"
                      className={`logs-sort-btn${sortBy === col.key ? ' is-active' : ''}`}
                      onClick={() => toggleSort(col.key)}
                    >
                      {col.label}
                      {sortIndicator(col.key)}
                    </button>
                  ) : (
                    col.label
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading && data.items.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="settings-cell-empty">
                  加载中…
                </td>
              </tr>
            ) : null}
            {!loading && data.items.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="settings-cell-empty">
                  无匹配记录
                </td>
              </tr>
            ) : null}
            {data.items.map((row) =>
              isAudit ? (
                <tr key={row.id}>
                  <td className="logs-cell-time">{row.ts_iso}</td>
                  <td>{row.actor}</td>
                  <td className="settings-cell-mono">{row.action}</td>
                  <td className="settings-cell-mono">{row.resource_id || '—'}</td>
                  <td>
                    <span className={row.result === 'success' ? 'log-badge-ok' : 'log-badge-err'}>
                      {row.result}
                    </span>
                  </td>
                </tr>
              ) : (
                <tr key={row.id}>
                  <td className="logs-cell-time">{row.ts_iso}</td>
                  <td className="settings-cell-mono">{row.event_type}</td>
                  <td className="settings-cell-mono">{row.camera_id || '—'}</td>
                  <td>{row.severity || '—'}</td>
                  <td className="logs-cell-summary">{row.summary || '—'}</td>
                </tr>
              ),
            )}
          </tbody>
      </table>

      <div className="settings-pagination">
        <button
          type="button"
          className="settings-btn-secondary"
          disabled={loading || page <= 1}
          onClick={() => setPage((p) => Math.max(1, p - 1))}
        >
          上一页
        </button>
        <span className="settings-page-indicator">
          第
          <input
            type="number"
            className="settings-page-input"
            min={1}
            max={totalPages}
            value={page}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (Number.isFinite(n) && n >= 1 && n <= totalPages) setPage(n);
            }}
          />
          / {totalPages} 页
        </span>
        <button
          type="button"
          className="settings-btn-secondary"
          disabled={loading || page >= totalPages}
          onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
        >
          下一页
        </button>
      </div>
    </section>
  );
}
