import './CameraSetupDrawer.css';
import './TopologyDetailDrawer.css';

function DetailSection({ title, children }) {
  return (
    <section className="drawer-section topology-drawer-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function EdgeList({ edges }) {
  if (!edges?.length) {
    return <p className="topology-drawer-empty">暂无连接数据</p>;
  }
  return (
    <div className="topology-edge-list">
      {edges.map((edge) => (
        <div key={edge.id} className={`topology-edge-row ${edge.health || 'unknown'}`}>
          <span className="topology-edge-dir">
            {edge.direction} · {edge.protocol}
            {edge.role ? ` · ${edge.role}` : ''}
          </span>
          <span className="topology-edge-endpoint">{edge.endpoint}</span>
        </div>
      ))}
    </div>
  );
}

function KvList({ rows }) {
  return (
    <dl className="topology-kv">
      {rows.map(([k, v]) => (
        <div key={k} style={{ display: 'contents' }}>
          <dt>{k}</dt>
          <dd>{v ?? '—'}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function TopologyDetailDrawer({
  open,
  path,
  edges = [],
  edgeSectionTitle = '链路连接',
  onClose,
}) {
  if (!open) return null;

  const inf = path?.inference || {};
  const mtx = path?.mediamtx || {};
  const rt = path?.runtime || {};
  const cfg = path?.configured || {};
  const title = path ? path.camera_name || path.camera_id : '链路详情';

  return (
    <div className="drawer-root" role="presentation">
      <button type="button" className="drawer-backdrop" aria-label="关闭" onClick={onClose} />
      <aside className="drawer-panel topology-drawer-panel" role="dialog" aria-labelledby="topology-drawer-title">
        <header className="drawer-header">
          <div>
            <h2 id="topology-drawer-title">链路详情</h2>
            <p className="drawer-subtitle">{title}</p>
          </div>
          <button type="button" className="drawer-close" onClick={onClose} aria-label="关闭">
            ×
          </button>
        </header>

        <div className="drawer-body">
          {!path ? (
            <p className="topology-drawer-empty">请选择摄像头或点击拓扑图中的源/推理节点。</p>
          ) : (
            <>
              <div className="topology-drawer-head">
                <span className={`topology-health-badge ${path.health || 'unknown'}`}>
                  {path.health}
                </span>
                <span className="topology-drawer-cam-id">{path.camera_id}</span>
              </div>

              {path.issues?.length ? (
                <DetailSection title="问题码">
                  <p className="topology-drawer-note">{path.issues.join(' · ')}</p>
                </DetailSection>
              ) : null}

              <DetailSection title={edgeSectionTitle}>
                <EdgeList edges={edges} />
              </DetailSection>

              <DetailSection title="配置地址">
                <KvList
                  rows={[
                    ['播放 URL', cfg.playback_url],
                    ['拉流 URL', cfg.pull_url],
                    ['标注来源', cfg.annotation_camera_url],
                    ['MTX path', cfg.mtx_path],
                    ['HLS', cfg.hls],
                  ]}
                />
              </DetailSection>

              <DetailSection title="运行时">
                <KvList
                  rows={[
                    ['infer 拉流', rt.infer_stream_url],
                    [
                      '拉流探测',
                      rt.infer_stream_probe?.reachable
                        ? `OK ${rt.infer_stream_probe.latency_ms ?? ''}ms`
                        : rt.infer_stream_probe?.error || '—',
                    ],
                    ['外部有流提示', rt.external_publish_hint || '—'],
                    ['MTX ready', mtx.ready ? '是' : '否'],
                    ['MTX 入站', mtx.bytes_received],
                    ['推理状态', inf.status],
                    ['后端', inf.backend],
                    ['容器', inf.container_name],
                    ['容器 IP', inf.ip],
                  ]}
                />
              </DetailSection>

              <DetailSection title="Pose / 事件">
                <KvList
                  rows={[
                    [
                      'pose 年龄',
                      rt.pose?.last_ts_age_sec != null
                        ? `${rt.pose.last_ts_age_sec}s`
                        : '—',
                    ],
                    ['pose frame_idx', rt.pose?.frame_idx ?? '—'],
                    [
                      'pose 发布中',
                      rt.pose?.publishing == null ? '—' : rt.pose.publishing ? '是' : '否',
                    ],
                    [
                      '关键点冻结',
                      rt.pose?.frozen == null ? '—' : rt.pose.frozen ? '是（旧帧）' : '否',
                    ],
                    ['stream Δframe', rt.pose?.recent_frame_delta ?? '—'],
                    [
                      'event pose 年龄',
                      path.event?.last_pose_age_sec != null
                        ? `${path.event.last_pose_age_sec}s`
                        : '—',
                    ],
                    ['event-worker', path.event?.worker_container || '—'],
                  ]}
                />
              </DetailSection>
            </>
          )}
        </div>
      </aside>
    </div>
  );
}
