import { useCallback, useEffect, useMemo, useState } from 'react';
import { apiGet } from '../api/client';
import TopologyDetailDrawer from '../components/TopologyDetailDrawer';
import TopologyGraph from '../components/TopologyGraph';
import {
  cameraIdFromNode,
  edgeBelongsToCamera,
  edgesIncidentToNode,
} from '../lib/topologyEdges';
import { formatUserError } from '../lib/userFacingText';
import './TopologyPage.css';

const DEFAULT_POLL_MS = 10000;

export default function TopologyPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [bootstrapping, setBootstrapping] = useState(true);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [selectedNodeId, setSelectedNodeId] = useState('');
  const [deepProbe, setDeepProbe] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const pollMs = data?.poll_recommended_ms || DEFAULT_POLL_MS;

  const load = useCallback(async () => {
    setRefreshing(true);
    try {
      const res = await apiGet(`/api/topology/overview?probe=${deepProbe ? 'true' : 'false'}`);
      if (res.error || res.status === 'error') {
        setErr(formatUserError(res.error || res.message) || '加载失败');
        setData(null);
        return;
      }
      if (!res.graph) {
        setErr('拓扑数据格式异常，请确认 UI 镜像已更新');
        setData(null);
        return;
      }
      setData(res);
      setErr('');
    } catch (e) {
      setErr(formatUserError(e.message) || '加载失败');
    } finally {
      setRefreshing(false);
      setBootstrapping(false);
    }
  }, [deepProbe]);

  useEffect(() => {
    load();
    const t = setInterval(load, pollMs);
    return () => clearInterval(t);
  }, [load, pollMs]);

  useEffect(() => {
    if (!drawerOpen) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') setDrawerOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [drawerOpen]);

  const paths = data?.paths || [];
  const selectedPath = useMemo(
    () => paths.find((p) => p.camera_id === selectedCameraId) || null,
    [paths, selectedCameraId],
  );

  const pathEdges = useMemo(() => {
    const edges = data?.graph?.edges || [];
    const cid = selectedPath?.camera_id;
    if (!cid) return edges;
    return edges.filter((e) => edgeBelongsToCamera(e, cid));
  }, [data, selectedPath]);

  const drawerEdges = useMemo(() => {
    if (!selectedNodeId) return pathEdges;
    return edgesIncidentToNode(pathEdges, selectedNodeId);
  }, [pathEdges, selectedNodeId]);

  const edgeSectionTitle = useMemo(() => {
    const cid = selectedPath?.camera_id;
    if (!cid) return '链路连接';
    if (!selectedNodeId) return `链路连接（${cid}）`;
    const node = (data?.graph?.nodes || []).find((n) => n.id === selectedNodeId);
    return `链路连接（${node?.label || selectedNodeId}）`;
  }, [data, selectedPath, selectedNodeId]);

  const updatedLabel = data?.generated_at
    ? new Date(data.generated_at * 1000).toLocaleTimeString()
    : '—';

  const clearSelection = () => {
    setSelectedCameraId('');
    setSelectedNodeId('');
  };

  const onSelectNode = (node) => {
    const cid = cameraIdFromNode(node);
    if (!cid) return;
    if (selectedNodeId === node.id && selectedCameraId === cid) {
      setDrawerOpen(false);
      clearSelection();
      return;
    }
    setSelectedNodeId(node.id);
    setSelectedCameraId(cid);
    setDrawerOpen(true);
  };

  const onCloseDrawer = () => {
    setDrawerOpen(false);
    clearSelection();
  };

  return (
    <div className="page topology-page">
      <header className="topology-toolbar">
        <div>
          <h1>服务拓扑</h1>
        </div>
        <div className="topology-toolbar-right">
          <span className="topology-updated">更新 {updatedLabel}</span>
          <label
            className={`monitor-layer-switch${deepProbe ? ' on' : ''}`}
            title="开启后对 infer 拉流做 ffprobe 探测（较慢）"
          >
            <input
              type="checkbox"
              checked={deepProbe}
              onChange={(e) => setDeepProbe(e.target.checked)}
              aria-label="RTSP 探测"
            />
            <span className="monitor-layer-switch-track" aria-hidden="true" />
            <span className="monitor-layer-switch-label">RTSP 探测</span>
          </label>
          <button type="button" className="topology-refresh" onClick={load} disabled={refreshing}>
            刷新
          </button>
        </div>
      </header>

      {err ? <p className="topology-msg err">{err}</p> : null}
      {bootstrapping && !data ? <p className="topology-msg">加载拓扑数据…</p> : null}

      {data?.graph ? (
        <div className="topology-graph-panel">
          <TopologyGraph
            nodes={data.graph.nodes}
            edges={data.graph.edges}
            focusCameraId={selectedCameraId}
            selectedNodeId={selectedNodeId}
            onSelectNode={onSelectNode}
          />
        </div>
      ) : null}

      {data && !paths.length ? <p className="topology-msg">暂无摄像头</p> : null}

      <TopologyDetailDrawer
        open={drawerOpen}
        path={selectedPath}
        edges={drawerEdges}
        edgeSectionTitle={edgeSectionTitle}
        onClose={onCloseDrawer}
      />
    </div>
  );
}
