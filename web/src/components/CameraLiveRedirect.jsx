import { useEffect, useState } from 'react';
import { Navigate, useParams } from 'react-router-dom';
import { apiGet } from '../api/client.js';
import { aisleLivePath } from '../lib/aisleNavigation.js';
import { formatUserError } from '../lib/userFacingText.js';

/** 旧 /camera/:id 书签 → 已编组则进巷道直播，否则回总览。 */
export default function CameraLiveRedirect() {
  const { cameraId = '' } = useParams();
  const [target, setTarget] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    const cid = String(cameraId || '').trim();
    if (!cid) {
      setTarget('/');
      return undefined;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await apiGet(`/api/aisles/by-camera/${encodeURIComponent(cid)}`);
        if (cancelled) return;
        const aisleId = res?.aisle?.aisle_id;
        const live = aisleLivePath(aisleId);
        setTarget(live || '/');
        if (!live && res?.error) setErr(formatUserError(res.error));
      } catch (e) {
        if (!cancelled) {
          setTarget('/');
          setErr(formatUserError(e.message));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [cameraId]);

  if (!target) {
    return <p className="matrix-msg">{err || '跳转中…'}</p>;
  }
  return <Navigate to={target} replace />;
}
