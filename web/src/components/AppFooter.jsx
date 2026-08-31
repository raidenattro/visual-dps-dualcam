import { useEffect, useState } from 'react';
import { APP_VERSION } from '../build-info.js';
import { apiGet } from '../api/client';
import './AppFooter.css';

function fallbackLine() {
  return `UI/API ${APP_VERSION} · Event — · Infer —`;
}

export default function AppFooter() {
  const [versionLine, setVersionLine] = useState(fallbackLine());

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await apiGet('/api/version');
        if (!cancelled && data?.display) {
          setVersionLine(data.display);
        }
      } catch {
        /* 未登录或 API 不可用时保留前端 build 号 */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <footer className="app-footer">
      <span className="app-footer-copy">
        © 2026-2027 米道（基于 MAPS<sup className="tm-mark">TM</sup> 开发）
      </span>
      <span className="app-footer-version" title={versionLine}>
        {versionLine}
      </span>
    </footer>
  );
}
