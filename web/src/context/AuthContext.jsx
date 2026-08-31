import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { apiGet, apiPost } from '../api/client';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [loading, setLoading] = useState(true);
  const [authConfig, setAuthConfig] = useState(null);
  const [user, setUser] = useState(null);

  const refresh = useCallback(async () => {
    const [cfg, me] = await Promise.all([apiGet('/api/auth/config'), apiGet('/api/auth/me')]);
    setAuthConfig(cfg);
    if (me.authenticated) {
      setUser(me.user);
    } else {
      setUser(null);
    }
    return { cfg, me };
  }, []);

  useEffect(() => {
    (async () => {
      try {
        await refresh();
      } finally {
        setLoading(false);
      }
    })();
  }, [refresh]);

  const login = useCallback(async (username, password) => {
    const resp = await fetch('/api/auth/login', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      const detail = data.detail;
      const msg = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail[0]?.msg : data.error;
      throw new Error(msg || '登录失败');
    }
    setUser(data.user);
    return data.user;
  }, []);

  const logout = useCallback(async () => {
    await apiPost('/api/auth/logout', {});
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({
      loading,
      authConfig,
      user,
      authRequired: authConfig?.enabled === true,
      login,
      logout,
      refresh,
    }),
    [loading, authConfig, user, login, logout, refresh],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
