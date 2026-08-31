import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import AppFooter from '../components/AppFooter';
import { useAuth } from '../context/AuthContext';
import { formatUserError } from '../lib/userFacingText';
import logoUrl from '../assets/logo.svg?url';
import './LoginPage.css';

const OAUTH_ERRORS = {
  access_denied: '已取消登录',
  invalid_state: '登录状态无效，请重试',
  missing_code: '授权未完成，请重试',
};

export default function LoginPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { authConfig, user, login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (user) {
      navigate('/', { replace: true });
    }
  }, [user, navigate]);

  useEffect(() => {
    const oauthErr = searchParams.get('error');
    if (oauthErr) {
      setError(OAUTH_ERRORS[oauthErr] || '单点登录失败，请重试');
    }
  }, [searchParams]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSubmitting(true);
    try {
      await login(username.trim(), password);
      navigate('/', { replace: true });
    } catch (err) {
      setError(formatUserError(err.message) || '登录失败');
    } finally {
      setSubmitting(false);
    }
  };

  const oauth = authConfig?.oauth2;
  const localEnabled = authConfig?.local?.enabled !== false;

  return (
    <div className="login-page">
      <div className="login-page-main">
      <div className="login-card">
        <h1 className="login-brand">
          <img src={logoUrl} alt="DiDPS" className="login-logo" width={172} height={40} />
          <span className="login-tagline">Digital Picking System</span>
        </h1>

        {error && <div className="login-error">{error}</div>}

        {localEnabled && (
          <form className="login-form" onSubmit={handleSubmit}>
            <label>
              用户名
              <input
                type="text"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </label>
            <label>
              密码
              <input
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </label>
            <button type="submit" disabled={submitting}>
              {submitting ? '登录中…' : '登录'}
            </button>
          </form>
        )}

        {oauth?.enabled && (
          <div className="login-oauth">
            {localEnabled && <span className="login-divider">或</span>}
            <a className="login-oauth-btn" href={oauth.loginPath}>
              {oauth.displayName || '企业单点登录'}
            </a>
          </div>
        )}
      </div>
      </div>
      <AppFooter />
    </div>
  );
}
