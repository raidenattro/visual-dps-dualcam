import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function RequireAuth({ children }) {
  const { loading, authRequired, user } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="auth-loading">
        <p>加载中…</p>
      </div>
    );
  }

  if (authRequired && !user) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }

  return children;
}
