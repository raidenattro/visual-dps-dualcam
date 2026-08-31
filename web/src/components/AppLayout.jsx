import { Outlet } from 'react-router-dom';
import AppFooter from './AppFooter';
import AppNav from './AppNav';
import './AppFooter.css';

export default function AppLayout() {
  return (
    <div className="app-shell">
      <main className="app-main">
        <div className="wrap">
          <AppNav />
          <Outlet />
        </div>
      </main>
      <AppFooter />
    </div>
  );
}
