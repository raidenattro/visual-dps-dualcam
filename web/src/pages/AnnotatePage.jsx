import { Navigate } from 'react-router-dom';

/** 旧 2D 透视标注入口改到双路巷道标注 */
export default function AnnotatePage() {
  return <Navigate to="/aisle" replace />;
}
