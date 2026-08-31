import { usePageTitle } from '../hooks/usePageTitle';

/** 根据路由同步 document.title */
export default function PageTitle() {
  usePageTitle();
  return null;
}
