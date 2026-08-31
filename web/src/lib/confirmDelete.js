/** 删除摄像头：二次确认（浏览器对话框） */
export function confirmDeleteCamera(name) {
  const label = name ? `「${name}」` : '该摄像头';
  if (!window.confirm(`确定删除摄像头${label}？`)) return false;
  if (!window.confirm('删除后无法恢复，请再次确认是否继续。')) return false;
  return true;
}

