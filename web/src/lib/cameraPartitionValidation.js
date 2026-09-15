/** Dualcam 巷道与 Legacy 单路在添加入口的互斥校验（与后端 partition 语义一致） */

export function groupedCameraIdSet(aisles) {
  const ids = new Set();
  for (const a of aisles || []) {
    if (a.camera_l) ids.add(String(a.camera_l));
    if (a.camera_r) ids.add(String(a.camera_r));
  }
  return ids;
}

function normPath(p) {
  return String(p || '').trim().replace(/^\/+/, '');
}

/** 已有巷道占用的编号：aisle_id、aisle_id-L、aisle_id-R */
export function legacyPathVsAisleConflict(path, aisles) {
  const p = normPath(path);
  if (!p) return '';
  for (const a of aisles || []) {
    const aid = String(a.aisle_id || '').trim();
    if (!aid) continue;
    if (p === aid) {
      return `通道「${p}」与现有巷道编号冲突，单路请改用其他通道号。`;
    }
    if (p === `${aid}-L` || p === `${aid}-R`) {
      return `通道「${p}」为巷道 ${aid} 的 L/R 专用通道，不能作为单路添加。`;
    }
  }
  return '';
}

/** 添加单路（2D）前的客户端校验 */
export function validateLegacyCameraCreate(form, cameras, aisles) {
  const path = normPath(form?.path);
  if (!path) return '请填写通道编号';
  const aisleNs = legacyPathVsAisleConflict(path, aisles);
  if (aisleNs) return aisleNs;
  const grouped = groupedCameraIdSet(aisles);
  for (const cam of cameras || []) {
    const camPath = normPath(cam.path || cam.id);
    if (camPath !== path) continue;
    if (grouped.has(String(cam.id))) {
      return `通道「${path}」对应摄像头已编入巷道，不能作为单路添加。请用「添加巷道」或先解绑巷道。`;
    }
    return `通道号已被使用：${path}`;
  }
  return '';
}

function cameraPathOf(cam) {
  return normPath(cam?.path || cam?.id);
}

/** 添加巷道（双路 3D）前的客户端校验（与单路通道互斥） */
export function validateAisleCreate(aisleForm, aisles, cameras) {
  const aid = String(aisleForm?.aisle_id || '').trim();
  if (!aid) return '请填写巷道编号';
  if ((aisles || []).some((a) => String(a.aisle_id) === aid)) {
    return `巷道 ${aid} 已存在`;
  }
  const grouped = groupedCameraIdSet(aisles);
  for (const cam of cameras || []) {
    if (grouped.has(String(cam.id))) continue;
    if (cameraPathOf(cam) === aid) {
      return `巷道编号「${aid}」与已有单路通道「${aid}」冲突，请先删除或修改该单路。`;
    }
  }
  const lp = normPath(aisleForm?.left?.path);
  const rp = normPath(aisleForm?.right?.path);
  if (!lp || !rp) return '请填写左右路通道编号';
  if (lp === rp) return '左右路通道号不能相同';
  for (const path of [lp, rp]) {
    for (const cam of cameras || []) {
      if (cameraPathOf(cam) !== path) continue;
      if (grouped.has(String(cam.id))) {
        return `通道「${path}」已编入其他巷道，不能重复用于新路`;
      }
      return `通道「${path}」已有单路摄像头；请删除该单路或改用新通道号创建巷道`;
    }
  }
  return '';
}
