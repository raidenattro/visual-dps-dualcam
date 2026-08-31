/** 计算 object-fit: contain 下图像在容器内的绘制区域 */

export function computeContainLayout(containerW, containerH, frameW, frameH) {
  const cw = Math.max(1, containerW);
  const ch = Math.max(1, containerH);
  const fw = Math.max(1, frameW || cw);
  const fh = Math.max(1, frameH || ch);
  const scale = Math.min(cw / fw, ch / fh);
  const drawW = fw * scale;
  const drawH = fh * scale;
  return {
    scale,
    offsetX: (cw - drawW) / 2,
    offsetY: (ch - drawH) / 2,
    drawW,
    drawH,
    frameW: fw,
    frameH: fh,
  };
}

function polygonMaxExtent(points) {
  let maxX = 0;
  let maxY = 0;
  if (!Array.isArray(points)) return { maxX, maxY };
  for (const pt of points) {
    if (!Array.isArray(pt) || pt.length < 2) continue;
    maxX = Math.max(maxX, Number(pt[0]) || 0);
    maxY = Math.max(maxY, Number(pt[1]) || 0);
  }
  return { maxX, maxY };
}

/** 与 inference_service._norm_polygon_valid 一致 */
export function isNormPolygonValid(normPts) {
  if (!Array.isArray(normPts) || normPts.length < 3) return false;
  for (const pt of normPts) {
    if (!Array.isArray(pt) || pt.length < 2) continue;
    const x = Number(pt[0]);
    const y = Number(pt[1]);
    if (x < -0.01 || x > 1.01 || y < -0.01 || y > 1.01) return false;
  }
  return true;
}

/**
 * 将标注多边形换算到当前视频帧像素坐标（与推理侧 _scale_polygon_to_frame 对齐）
 */
export function resolvePolygonFramePoints(polygon, normPolygon, annotationSize, frameW, frameH) {
  const tw = Math.max(1, frameW);
  const th = Math.max(1, frameH);
  if (isNormPolygonValid(normPolygon)) {
    return normPolygon.map(([x, y]) => [Number(x) * tw, Number(y) * th]);
  }
  if (!Array.isArray(polygon) || polygon.length < 3) return [];

  const { maxX, maxY } = polygonMaxExtent(polygon);
  const annW = Math.max(1, Number(annotationSize?.width) || tw);
  const annH = Math.max(1, Number(annotationSize?.height) || th);
  let sx = tw / annW;
  let sy = th / annH;
  if (maxX > annW * 1.05) sx = maxX > 0 ? tw / maxX : sx;
  if (maxY > annH * 1.05) sy = maxY > 0 ? th / maxY : sy;
  return polygon.map(([x, y]) => [Number(x) * sx, Number(y) * sy]);
}

/** 帧坐标 → viewport 显示坐标（layout 由 computeContainLayout 得到） */
export function mapPointToDisplay(x, y, layout) {
  const frameW = Math.max(1, layout.frameW);
  const frameH = Math.max(1, layout.frameH);
  const sx = layout.drawW / frameW;
  const sy = layout.drawH / frameH;
  return [layout.offsetX + Number(x) * sx, layout.offsetY + Number(y) * sy];
}

/** 将标注坐标映射到当前视频帧像素（忽略无效的 video_polygon_norm） */
export function mapPointsToVideoFrame(points, normPolygon, annotationSize, frameW, frameH) {
  const norm = isNormPolygonValid(normPolygon) ? normPolygon : null;
  return resolvePolygonFramePoints(points, norm, annotationSize, frameW, frameH);
}

export function polygonToPoints(polygon, layout, annotationSize, normPolygon = null) {
  const framePts = mapPointsToVideoFrame(
    polygon,
    normPolygon,
    annotationSize,
    layout.frameW,
    layout.frameH,
  );
  return framePts.map(([x, y]) => mapPointToDisplay(x, y, layout).join(',')).join(' ');
}

/** 帧像素坐标 → SVG 点串（与视频同尺寸叠层时使用） */
export function polygonToFramePoints(polygon, annotationSize, normPolygon, frameW, frameH) {
  const framePts = mapPointsToVideoFrame(polygon, normPolygon, annotationSize, frameW, frameH);
  return framePts.map(([x, y]) => `${x},${y}`).join(' ');
}

/** 映射后是否仍有任意顶点落在画面内（用于隐藏完全离屏的货架 ROI） */
export function isGeometryInFrame(framePts, frameW, frameH, marginRatio = 0.02) {
  if (!Array.isArray(framePts) || !framePts.length) return false;
  const fw = Math.max(1, frameW);
  const fh = Math.max(1, frameH);
  const mx = fw * marginRatio;
  const mh = fh * marginRatio;
  for (const pt of framePts) {
    if (!Array.isArray(pt) || pt.length < 2) continue;
    const x = Number(pt[0]);
    const y = Number(pt[1]);
    if (x >= -mx && x <= fw + mx && y >= -mh && y <= fh + mh) return true;
  }
  const cx = framePts.reduce((s, p) => s + (Number(p[0]) || 0), 0) / framePts.length;
  const cy = framePts.reduce((s, p) => s + (Number(p[1]) || 0), 0) / framePts.length;
  return cx >= -mx && cx <= fw + mx && cy >= -mh && cy <= fh + mh;
}
