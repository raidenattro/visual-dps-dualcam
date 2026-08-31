import { lusolve } from 'mathjs';

export function solveLinearSystem(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < 1e-12) {
      throw new Error('矩阵不可逆，无法计算透视变换');
    }
    if (pivot !== col) [M[col], M[pivot]] = [M[pivot], M[col]];

    const div = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= div;

    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }

  return M.map((row) => row[n]);
}

export function getPerspectiveTransform(src, dst) {
  const A = [];
  const B = [];
  for (let i = 0; i < 4; i++) {
    A.push([src[i][0], src[i][1], 1, 0, 0, 0, -src[i][0] * dst[i][0], -src[i][1] * dst[i][0]]);
    A.push([0, 0, 0, src[i][0], src[i][1], 1, -src[i][0] * dst[i][1], -src[i][1] * dst[i][1]]);
    B.push(dst[i][0]);
    B.push(dst[i][1]);
  }
  let h;
  try {
    h = lusolve(A, B).map((x) => x[0]);
  } catch {
    h = solveLinearSystem(A, B);
  }
  h.push(1.0);
  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], h[8]],
  ];
}

export function perspectiveTransform(pt, M) {
  const z = M[2][0] * pt[0] + M[2][1] * pt[1] + M[2][2];
  const x = (M[0][0] * pt[0] + M[0][1] * pt[1] + M[0][2]) / z;
  const y = (M[1][0] * pt[0] + M[1][1] * pt[1] + M[1][2]) / z;
  return [x, y];
}

export function getDist(p, a, b) {
  const l2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
  if (l2 === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / l2));
  return Math.hypot(p[0] - (a[0] + t * (b[0] - a[0])), p[1] - (a[1] + t * (b[1] - a[1])));
}

export function getPointDist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

/** 射线法判断点是否在多边形内 */
export function pointInPolygon(point, polygon) {
  if (!Array.isArray(polygon) || polygon.length < 3) return false;
  const [x, y] = point;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0];
    const yi = polygon[i][1];
    const xj = polygon[j][0];
    const yj = polygon[j][1];
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi || 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

/** 四边形各顶点向中心收缩，ratio 为边长方向内缩比例（如 0.08 ≈ 8% 边距） */
export function insetConvexQuad(poly, ratio = 0.08) {
  if (!Array.isArray(poly) || poly.length < 4 || ratio <= 0) {
    return poly;
  }
  const n = 4;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < n; i++) {
    cx += poly[i][0];
    cy += poly[i][1];
  }
  cx /= n;
  cy /= n;
  const scale = 1 - Math.min(ratio, 0.4);
  return poly.slice(0, 4).map(([x, y]) => [cx + (x - cx) * scale, cy + (y - cy) * scale]);
}
