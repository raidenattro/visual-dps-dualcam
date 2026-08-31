/** 双路货格几何：3D 网格为唯一真值，两路画面都是投影。口径对齐 scripts/dualcam_geom.py。 */

const EPS = 1e-8;
const MIN_DEPTH = 0.05;
export const DEFAULT_LAYER_PITCH = 0.45;
export const DEFAULT_CONTACT_M = 0;

function v3(a) {
  return [Number(a[0]), Number(a[1]), Number(a[2])];
}
function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function mul(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function norm(a) {
  const n = Math.hypot(a[0], a[1], a[2]) || EPS;
  return mul(a, 1 / n);
}

export function projectPix(p, cam) {
  const C = v3(cam.C);
  const v = sub(v3(p), C);
  const zc = dot(v, cam.fwd);
  if (zc < MIN_DEPTH) return null;
  return [
    cam.cx + cam.f * dot(v, cam.right) / zc,
    cam.cy + cam.f * dot(v, cam.down) / zc,
  ];
}

export function pixelRay(u, v, cam) {
  const d = norm(add(add(mul(cam.right, (u - cam.cx) / cam.f), mul(cam.down, (v - cam.cy) / cam.f)), cam.fwd));
  return { C: v3(cam.C), d };
}

export function wallPlane(corners) {
  const c0 = v3(corners[0]), c1 = v3(corners[1]), c3 = v3(corners[3]);
  return { p0: c0, n: norm(cross(sub(c1, c0), sub(c3, c0))) };
}

export function rayPlane(u, v, cam, p0, n) {
  const { C, d } = pixelRay(u, v, cam);
  const den = dot(d, n);
  if (Math.abs(den) < 1e-8) return null;
  const t = dot(sub(p0, C), n) / den;
  if (t < MIN_DEPTH) return null;
  return add(C, mul(d, t));
}

/** 墙角沿 X 朝巷道平移 inset 米。 */
export function offsetCorners(corners, sign, inset) {
  const dx = -Number(sign) * Number(inset || 0);
  return corners.map((p) => [p[0] + dx, p[1], p[2]]);
}

export function triangulateRays(C1, d1, C2, d2, maxGap = 0.5, attach = 'mid') {
  const w0 = sub(C1, C2);
  const a = dot(d1, d1), b = dot(d1, d2), c = dot(d2, d2);
  const d = dot(d1, w0), e = dot(d2, w0);
  const denom = a * c - b * b;
  if (Math.abs(denom) < 1e-12) return null;
  const t = (b * e - c * d) / denom;
  const s = (a * e - b * d) / denom;
  if (t < MIN_DEPTH || s < MIN_DEPTH) return null;
  const p1 = add(C1, mul(d1, t));
  const p2 = add(C2, mul(d2, s));
  if (Math.hypot(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]) > maxGap) return null;
  if (attach === 'first') return p1;
  if (attach === 'second') return p2;
  return mul(add(p1, p2), 0.5);
}

export function triangulatePixels(u1, v1, cam1, u2, v2, cam2, attach = 'mid') {
  const a = pixelRay(u1, v1, cam1);
  const b = pixelRay(u2, v2, cam2);
  return triangulateRays(a.C, a.d, b.C, b.d, 0.5, attach);
}

/** 默认射线∩开口平面。stereo 时本路跟鼠标。 */
export function dragVertex(u, v, cam, otherCam, current, p0, n, stereo = false) {
  if (stereo) {
    const uvO = projectPix(current, otherCam);
    if (uvO) {
      const p = triangulatePixels(u, v, cam, uvO[0], uvO[1], otherCam, 'first');
      if (p) return p;
    }
  }
  return rayPlane(u, v, cam, p0, n);
}

export function bilinearOnWall(corners, ty, tz) {
  const c0 = v3(corners[0]), c1 = v3(corners[1]), c2 = v3(corners[2]), c3 = v3(corners[3]);
  return add(add(
    add(mul(c0, (1 - ty) * (1 - tz)), mul(c1, (1 - ty) * tz)),
    mul(c2, ty * tz),
  ), mul(c3, ty * (1 - tz)));
}

export function makeGridVertices(corners, rows, cols) {
  const out = [];
  for (let r = 0; r <= rows; r++) {
    const ty = r / rows;
    for (let c = 0; c <= cols; c++) out.push(bilinearOnWall(corners, ty, c / cols));
  }
  return out;
}

export function wallYSpan(corners) {
  const ys = corners.map((p) => Number(p[1]));
  return [Math.min(...ys), Math.max(...ys)];
}

export const MIN_LAYER_GAP = 0.03;

export function equalRowYs(yBottom, yTop, nLayers) {
  const n = Math.max(1, Math.round(Number(nLayers)) || 1);
  yBottom = Number(yBottom);
  yTop = Number(yTop);
  const h = yTop - yBottom;
  const ys = [];
  for (let i = 0; i <= n; i++) ys.push(yTop - i * h / n);
  return ys;
}

export function rowYsFromMesh(mesh) {
  if (Array.isArray(mesh.row_ys) && mesh.row_ys.length >= 2) {
    return mesh.row_ys.map(Number);
  }
  const rows = mesh.rows, cols = mesh.cols, verts = mesh.vertices;
  const ys = [];
  for (let r = 0; r <= rows; r++) ys.push(Number(verts[r * (cols + 1)][1]));
  return ys;
}

export function meshFromRowYs(wallId, corners, rowYs, cols = 4, extra = null) {
  const [yBot, yTop] = wallYSpan(corners);
  let ys = (rowYs && rowYs.length >= 2) ? rowYs.map(Number) : [yTop, yBot];
  ys[0] = yTop;
  ys[ys.length - 1] = yBot;
  cols = Math.max(1, Math.round(Number(cols)) || 1);
  const height = yTop - yBot;
  const tys = ys.map((y) => (Math.abs(height) < 1e-9 ? 0 : (yTop - y) / height));
  const vertices = [];
  for (const ty of tys) {
    for (let c = 0; c <= cols; c++) vertices.push(bilinearOnWall(corners, ty, c / cols));
  }
  const rows = ys.length - 1;
  const out = {
    wall_id: wallId,
    rows,
    cols,
    n_layers: rows,
    row_ys: ys.map((y) => Math.round(y * 1e4) / 1e4),
    vertices,
  };
  if (extra && typeof extra === 'object') {
    const code = String(extra.shelf_code || '').trim();
    if (code) out.shelf_code = code;
    if (extra.cell_ids && typeof extra.cell_ids === 'object') out.cell_ids = extra.cell_ids;
    if (Array.isArray(extra.deleted) && extra.deleted.length) out.deleted = extra.deleted;
  }
  return out;
}

export function moveLayerRow(mesh, corners, r, y) {
  const ys = rowYsFromMesh(mesh);
  const rows = ys.length - 1;
  r = Math.round(Number(r));
  if (r <= 0 || r >= rows) return meshFromRowYs(mesh.wall_id, corners, ys, mesh.cols, mesh);
  const hi = ys[r - 1] - MIN_LAYER_GAP;
  const lo = ys[r + 1] + MIN_LAYER_GAP;
  if (hi < lo) ys[r] = 0.5 * (ys[r - 1] + ys[r + 1]);
  else ys[r] = Math.min(Math.max(Number(y), lo), hi);
  return meshFromRowYs(mesh.wall_id, corners, ys, mesh.cols, mesh);
}

export function makeLayerMesh(wallId, corners, pitch = DEFAULT_LAYER_PITCH, nLayers = 4, cols = 4) {
  const [yBot, yTop] = wallYSpan(corners);
  return meshFromRowYs(wallId, corners, equalRowYs(yBot, yTop, nLayers), cols);
}

export function vertIndex(rows, cols, r, c) {
  return r * (cols + 1) + c;
}

export function meshCells(mesh) {
  const rows = mesh.rows, cols = mesh.cols, verts = mesh.vertices;
  const cells = [];
    for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const key = `r${i}c${j}`;
      const deleted = new Set((mesh.deleted || []).map(String));
      if (deleted.has(key)) continue;
      const ids = mesh.cell_ids || {};
      cells.push({
        row: i,
        col: j,
        box_id: ids[key] || key,
        slot_key: key,
        corners: [
          verts[vertIndex(rows, cols, i, j)],
          verts[vertIndex(rows, cols, i, j + 1)],
          verts[vertIndex(rows, cols, i + 1, j + 1)],
          verts[vertIndex(rows, cols, i + 1, j)],
        ],
      });
    }
  }
  return cells;
}

export function wallById(solved, wallId) {
  return ((solved && solved.walls) || []).find((w) => w.wall_id === wallId) || null;
}

export function signedWallDist(p, wall) {
  const p0 = v3(wall.corners[0]);
  const nx = Number(wall.sign) < 0 ? 1 : -1;
  return (p[0] - p0[0]) * nx;
}

export function contactSlots(p, meshes, solved, contactM = DEFAULT_CONTACT_M) {
  if (!p) return [];
  const hits = [];
  for (const mesh of meshes || []) {
    const wall = wallById(solved, mesh.wall_id);
    if (!wall) continue;
    const d = signedWallDist(p, wall);
    if (d >= contactM) continue;
    const yz = [p[1], p[2]];
    for (const cell of meshCells(mesh)) {
      const poly = cell.corners.map((c) => [c[1], c[2]]);
      if (pointInPolygon(yz, poly)) {
        hits.push({
          wall_id: mesh.wall_id,
          box_id: cell.box_id,
          row: cell.row,
          col: cell.col,
          d,
        });
      }
    }
  }
  return hits;
}

export function pointInPolygon(point, polygon) {
  if (!polygon || polygon.length < 3) return false;
  const [x, y] = point;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];
    const hit = (yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi || 1e-12) + xi;
    if (hit) inside = !inside;
  }
  return inside;
}

/** 3D 四角投影到某路；缺深度则整格丢掉。 */
export function projectCorners(corners, cam) {
  const pts = [];
  for (const p of corners) {
    const uv = projectPix(p, cam);
    if (!uv) return null;
    pts.push(uv);
  }
  return pts;
}
