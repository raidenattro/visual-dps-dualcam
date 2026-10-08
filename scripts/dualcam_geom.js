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

/** 列界从远到近，与顶点列序一致：第 0 列在顶远一侧。 */
export function equalColZs(corners, nCols) {
  const n = Math.max(1, Math.round(Number(nCols)) || 1);
  const zFar = Number(corners[0][2]);
  const zNear = Number(corners[1][2]);
  const zs = [];
  for (let c = 0; c <= n; c++) zs.push(zFar + (zNear - zFar) * c / n);
  return zs;
}

export function colZsFromMesh(mesh, corners) {
  if (Array.isArray(mesh.col_zs) && mesh.col_zs.length >= 2) {
    return mesh.col_zs.map(Number);
  }
  if (corners) return equalColZs(corners, mesh.cols);
  const cols = mesh.cols, verts = mesh.vertices;
  const zs = [];
  for (let c = 0; c <= cols; c++) zs.push(Number(verts[c][2]));
  return zs;
}

function round4(v) { return Math.round(v * 1e4) / 1e4; }

/** 行界、列界都按墙上的米制坐标。列界缺省时沿墙宽均分。 */
export function meshFromGrid(wallId, corners, rowYs, colZs) {
  const [yBot, yTop] = wallYSpan(corners);
  let ys = (rowYs && rowYs.length >= 2) ? rowYs.map(Number) : [yTop, yBot];
  ys[0] = yTop;
  ys[ys.length - 1] = yBot;
  let zs = (colZs && colZs.length >= 2) ? colZs.map(Number) : equalColZs(corners, 4);
  const zFar = Number(corners[0][2]);
  const zNear = Number(corners[1][2]);
  zs[0] = zFar;
  zs[zs.length - 1] = zNear;
  const cols = zs.length - 1;
  const height = yTop - yBot;
  const span = zFar - zNear;
  const tys = ys.map((y) => (Math.abs(height) < 1e-9 ? 0 : (yTop - y) / height));
  const tzs = zs.map((z) => (Math.abs(span) < 1e-9 ? 0 : (zFar - z) / span));
  const vertices = [];
  for (const ty of tys) {
    for (const tz of tzs) vertices.push(bilinearOnWall(corners, ty, tz));
  }
  const rows = ys.length - 1;
  return {
    wall_id: wallId,
    rows,
    cols,
    n_layers: rows,
    row_ys: ys.map(round4),
    col_zs: zs.map(round4),
    vertices,
  };
}

export function meshFromRowYs(wallId, corners, rowYs, cols = 4, colZs = null) {
  const zs = (colZs && colZs.length >= 2) ? colZs : equalColZs(corners, cols);
  return meshFromGrid(wallId, corners, rowYs, zs);
}

export function moveLayerRow(mesh, corners, r, y) {
  const ys = rowYsFromMesh(mesh);
  const zs = colZsFromMesh(mesh, corners);
  const rows = ys.length - 1;
  r = Math.round(Number(r));
  if (r <= 0 || r >= rows) return meshFromGrid(mesh.wall_id, corners, ys, zs);
  const hi = ys[r - 1] - MIN_LAYER_GAP;
  const lo = ys[r + 1] + MIN_LAYER_GAP;
  if (hi < lo) ys[r] = 0.5 * (ys[r - 1] + ys[r + 1]);
  else ys[r] = Math.min(Math.max(Number(y), lo), hi);
  return meshFromGrid(mesh.wall_id, corners, ys, zs);
}

export function moveLayerCol(mesh, corners, c, z) {
  const ys = rowYsFromMesh(mesh);
  const zs = colZsFromMesh(mesh, corners);
  const cols = zs.length - 1;
  c = Math.round(Number(c));
  if (c <= 0 || c >= cols) return meshFromGrid(mesh.wall_id, corners, ys, zs);
  const far = zs[c - 1];
  const near = zs[c + 1];
  const hi = Math.max(far, near) - MIN_LAYER_GAP;
  const lo = Math.min(far, near) + MIN_LAYER_GAP;
  if (hi < lo) zs[c] = 0.5 * (far + near);
  else zs[c] = Math.min(Math.max(Number(z), lo), hi);
  return meshFromGrid(mesh.wall_id, corners, ys, zs);
}

/** 同侧不反解时的墙矩形：墙1 在 x=-巷道/2，墙2 在对面。角序仍是顶远、顶近、底近、底远。 */
export function nominalWall(wallId, width, height, base, aisle) {
  const sign = Number(wallId) === 2 ? 1 : -1;
  const x = sign * Number(aisle) / 2;
  const y0 = Number(base) || 0;
  const y1 = y0 + Number(height);
  const w = Number(width);
  return {
    wall_id: Number(wallId),
    sign,
    z_near: 0,
    corners: [
      [x, y1, w],
      [x, y1, 0],
      [x, y0, 0],
      [x, y0, w],
    ],
  };
}

function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    const tmp = M[col]; M[col] = M[piv]; M[piv] = tmp;
    const div = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= div;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row) => row[n]);
}

/** 四个角像素 ↔ 墙面 (z, y)。同侧格子投影用它，不用反解相机。 */
export function fitHomography(src, dst) {
  const A = [];
  const b = [];
  for (let i = 0; i < 4; i++) {
    const u = src[i][0], v = src[i][1];
    const x = dst[i][0], y = dst[i][1];
    A.push([u, v, 1, 0, 0, 0, -x * u, -x * v]);
    b.push(x);
    A.push([0, 0, 0, u, v, 1, -y * u, -y * v]);
    b.push(y);
  }
  const h = solveLinear(A, b);
  if (!h) return null;
  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1],
  ];
}

export function applyH(H, uv) {
  if (!H) return null;
  const x = H[0][0] * uv[0] + H[0][1] * uv[1] + H[0][2];
  const y = H[1][0] * uv[0] + H[1][1] * uv[1] + H[1][2];
  const w = H[2][0] * uv[0] + H[2][1] * uv[1] + H[2][2];
  if (Math.abs(w) < 1e-12) return null;
  return [x / w, y / w];
}

export function wallRectDst(width, height, base) {
  const y0 = Number(base) || 0;
  const y1 = y0 + Number(height);
  const w = Number(width);
  return [[w, y1], [0, y1], [0, y0], [w, y0]];
}

/** 单应 (z, y) 落在手拖行列界的哪一格。 */
export function cellAtZy(mesh, zy) {
  if (!mesh || !zy) return null;
  const ys = rowYsFromMesh(mesh);
  const zs = colZsFromMesh(mesh);
  const y = Number(zy[1]);
  const z = Number(zy[0]);
  let row = -1;
  let col = -1;
  for (let i = 0; i < ys.length - 1; i++) {
    const hi = Math.max(ys[i], ys[i + 1]);
    const lo = Math.min(ys[i], ys[i + 1]);
    if (y <= hi + 1e-4 && y >= lo - 1e-4) { row = i; break; }
  }
  for (let j = 0; j < zs.length - 1; j++) {
    const hi = Math.max(zs[j], zs[j + 1]);
    const lo = Math.min(zs[j], zs[j + 1]);
    if (z <= hi + 1e-4 && z >= lo - 1e-4) { col = j; break; }
  }
  if (row < 0 || col < 0) return null;
  return { row, col, box_id: `r${row}c${col}` };
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
      cells.push({
        row: i,
        col: j,
        box_id: `r${i}c${j}`,
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
