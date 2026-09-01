/** COCO-17 骨架连线（与 Monitor 页一致） */
export const COCO_LINES = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
  [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
];

/** 实验仓 dualcam_player / aisle3d_viewer 的 3D 骨线（含鼻-肩，不含过长五官链也可被长度过滤） */
export const AISLE_3D_EDGES = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 5], [0, 6],
];

/** 人体骨段上限（米）。单路贴墙预览超出此长度的边不画，避免「射向货架」的长线 */
export const MAX_BONE_M = 0.85;

export const SKELETON_CONF = 0.2;

export function scaleInferPoint(x, y, inferW, inferH, frameW, frameH) {
  const iw = inferW > 0 ? inferW : frameW;
  const ih = inferH > 0 ? inferH : frameH;
  if (!iw || !ih || !frameW || !frameH) return [x, y];
  return [(x / iw) * frameW, (y / ih) * frameH];
}
