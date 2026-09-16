/** COCO-17 骨架连线（与 Monitor 页一致） */
export const COCO_LINES = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
  [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
];

/** 巷道 3D：不画 COCO 0–4（鼻/眼/耳），避免抬升异常时头点贴地干扰观感 */
export const AISLE_3D_HEAD_JOINTS = new Set([0, 1, 2, 3, 4]);

export const AISLE_3D_EDGES = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
];

export const SKELETON_CONF = 0.2;

export function scaleInferPoint(x, y, inferW, inferH, frameW, frameH) {
  const iw = inferW > 0 ? inferW : frameW;
  const ih = inferH > 0 ? inferH : frameH;
  if (!iw || !ih || !frameW || !frameH) return [x, y];
  return [(x / iw) * frameW, (y / ih) * frameH];
}
