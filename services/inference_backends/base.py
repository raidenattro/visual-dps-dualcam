"""推理后端公共数据结构（与具体模型实现解耦）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PoseBatch:
    """单帧多人姿态，keypoints 为 COCO-17 格式（与现有碰撞逻辑一致）。"""

    keypoints: np.ndarray
    keypoint_scores: np.ndarray

    @property
    def num_persons(self) -> int:
        if self.keypoints.size == 0:
            return 0
        return int(self.keypoints.shape[0])

    @classmethod
    def empty(cls) -> PoseBatch:
        return cls(
            keypoints=np.empty((0, 17, 2), dtype=np.float32),
            keypoint_scores=np.empty((0, 17), dtype=np.float32),
        )
