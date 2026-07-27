"""Depth Anything V2 封装。

输出是**相对逆深度**：值越大越近。绝对尺度无意义，所以下游一律做同帧内归一化，
不跨帧比较——这样即使模型的 scale/shift 逐帧漂移也不影响特征。

依赖 torch/transformers，见 requirements-depth.txt。权重走 HF_ENDPOINT 镜像。
"""

from __future__ import annotations

import os

import numpy as np

DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        hf_endpoint: str = "https://hf-mirror.com",
    ):
        if hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()

    def infer_bgr(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        """输入 BGR uint8 帧列表（同尺寸），返回 (N, H, W) float32 逆深度。"""
        if not frames_bgr:
            return np.zeros((0, 0, 0), dtype=np.float32)
        h, w = frames_bgr[0].shape[:2]
        rgb = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames_bgr]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model(**inputs).predicted_depth
        resized = self._torch.nn.functional.interpolate(
            out.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze(1)
        return resized.float().cpu().numpy()
