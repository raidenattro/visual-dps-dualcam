"""模拟线上抽帧处理。

离线数据是逐帧的（约 25fps），线上按更低帧率抽帧后才送推理。
时序特征、分数平滑、连续帧告警都是逐帧推进的，所以抽帧必须在喂进 pipeline 之前做，
不能在全帧结果上后处理。
"""

from __future__ import annotations

from collections.abc import Iterable


def sample_frame_indices(
    indices: Iterable[int], *, source_fps: float, target_fps: float
) -> list[int]:
    """按目标帧率均匀抽样有序帧号。25→15 得到每 5 帧留 3 帧。"""
    idx = sorted(int(i) for i in indices)
    if target_fps <= 0 or source_fps <= 0 or target_fps >= source_fps:
        return idx
    num = int(round(target_fps * 1000))
    den = int(round(source_fps * 1000))
    keep: list[int] = []
    prev = -1
    for i, frame in enumerate(idx):
        cur = (i * num) // den
        if cur != prev:
            keep.append(frame)
            prev = cur
    return keep
