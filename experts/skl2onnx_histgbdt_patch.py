"""skl2onnx 导出 HistGradientBoosting 时叶节点 bool 写入 ONNX int 字段的补丁。"""

from __future__ import annotations

_PATCHED = False


def apply_histgbdt_skl2onnx_patch() -> None:
    """HistGBDT 叶节点 missing=False 须转为 0，否则 onnx 1.22 序列化失败。"""
    global _PATCHED
    if _PATCHED:
        return
    import skl2onnx.common.tree_ensemble as te

    orig = te.add_node

    def add_node_patched(*args, **kwargs):
        if "nodes_missing_value_tracks_true" in kwargs:
            kwargs["nodes_missing_value_tracks_true"] = int(
                kwargs["nodes_missing_value_tracks_true"]
            )
        return orig(*args, **kwargs)

    te.add_node = add_node_patched
    _PATCHED = True
