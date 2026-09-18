"""推理 bind mount 清单须与 inference_container_service 一致。"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bind_list_from_inference_container_service() -> list[str]:
    text = (ROOT / "services/inference_container_service.py").read_text(encoding="utf-8")
    m = re.search(r"for rel in \(\s*(.*?)\s*\):\s*\n\s*binds\.append", text, re.S)
    assert m, "未找到 inference bind mount 列表"
    items: list[str] = []
    for line in m.group(1).splitlines():
        line = line.strip().rstrip(",").strip('"').strip("'")
        if line and not line.startswith("#"):
            items.append(line)
    return ["core/config.py", "core/ort_runtime.py", *items]


def _bind_list_from_infer_bind_mounts_sh() -> list[str]:
    text = (ROOT / "scripts/deploy-only/infer-bind-mounts.sh").read_text(encoding="utf-8")
    block = text.split("INFER_BIND_MOUNT_FILES=(")[1].split(")")[0]
    items: list[str] = []
    for line in block.splitlines():
        line = line.strip().rstrip(",").strip('"').strip("'")
        if line and not line.startswith("#"):
            items.append(line)
    return items


def test_infer_bind_mount_lists_match():
    from_service = _bind_list_from_inference_container_service()
    from_script = _bind_list_from_infer_bind_mounts_sh()
    assert set(from_service) == set(from_script), (
        f"infer-bind-mounts.sh 与 inference_container_service 不一致:\n"
        f"  仅 service: {set(from_service) - set(from_script)}\n"
        f"  仅 script: {set(from_script) - set(from_service)}"
    )
