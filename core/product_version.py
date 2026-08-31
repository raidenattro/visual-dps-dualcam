"""产品版本号（与 web/version.json 同步，由 scripts/bump-build.mjs 维护）。"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VERSION_CANDIDATES = (
    Path(os.environ.get("VISUAL_DPS_VERSION_FILE", "")),
    _REPO_ROOT / "version.json",
    _REPO_ROOT / "web" / "version.json",
)

# 构建脚本 tag：YYYYMMDD-HHMMSS-<git短哈希>
_DATED_TAG_RE = re.compile(r"^(\d{8})-(\d{6})-([0-9a-f]+)$", re.I)
_HEX_ID_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{12,64}$", re.I)


@lru_cache(maxsize=1)
def load_version_dict() -> dict:
    for path in _VERSION_CANDIDATES:
        if not path or not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            continue
    return {"major": 0, "minor": 0, "patch": 0, "build": 0}


def format_product_version(data: dict | None = None) -> str:
    v = data if isinstance(data, dict) else load_version_dict()
    major = int(v.get("major", 0))
    minor = int(v.get("minor", 0))
    patch = int(v.get("patch", 0))
    build = int(v.get("build", 0))
    return f"v{major}.{minor}.{patch}.build{build}"


def split_image_ref(image: str) -> tuple[str, str]:
    """返回 (repository, tag)；无 tag 时 tag 为 latest。"""
    s = str(image or "").strip()
    if not s or s == "—":
        return "", ""
    if "@" in s:
        s = s.split("@", 1)[0].strip()
    if ":" in s and not s.startswith("sha256:"):
        repo, tag = s.rsplit(":", 1)
        return repo.strip(), tag.strip() or "latest"
    return s, "latest"


def is_opaque_image_id(value: str) -> bool:
    """是否为 Docker 镜像 ID（非人类可读 tag）。"""
    s = str(value or "").strip()
    if not s:
        return False
    if "@" in s:
        s = s.split("@", 1)[1].strip()
    if s.startswith("sha256:"):
        return True
    if _HEX_ID_RE.match(s):
        return True
    if ":" not in s:
        return False
    _repo, tag = split_image_ref(s)
    if not tag or tag == "latest":
        return False
    return bool(_HEX_ID_RE.match(tag))


def format_build_tag(tag: str) -> str:
    """20260528-161546-f10ac06 → 2026-05-28 16:15:46 (f10ac06)。"""
    raw = str(tag or "").strip()
    if not raw:
        return ""
    m = _DATED_TAG_RE.match(raw)
    if not m:
        return raw[:48]
    d, t, git = m.group(1), m.group(2), m.group(3)
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]} {t[0:2]}:{t[2:4]}:{t[4:6]} ({git})"


def format_image_version_display(image: str) -> str:
    """页脚等展示：优先可读构建 tag，避免裸镜像 ID。"""
    ref = str(image or "").strip()
    if not ref or ref == "—":
        return "—"
    if is_opaque_image_id(ref):
        return "未标定"

    _repo, tag = split_image_ref(ref)
    if not tag:
        return "—"
    if tag == "latest":
        env_tag = os.environ.get("VISUAL_DPS_IMAGE_TAG", "").strip()
        if env_tag and env_tag != "latest" and _DATED_TAG_RE.match(env_tag):
            return format_build_tag(env_tag)
        return "latest"
    if is_opaque_image_id(tag):
        return "未标定"
    return format_build_tag(tag) or tag


def short_image_ref(image: str) -> str:
    """兼容旧调用：等同 format_image_version_display。"""
    return format_image_version_display(image)
