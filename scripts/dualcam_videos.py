#!/usr/bin/env python3
"""双路标注可选的 mp4：output/dualcam 与 data 下的文件，禁止跳出这两处。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOTS = (
    ROOT / "output" / "dualcam",
    ROOT / "data",
)
DEFAULT_SRC = "output/dualcam/src.mp4"


def list_videos(root: Path | None = None) -> list[dict]:
    """按相对路径列出可选 mp4。root 仅用于测试，正式调用扫 VIDEO_ROOTS。"""
    bases = _roots(root)
    found: list[dict] = []
    for base in bases:
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*.mp4")):
            if not path.is_file():
                continue
            rel = path.resolve().relative_to(_repo(root)).as_posix()
            found.append({
                "path": rel,
                "label": rel,
                "bytes": path.stat().st_size,
            })
    return found


def resolve_video(src: str | None, root: Path | None = None) -> Path | None:
    """把标注里存的相对路径收成磁盘文件。绝对路径、..、非 mp4 一律拒绝。"""
    raw = (src or "").strip() or DEFAULT_SRC
    rel = Path(raw)
    if rel.is_absolute() or ".." in rel.parts:
        return None
    if rel.suffix.lower() != ".mp4":
        return None
    repo = _repo(root)
    path = (repo / rel).resolve()
    for base in _roots(root):
        try:
            path.relative_to(base.resolve())
        except ValueError:
            continue
        if path.is_file():
            return path
        return None
    return None


def _repo(root: Path | None) -> Path:
    return Path(root) if root is not None else ROOT


def _roots(root: Path | None) -> tuple[Path, ...]:
    if root is None:
        return VIDEO_ROOTS
    base = Path(root)
    return (base / "output" / "dualcam", base / "data")
