"""可选视频路径：只允许 output/dualcam 与 data 下的 mp4。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dualcam_videos import list_videos, resolve_video


def test_resolve_rejects_escape(tmp_path: Path):
    (tmp_path / "output" / "dualcam").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    good = tmp_path / "output" / "dualcam" / "src.mp4"
    good.write_bytes(b"x")
    assert resolve_video("output/dualcam/src.mp4", tmp_path) == good.resolve()
    assert resolve_video("../etc/passwd", tmp_path) is None
    assert resolve_video("/etc/passwd.mp4", tmp_path) is None
    assert resolve_video("output/dualcam/missing.mp4", tmp_path) is None


def test_list_only_mp4_under_roots(tmp_path: Path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "a.mp4").write_bytes(b"a")
    (d / "note.txt").write_text("no", encoding="utf-8")
    (tmp_path / "output" / "dualcam").mkdir(parents=True)
    paths = [v["path"] for v in list_videos(tmp_path)]
    assert paths == ["data/a.mp4"]
