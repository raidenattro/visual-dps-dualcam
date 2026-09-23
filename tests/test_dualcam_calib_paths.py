"""144/24 标定路径与误存迁移。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dualcam_calib_paths import (
    CALIB_144_24,
    CALIB_ID,
    is_144_24_payload,
    stamp_144_24,
    wall2_complete,
)


def test_stamp_144_24():
    out = stamp_144_24({"aisle": 2})
    assert out["calib_id"] == CALIB_ID
    assert out["video_sources"]["L"] == "data/test-144.mp4"


def test_skel_file_follows_video_stem():
    from scripts.dualcam_calib_paths import SKEL_144_24, skel_file
    assert skel_file({"mode": "stitched", "src": "output/dualcam/src.mp4"}) == SKEL_144_24
    other = skel_file({"mode": "stitched", "src": "output/dualcam/test2.mp4"})
    assert other.name == "skel3d_test2.json"
    split = skel_file({"mode": "split", "L": "data/a.mp4", "R": "data/b.mp4"})
    assert split.name == "skel3d_a__b.json"


def test_calib_file_keeps_144_and_splits_others():
    from scripts.dualcam_calib_paths import calib_file, CALIB_144_24
    assert calib_file({"mode": "stitched", "src": "output/dualcam/src.mp4"}) == CALIB_144_24
    other = calib_file({"mode": "stitched", "src": "output/dualcam/new_aisle.mp4"})
    assert other.name == "dual_new_aisle.json"
    assert other != CALIB_144_24
    out = stamp_144_24({
        "video_sources": {"mode": "split", "src": "data/left.mp4", "L": "data/left.mp4", "R": "data/right.mp4"},
    })
    assert out["video_sources"]["mode"] == "split"
    assert out["video_sources"]["L"] == "data/left.mp4"
    assert out["video_sources"]["R"] == "data/right.mp4"
    assert is_144_24_payload({"calib_id": "dual_144-24", "views": {}})
    assert is_144_24_payload({
        "layout": "same_side",
        "prior": {"wallDist82": 0.7},
        "views": {"L": {}, "R": {}},
    })


def test_migrated_file_on_disk():
    if not CALIB_144_24.is_file():
        return
    data = json.loads(CALIB_144_24.read_text(encoding="utf-8"))
    assert data.get("calib_id") == CALIB_ID
    assert wall2_complete(data), "144/24 标定应含两路墙2 四角"
