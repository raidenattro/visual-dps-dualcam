"""SSE 合并必须带上 event.persons_3d，否则巷道直播只有 2D 骨架、没有 3D 姿态。"""

from services.live_bus import merge_live_frame


def test_merge_live_frame_keeps_stereo_persons_3d():
    pose = {"ts": 2.0, "persons": [{"id": 1}], "infer_width": 640, "infer_height": 360}
    event = {
        "ts": 1.5,
        "skeletons": [{"id": 1}],
        "persons_3d": [{"xyz": [[0, 1, 2]] * 17, "preview": False}],
        "collisions": [],
        "alarm_collisions": [],
    }
    merged = merge_live_frame(pose, event)
    assert len(merged["skeletons"]) == 1
    assert merged["persons_3d"][0]["preview"] is False
    assert merged["persons_3d"][0]["xyz"][0] == [0, 1, 2]


def test_merge_live_frame_empty_event_has_persons_3d_key():
    merged = merge_live_frame({"ts": 1, "persons": []}, None)
    assert merged["persons_3d"] == []


def test_merge_live_frame_empty_persons_3d_clears_skeleton():
    pose = {"ts": 3.0, "persons": []}
    event = {"ts": 3.0, "persons_3d": [], "collisions": [], "alarm_collisions": []}
    merged = merge_live_frame(pose, event)
    assert merged["persons_3d"] == []
    assert merged["skeletons"] == []
