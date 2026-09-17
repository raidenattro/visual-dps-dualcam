"""标注 JSON 文件名与 camera.path 对齐。"""

import json
import os

from services.camera_annotation_paths import camera_annotation_path, existing_camera_annotation_path


def test_path_based_filename(tmp_path):
    json_dir = str(tmp_path / "json")
    cam_file = tmp_path / "camera_ips.json"
    cam_file.write_text(
        json.dumps([{"id": "7", "path": "aisle4-L", "name": "L"}]),
        encoding="utf-8",
    )
    os.makedirs(json_dir + "/cameras", exist_ok=True)
    target = camera_annotation_path(json_dir, "7", camera_ips_file=str(cam_file))
    assert target.endswith(os.path.join("cameras", "aisle4-L.json"))
    legacy = os.path.join(json_dir, "cameras", "7.json")
    with open(legacy, "w", encoding="utf-8") as f:
        f.write("{}")
    assert existing_camera_annotation_path(json_dir, "7", camera_ips_file=str(cam_file)) == legacy
