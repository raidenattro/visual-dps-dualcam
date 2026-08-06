from adapters.review_labels import (
    camera_review_aliases,
    normalize_box_token,
    normalize_verified_true,
)


def test_normalize_box_token():
    assert normalize_box_token("Box_2093") == "Box_2093"
    assert normalize_box_token("79:1024") == "Box_1024"
    assert normalize_box_token("1024") == "Box_1024"


def test_bindings_schema2():
    raw = [
        {
            "frame_idx": 10,
            "source_frame_idx": 10,
            "detected_box_tokens": ["Box_1", "Box_2"],
            "bindings": [
                {
                    "confirmed_box_tokens": ["Box_2"],
                    "person_id": 0,
                    "person_track_id": "2",
                }
            ],
        }
    ]
    out = normalize_verified_true(raw)
    assert len(out) == 1
    assert out[0]["confirmed_box_tokens"] == ["Box_2"]
    assert out[0]["person_track_ids"] == ["2"]


def test_camera_aliases():
    al = camera_review_aliases("1-6-2-(4)")
    assert "1-6-2-(4)" in al
    assert "1-6-2-_4" in al
    al2 = camera_review_aliases("1-6-2-_4")
    assert "1-6-2-(4)" in al2
