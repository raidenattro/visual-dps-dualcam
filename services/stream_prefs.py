"""实时预览流参数校验。"""

STREAM_HEIGHT_MIN = 320
STREAM_HEIGHT_MAX = 720
STREAM_HEIGHT_DEFAULT = 480
STREAM_HEIGHT_CHOICES = (320, 480, 720)


def clamp_stream_height(value) -> int:
    try:
        h = int(value)
    except (TypeError, ValueError):
        h = STREAM_HEIGHT_DEFAULT
    return max(STREAM_HEIGHT_MIN, min(STREAM_HEIGHT_MAX, h))
