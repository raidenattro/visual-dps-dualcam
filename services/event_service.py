"""业务事件写入（告警、回调、启停）。"""

from __future__ import annotations

from services.log_store import insert_event


def camera_id_from_upload_tag(upload_tag: str) -> str:
    tag = str(upload_tag or "").strip()
    if tag.startswith("infer_"):
        return tag[6:]
    return tag


def record_event(
    event_type: str,
    *,
    camera_id: str = "",
    severity: str = "info",
    summary: str = "",
    detail: dict | None = None,
) -> None:
    try:
        insert_event(
            event_type=event_type,
            camera_id=camera_id,
            severity=severity,
            summary=summary,
            detail=detail,
        )
    except Exception:
        pass
