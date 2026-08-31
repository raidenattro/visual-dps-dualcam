"""操作审计写入。"""

from __future__ import annotations

from fastapi import Request

from services.log_store import insert_audit


def _client_ip(request: Request | None) -> str:
    if request is None:
        return ""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host or ""
    return ""


def _actor(request: Request | None) -> str:
    if request is None:
        return "system"
    user = getattr(request.state, "user", None) or {}
    return str(user.get("username") or "system")


def record_audit(
    request: Request | None,
    action: str,
    *,
    resource_type: str = "",
    resource_id: str = "",
    result: str = "success",
    detail: dict | None = None,
) -> None:
    try:
        insert_audit(
            actor=_actor(request),
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            result=result,
            detail=detail,
            ip=_client_ip(request),
        )
    except Exception:
        pass


def audit_from_result(request: Request | None, action: str, resource_type: str, resource_id: str, data: dict):
    result = "success" if not data.get("error") else "error"
    detail = {"error": data.get("error")} if data.get("error") else {}
    record_audit(request, action, resource_type=resource_type, resource_id=resource_id, result=result, detail=detail)
