"""管理员：日志查询、用户管理、全局设置。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.auth_admin import require_admin
from core.auth_settings import load_auth_settings
from services.audit_service import record_audit
from services.auth_service import create_user, delete_user, get_user, list_users, update_user
from services.log_store import query_audit_logs, query_event_logs
from services.runtime_config_service import get_public_settings, patch_public_settings


class UserCreateBody(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)
    display_name: str = Field("", max_length=128)
    role: str = Field("operator", max_length=32)


class UserUpdateBody(BaseModel):
    display_name: str | None = None
    role: str | None = None
    password: str | None = Field(None, min_length=6, max_length=128)


def register_admin_routes(router: APIRouter, app_config: dict | None = None) -> None:
    auth_cfg = load_auth_settings(app_config)

    @router.get("/settings")
    async def read_settings(request: Request):
        require_admin(request, app_config)
        return get_public_settings(app_config)

    @router.patch("/settings")
    async def update_settings(request: Request, body: dict):
        require_admin(request, app_config)
        data = patch_public_settings(body)
        if data.get("status") != "success":
            record_audit(request, "settings.update", result="error", detail=data)
            return data
        record_audit(request, "settings.update", detail={"applied": data.get("applied")})
        return get_public_settings(app_config)

    @router.get("/logs/audit")
    async def logs_audit(
        request: Request,
        page: int = 1,
        page_size: int = 50,
        actor: str = "",
        action: str = "",
        result: str = "",
        resource_id: str = "",
        sort_by: str = "ts",
        sort_order: str = "desc",
    ):
        require_admin(request, app_config)
        return query_audit_logs(
            page=page,
            page_size=page_size,
            actor=actor,
            action=action,
            result=result,
            resource_id=resource_id,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    @router.get("/logs/events")
    async def logs_events(
        request: Request,
        page: int = 1,
        page_size: int = 50,
        event_type: str = "",
        camera_id: str = "",
        severity: str = "",
        summary: str = "",
        sort_by: str = "ts",
        sort_order: str = "desc",
    ):
        require_admin(request, app_config)
        return query_event_logs(
            page=page,
            page_size=page_size,
            event_type=event_type,
            camera_id=camera_id,
            severity=severity,
            summary=summary,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    @router.get("/users")
    async def users_list(request: Request):
        require_admin(request, app_config)
        return {"status": "success", "items": list_users(auth_cfg["local"]["users_file"])}

    @router.get("/users/{username}")
    async def users_get(request: Request, username: str):
        require_admin(request, app_config)
        found = get_user(auth_cfg["local"]["users_file"], username)
        if not found:
            return {"status": "error", "error": "用户不存在"}
        return {"status": "success", "user": found}

    @router.post("/users")
    async def users_create(request: Request, body: UserCreateBody):
        require_admin(request, app_config)
        user, err = create_user(
            auth_cfg["local"]["users_file"],
            body.username,
            body.password,
            body.display_name,
            body.role,
        )
        if err:
            record_audit(request, "user.create", resource_type="user", resource_id=body.username, result="error", detail={"error": err})
            return {"status": "error", "error": err}
        record_audit(request, "user.create", resource_type="user", resource_id=body.username, detail={"role": body.role})
        return {"status": "success", "user": user}

    @router.patch("/users/{username}")
    async def users_update(request: Request, username: str, body: UserUpdateBody):
        require_admin(request, app_config)
        user, err = update_user(
            auth_cfg["local"]["users_file"],
            username,
            display_name=body.display_name,
            role=body.role,
            password=body.password,
        )
        if err:
            record_audit(request, "user.update", resource_type="user", resource_id=username, result="error", detail={"error": err})
            return {"status": "error", "error": err}
        record_audit(request, "user.update", resource_type="user", resource_id=username)
        return {"status": "success", "user": user}

    @router.delete("/users/{username}")
    async def users_delete(request: Request, username: str):
        require_admin(request, app_config)
        err = delete_user(auth_cfg["local"]["users_file"], username)
        if err:
            record_audit(request, "user.delete", resource_type="user", resource_id=username, result="error", detail={"error": err})
            return {"status": "error", "error": err}
        record_audit(request, "user.delete", resource_type="user", resource_id=username)
        return {"status": "success"}
