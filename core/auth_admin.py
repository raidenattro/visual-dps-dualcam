"""管理员校验与用户角色。"""

from __future__ import annotations

from fastapi import HTTPException, Request

from core.auth_settings import load_auth_settings
from services.auth_service import get_user_record


def is_admin_user(users_file: str, username: str) -> bool:
    rec = get_user_record(users_file, username)
    if not rec:
        return False
    role = str(rec.get("role") or "").strip().lower()
    if role == "admin":
        return True
    return username == "admin" and not role


def enrich_session_user(user: dict, users_file: str) -> dict:
    if not user or not user.get("username"):
        return user
    rec = get_user_record(users_file, user["username"])
    role = "operator"
    if rec:
        role = str(rec.get("role") or ("admin" if user["username"] == "admin" else "operator")).lower()
    elif user["username"] == "admin":
        role = "admin"
    out = dict(user)
    out["role"] = role
    out["is_admin"] = role == "admin"
    return out


def require_admin(request: Request, app_config: dict | None = None) -> dict:
    settings = load_auth_settings(app_config)
    user = getattr(request.state, "user", None)
    if not user or not user.get("username"):
        raise HTTPException(status_code=401, detail="未登录或会话已过期")
    users_file = settings["local"]["users_file"]
    enriched = enrich_session_user(user, users_file)
    if not enriched.get("is_admin"):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return enriched
