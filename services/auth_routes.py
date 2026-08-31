"""认证 API：本地登录、OAuth2、会话。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from core.auth_admin import enrich_session_user
from core.auth_settings import load_auth_settings, oauth2_public_config
from services.audit_service import record_audit
from services.auth_service import (
    build_oauth_authorize_url,
    clear_session_cookie,
    create_oauth_state,
    exchange_oauth_code,
    get_session_user,
    issue_session_cookie,
    update_user,
    validate_oauth_state,
    verify_local_login,
)


class LoginBody(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=256)


class PasswordChangeBody(BaseModel):
    old_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6, max_length=128)


def register_auth_routes(router: APIRouter, app_config: dict | None = None) -> None:
    settings_holder = {"settings": load_auth_settings(app_config)}

    def settings() -> dict:
        return settings_holder["settings"]

    @router.get("/auth/config")
    async def auth_config():
        s = settings()
        if not s["enabled"]:
            return {"enabled": False, "local": {"enabled": False}, "oauth2": None}
        return {
            "enabled": True,
            "local": {"enabled": bool(s["local"]["enabled"])},
            "oauth2": oauth2_public_config(s),
        }

    @router.get("/auth/me")
    async def auth_me(request: Request):
        s = settings()
        if not s["enabled"]:
            return {"authenticated": True, "user": {"username": "guest", "display_name": "访客"}}
        user = get_session_user(request, s)
        if not user:
            return {"authenticated": False}
        enriched = enrich_session_user(user, s["local"]["users_file"])
        return {"authenticated": True, "user": enriched}

    @router.post("/auth/login")
    async def auth_login(body: LoginBody, response: Response, request: Request):
        s = settings()
        if not s["enabled"]:
            return {"status": "success", "user": {"username": "guest"}}
        if not s["local"]["enabled"]:
            raise HTTPException(status_code=400, detail="未启用本地登录")
        user = verify_local_login(s["local"]["users_file"], body.username, body.password)
        if not user:
            record_audit(request, "auth.login", resource_type="user", resource_id=body.username, result="error")
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        issue_session_cookie(response, s, user)
        record_audit(request, "auth.login", resource_type="user", resource_id=user["username"])
        return {"status": "success", "user": user}

    @router.post("/auth/password")
    async def change_password(body: PasswordChangeBody, request: Request):
        s = settings()
        user = get_session_user(request, s)
        if not user:
            raise HTTPException(status_code=401, detail="未登录")
        name = user["username"]
        if not verify_local_login(s["local"]["users_file"], name, body.old_password):
            record_audit(request, "user.password_change", resource_type="user", resource_id=name, result="error")
            raise HTTPException(status_code=400, detail="原密码错误")
        updated, err = update_user(s["local"]["users_file"], name, password=body.new_password)
        if err:
            return {"status": "error", "error": err}
        record_audit(request, "user.password_change", resource_type="user", resource_id=name)
        return {"status": "success", "user": enrich_session_user(updated, s["local"]["users_file"])}

    @router.post("/auth/logout")
    async def auth_logout(response: Response, request: Request):
        s = settings()
        user = get_session_user(request, s)
        clear_session_cookie(response, s)
        if user:
            record_audit(request, "auth.logout", resource_type="user", resource_id=user.get("username", ""))
        return {"status": "success"}

    @router.get("/auth/oauth2/login")
    async def oauth2_login(request: Request):
        s = settings()
        if not s["enabled"] or not oauth2_public_config(s):
            raise HTTPException(status_code=404, detail="未配置单点登录")
        state = create_oauth_state()
        url = build_oauth_authorize_url(request, s, state)
        resp = RedirectResponse(url=url, status_code=302)
        resp.set_cookie(
            "visual_dps_oauth_state",
            state,
            max_age=600,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return resp

    @router.get("/auth/oauth2/callback")
    async def oauth2_callback(
        request: Request,
        response: Response,
        code: str | None = None,
        state: str | None = None,
        error: str | None = None,
    ):
        s = settings()
        if error:
            return RedirectResponse(url=f"/login?error={error}", status_code=302)
        if not code:
            return RedirectResponse(url="/login?error=missing_code", status_code=302)
        cookie_state = request.cookies.get("visual_dps_oauth_state")
        if not validate_oauth_state(state) or state != cookie_state:
            return RedirectResponse(url="/login?error=invalid_state", status_code=302)
        user = await exchange_oauth_code(request, s, code)
        issue_session_cookie(response, s, user)
        redirect = RedirectResponse(url="/", status_code=302)
        redirect.delete_cookie("visual_dps_oauth_state", path="/")
        return redirect
