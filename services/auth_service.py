"""本地账号、会话 Cookie、OAuth2 登录。"""

from __future__ import annotations

import json
import os
import secrets
import time
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException, Request, Response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from passlib.context import CryptContext

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
_oauth_states: dict[str, float] = {}
_STATE_TTL = 600


def _serializer(secret: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret, salt="visual-dps-session")


def ensure_users_file(users_file: str, default_password: str | None = None) -> None:
    if os.path.isfile(users_file):
        return
    os.makedirs(os.path.dirname(users_file) or ".", exist_ok=True)
    pwd = default_password or os.environ.get("AUTH_DEFAULT_PASSWORD", "admin123")
    payload = {
        "users": [
            {
                "username": "admin",
                "display_name": "管理员",
                "role": "admin",
                "password_hash": _pwd.hash(pwd),
            }
        ]
    }
    with open(users_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_users(users_file: str) -> list[dict]:
    ensure_users_file(users_file)
    with open(users_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    users = data.get("users") if isinstance(data, dict) else []
    return users if isinstance(users, list) else []


def verify_local_login(users_file: str, username: str, password: str) -> dict | None:
    name = (username or "").strip()
    if not name or not password:
        return None
    for user in _load_users(users_file):
        if str(user.get("username", "")).strip() != name:
            continue
        hashed = user.get("password_hash") or ""
        if hashed and _pwd.verify(password, hashed):
            role = str(user.get("role") or ("admin" if name == "admin" else "operator")).lower()
            return {
                "username": name,
                "display_name": user.get("display_name") or name,
                "role": role,
                "is_admin": role == "admin",
                "auth_method": "local",
            }
    return None


def get_user_record(users_file: str, username: str) -> dict | None:
    name = (username or "").strip()
    for user in _load_users(users_file):
        if str(user.get("username", "")).strip() == name:
            return user
    return None


def _public_user(user: dict) -> dict:
    name = str(user.get("username", "")).strip()
    role = str(user.get("role") or ("admin" if name == "admin" else "operator")).lower()
    return {
        "username": name,
        "display_name": user.get("display_name") or name,
        "role": role,
        "is_admin": role == "admin",
    }


def list_users(users_file: str) -> list[dict]:
    return [_public_user(u) for u in _load_users(users_file)]


def get_user(users_file: str, username: str) -> dict | None:
    rec = get_user_record(users_file, username)
    return _public_user(rec) if rec else None


def _write_users(users_file: str, users: list[dict]) -> None:
    os.makedirs(os.path.dirname(users_file) or ".", exist_ok=True)
    with open(users_file, "w", encoding="utf-8") as f:
        json.dump({"users": users}, f, ensure_ascii=False, indent=2)


def create_user(
    users_file: str, username: str, password: str, display_name: str = "", role: str = "operator"
) -> tuple[dict | None, str | None]:
    name = (username or "").strip()
    if not name:
        return None, "用户名不能为空"
    if role not in ("admin", "operator"):
        return None, "角色无效"
    users = _load_users(users_file)
    if any(str(u.get("username", "")).strip() == name for u in users):
        return None, "用户名已存在"
    users.append(
        {
            "username": name,
            "display_name": (display_name or "").strip() or name,
            "role": role,
            "password_hash": _pwd.hash(password),
        }
    )
    _write_users(users_file, users)
    return _public_user(users[-1]), None


def update_user(
    users_file: str,
    username: str,
    *,
    display_name: str | None = None,
    role: str | None = None,
    password: str | None = None,
) -> tuple[dict | None, str | None]:
    name = (username or "").strip()
    users = _load_users(users_file)
    idx = next((i for i, u in enumerate(users) if str(u.get("username", "")).strip() == name), -1)
    if idx < 0:
        return None, "用户不存在"
    if role is not None:
        if role not in ("admin", "operator"):
            return None, "角色无效"
        users[idx]["role"] = role
    if display_name is not None:
        users[idx]["display_name"] = display_name.strip() or name
    if password:
        users[idx]["password_hash"] = _pwd.hash(password)
    _write_users(users_file, users)
    return _public_user(users[idx]), None


def delete_user(users_file: str, username: str) -> str | None:
    name = (username or "").strip()
    users = _load_users(users_file)
    admins = [
        u
        for u in users
        if str(u.get("role", "")).lower() == "admin" or str(u.get("username", "")).strip() == "admin"
    ]
    target = get_user_record(users_file, name)
    if not target:
        return "用户不存在"
    is_target_admin = str(target.get("role", "")).lower() == "admin" or name == "admin"
    if is_target_admin and len(admins) <= 1:
        return "不能删除最后一个管理员"
    new_users = [u for u in users if str(u.get("username", "")).strip() != name]
    _write_users(users_file, new_users)
    return None


def issue_session_cookie(response: Response, settings: dict, user: dict) -> None:
    payload = {
        "username": user["username"],
        "display_name": user.get("display_name") or user["username"],
        "role": user.get("role", "operator"),
        "is_admin": bool(user.get("is_admin")),
        "auth_method": user.get("auth_method", "local"),
    }
    token = _serializer(settings["session_secret"]).dumps(payload)
    max_age = int(settings["session_ttl_seconds"])
    response.set_cookie(
        key=settings["cookie_name"],
        value=token,
        max_age=max_age,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )


def clear_session_cookie(response: Response, settings: dict) -> None:
    response.delete_cookie(settings["cookie_name"], path="/")


def get_session_user(request: Request, settings: dict) -> dict | None:
    token = request.cookies.get(settings["cookie_name"])
    if not token:
        return None
    try:
        data = _serializer(settings["session_secret"]).loads(
            token, max_age=int(settings["session_ttl_seconds"])
        )
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(data, dict) or not data.get("username"):
        return None
    return data


def require_user(request: Request, settings: dict) -> dict:
    user = get_session_user(request, settings)
    if not user:
        raise HTTPException(status_code=401, detail="未登录或会话已过期")
    return user


def _public_base_url(request: Request) -> str:
    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_host = request.headers.get("x-forwarded-host")
    if forwarded_proto and forwarded_host:
        return f"{forwarded_proto}://{forwarded_host}"
    return str(request.base_url).rstrip("/")


def oauth2_redirect_uri(request: Request, settings: dict) -> str:
    path = settings["oauth2"]["redirect_path"]
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return f"{_public_base_url(request)}{path}"


def create_oauth_state() -> str:
    _purge_oauth_states()
    state = secrets.token_urlsafe(24)
    _oauth_states[state] = time.time() + _STATE_TTL
    return state


def validate_oauth_state(state: str | None) -> bool:
    if not state or state not in _oauth_states:
        return False
    expires = _oauth_states.pop(state, 0)
    return time.time() <= expires


def _purge_oauth_states() -> None:
    now = time.time()
    expired = [k for k, v in _oauth_states.items() if v < now]
    for k in expired:
        _oauth_states.pop(k, None)


def build_oauth_authorize_url(request: Request, settings: dict, state: str) -> str:
    oauth = settings["oauth2"]
    query = urlencode(
        {
            "response_type": "code",
            "client_id": oauth["client_id"],
            "redirect_uri": oauth2_redirect_uri(request, settings),
            "scope": oauth["scopes"],
            "state": state,
        }
    )
    sep = "&" if "?" in oauth["authorize_url"] else "?"
    return f"{oauth['authorize_url']}{sep}{query}"


async def exchange_oauth_code(request: Request, settings: dict, code: str) -> dict:
    oauth = settings["oauth2"]
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": oauth2_redirect_uri(request, settings),
        "client_id": oauth["client_id"],
        "client_secret": oauth["client_secret"],
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        token_resp = await client.post(
            oauth["token_url"],
            data=data,
            headers={"Accept": "application/json"},
        )
        if token_resp.status_code >= 400:
            raise HTTPException(status_code=400, detail="单点登录令牌交换失败")
        token_json = token_resp.json()
        access_token = token_json.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="单点登录未返回访问令牌")

        profile: dict[str, Any] = {}
        if oauth.get("userinfo_url"):
            ui_resp = await client.get(
                oauth["userinfo_url"],
                headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            )
            if ui_resp.status_code < 400:
                profile = ui_resp.json()

    username = (
        profile.get(oauth["username_claim"])
        or profile.get("sub")
        or profile.get(oauth["email_claim"])
        or profile.get("email")
    )
    if not username:
        raise HTTPException(status_code=400, detail="无法识别单点登录用户")
    username = str(username).strip()
    return {
        "username": username,
        "display_name": profile.get("name") or profile.get("display_name") or username,
        "role": "operator",
        "is_admin": False,
        "auth_method": "oauth2",
    }
