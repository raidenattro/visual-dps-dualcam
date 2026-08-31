"""认证配置加载（app_config.auth 或 localdata/auth_config.json + 环境变量）。"""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_AUTH_CONFIG_FILE = "localdata/auth_config.json"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _load_json_file(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_auth_settings(app_config: dict | None = None) -> dict[str, Any]:
    file_cfg = {}
    if app_config and isinstance(app_config.get("auth"), dict):
        file_cfg = dict(app_config["auth"])
    else:
        path = os.environ.get("AUTH_CONFIG_FILE", DEFAULT_AUTH_CONFIG_FILE)
        file_cfg = _load_json_file(path)

    enabled = _env_bool("AUTH_ENABLED", bool(file_cfg.get("enabled", True)))
    local_cfg = file_cfg.get("local") if isinstance(file_cfg.get("local"), dict) else {}
    oauth_cfg = file_cfg.get("oauth2") if isinstance(file_cfg.get("oauth2"), dict) else {}

    secret = (
        os.environ.get("AUTH_SESSION_SECRET")
        or file_cfg.get("session_secret")
        or "dev-change-me-visual-dps-auth-secret"
    )
    users_file = os.environ.get("AUTH_USERS_FILE") or local_cfg.get("users_file") or "localdata/auth_users.json"
    ttl_hours = int(os.environ.get("AUTH_SESSION_TTL_HOURS", file_cfg.get("session_ttl_hours", 24)))

    oauth_client_secret = os.environ.get(
        oauth_cfg.get("client_secret_env") or "OAUTH2_CLIENT_SECRET",
        oauth_cfg.get("client_secret", ""),
    )
    oauth_enabled = (
        bool(oauth_cfg.get("enabled"))
        and _oauth2_complete(oauth_cfg)
        and bool(str(oauth_client_secret).strip())
    )

    return {
        "enabled": enabled,
        "session_secret": secret,
        "cookie_name": file_cfg.get("cookie_name", "visual_dps_session"),
        "session_ttl_seconds": max(3600, ttl_hours * 3600),
        "local": {
            "enabled": bool(local_cfg.get("enabled", True)),
            "users_file": users_file,
        },
        "oauth2": {
            "enabled": oauth_enabled,
            "display_name": str(oauth_cfg.get("display_name") or "企业单点登录"),
            "client_id": str(oauth_cfg.get("client_id") or ""),
            "client_secret": oauth_client_secret,
            "authorize_url": str(oauth_cfg.get("authorize_url") or ""),
            "token_url": str(oauth_cfg.get("token_url") or ""),
            "userinfo_url": str(oauth_cfg.get("userinfo_url") or ""),
            "scopes": str(oauth_cfg.get("scopes") or "openid profile email"),
            "redirect_path": str(oauth_cfg.get("redirect_path") or "/api/auth/oauth2/callback"),
            "username_claim": str(oauth_cfg.get("username_claim") or "preferred_username"),
            "email_claim": str(oauth_cfg.get("email_claim") or "email"),
        },
    }


def _oauth2_complete(oauth_cfg: dict) -> bool:
    return bool(
        oauth_cfg.get("enabled")
        and oauth_cfg.get("client_id")
        and oauth_cfg.get("authorize_url")
        and oauth_cfg.get("token_url")
    )


def oauth2_public_config(settings: dict) -> dict | None:
    oauth = settings.get("oauth2") or {}
    if not oauth.get("enabled"):
        return None
    return {
        "enabled": True,
        "displayName": oauth.get("display_name") or "企业单点登录",
        "loginPath": "/api/auth/oauth2/login",
    }
