"""将 Vite 构建产物挂载为 SPA 静态站点。"""

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse

_SPA_NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate"}
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException


def mount_spa(app: FastAPI, dist_dir: str) -> None:
    """挂载 /assets 与根目录静态文件；SPA 回退须在此之后调用 register_spa_fallback。"""
    dist_dir = os.path.abspath(dist_dir)
    index_path = os.path.join(dist_dir, "index.html")
    assets_dir = os.path.join(dist_dir, "assets")

    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"SPA index not found: {index_path}")

    app.state.spa_index_path = index_path
    app.state.spa_dist_dir = dist_dir

    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="spa-assets")

    for filename in (
        "favicon.ico",
        "favicon.svg",
        "apple-touch-icon.png",
        "logo.svg",
        "logo-mark.svg",
    ):
        filepath = os.path.join(dist_dir, filename)
        if not os.path.isfile(filepath):
            continue

        def _make_handler(file_path: str):
            async def _handler():
                return FileResponse(file_path)

            return _handler

        app.get(f"/{filename}", name=f"spa-{filename}")(_make_handler(filepath))


def register_spa_fallback(app: FastAPI, dist_dir: str | None = None) -> None:
    """任意前端路径回退 index.html（须在 /api 等路由注册之后调用）。"""
    index_path = getattr(app.state, "spa_index_path", None)
    dist_dir = dist_dir or getattr(app.state, "spa_dist_dir", None)
    if not index_path or not dist_dir:
        dist_dir = os.path.abspath(dist_dir or "")
        index_path = os.path.join(dist_dir, "index.html")
    if not os.path.isfile(index_path):
        return

    reserved_prefixes = ("api/", "api", "assets/", "assets", "ws/", "ws")

    @app.get("/")
    async def spa_root():
        return FileResponse(index_path, headers=_SPA_NO_CACHE)

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        if full_path.startswith(reserved_prefixes):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = os.path.join(dist_dir, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(index_path, headers=_SPA_NO_CACHE)
