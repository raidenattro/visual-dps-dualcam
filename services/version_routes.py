"""版本信息 API。"""

from fastapi import APIRouter

from services.version_service import get_deployment_versions


def register_version_routes(router: APIRouter) -> None:
    @router.get("/version")
    async def api_version():
        return get_deployment_versions()
