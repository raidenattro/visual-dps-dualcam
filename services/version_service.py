"""部署组件版本（Docker 镜像 tag 等）。"""

from __future__ import annotations

import os

from core.product_version import (
    format_image_version_display,
    format_product_version,
    is_opaque_image_id,
)


def _docker_client():
    try:
        import docker

        return docker.from_env()
    except Exception:
        return None


def _repo_tag_from_image_id(image_id: str, client) -> str:
    try:
        info = client.api.inspect_image(image_id)
        tags = [str(t) for t in (info.get("RepoTags") or []) if t]
    except Exception:
        return ""
    if not tags:
        return ""
    dated = sorted(t for t in tags if ":latest" not in t)
    return dated[-1] if dated else tags[0]


def _running_container_image(container_name: str) -> str:
    """优先 Config.Image（如 repo:latest），避免 SDK image.tags 为空时落到短 ID。"""
    client = _docker_client()
    if not client:
        return ""
    try:
        c = client.containers.get(container_name)
        cfg = str((c.attrs.get("Config") or {}).get("Image") or "").strip()
        if cfg and not is_opaque_image_id(cfg):
            return cfg
        return _repo_tag_from_image_id(c.image.id, client) or cfg
    except Exception:
        return ""


def _image_exists(client, image: str) -> bool:
    import docker

    if not str(image or "").strip():
        return False
    try:
        client.images.get(image)
        return True
    except docker.errors.ImageNotFound:
        return False


def _first_local_image(client, repo: str) -> str:
    repo = str(repo or "").strip()
    if not repo:
        return ""
    try:
        found = client.images.list(name=repo)
    except Exception:
        return ""
    tags: list[str] = []
    for img in found:
        tags.extend(img.tags or [])
    candidates = [t for t in tags if t.startswith(f"{repo}:")]
    if not candidates:
        return ""
    dated = [t for t in candidates if ":latest" not in t]
    return sorted(dated or candidates)[-1]


def _running_inference_image_ref(client) -> str:
    """取任一路在跑推理容器的镜像名（通常为带日期的 tag）。"""
    try:
        refs: list[str] = []
        for c in client.containers.list(
            filters={"label": "visual-dps.role=inference", "status": "running"}
        ):
            cfg = str((c.attrs.get("Config") or {}).get("Image") or "").strip()
            if cfg and not is_opaque_image_id(cfg):
                refs.append(cfg)
        if refs:
            dated = sorted(r for r in refs if ":latest" not in r)
            return dated[-1] if dated else sorted(refs)[0]
    except Exception:
        pass
    return ""


def _resolve_inference_image_ref() -> str:
    client = _docker_client()
    onnx_env = os.environ.get("INFERENCE_LITE_GPU_ONNX_IMAGE", "").strip()
    gpu_env = os.environ.get("INFERENCE_LITE_GPU_IMAGE", "").strip()
    lite_env = os.environ.get("INFERENCE_LITE_IMAGE", "").strip()
    if client:
        running = _running_inference_image_ref(client)
        if running:
            return running
    if not client:
        return onnx_env or gpu_env or lite_env
    use_gpu = os.environ.get("INFERENCE_USE_GPU", "0") == "1"
    if use_gpu:
        for candidate in (
            onnx_env,
            _first_local_image(client, "visual-dps-inference-lite-gpu-onnx"),
            gpu_env,
            _first_local_image(client, "visual-dps-inference-lite-gpu"),
        ):
            if candidate and _image_exists(client, candidate):
                return candidate
    for candidate in (
        lite_env,
        _first_local_image(client, "visual-dps-inference-lite"),
    ):
        if candidate and _image_exists(client, candidate):
            return candidate
    return onnx_env or gpu_env or lite_env


def get_deployment_versions() -> dict:
    product = format_product_version()
    ui_image = (
        os.environ.get("VISUAL_DPS_UI_IMAGE", "").strip()
        or _running_container_image("visual-dps-ui")
    )
    event_image = (
        os.environ.get("VISUAL_DPS_EVENT_WORKER_IMAGE", "").strip()
        or _running_container_image("visual-dps-event-worker")
    )
    infer_image = _resolve_inference_image_ref()

    ui_label = format_image_version_display(ui_image)
    event_label = format_image_version_display(event_image)
    infer_label = format_image_version_display(infer_image)

    components = {
        "product": product,
        "ui_api": product,
        "ui_image": ui_image or "—",
        "event_worker": event_image or "—",
        "inference": infer_image or "—",
        "ui_image_display": ui_label,
        "event_worker_display": event_label,
        "inference_display": infer_label,
    }

    display_parts = [
        f"UI/API {product}",
        f"Event {event_label}",
        f"Infer {infer_label}",
    ]
    return {
        "status": "success",
        "product": product,
        "components": components,
        "display": " · ".join(display_parts),
    }
