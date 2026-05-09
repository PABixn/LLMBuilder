from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ResolvedRunPodTarget:
    api_key: str
    api_key_source: str
    image_name: str
    gpu_type_id: str
    gpu_count: int
    cloud_type: str
    data_center_id: str | None
    volume_size_gb: int
    container_disk_gb: int
    volume_mount_path: str
    agent_port: int
    agent_port_protocol: str
    cleanup_policy: dict[str, Any]
    interruptible: bool


def resolve_runpod_target(target: dict[str, Any], settings: Any) -> ResolvedRunPodTarget:
    ui_api_key = str(target.get("api_key") or "").strip()
    env_api_key = str(settings.runpod_api_key or "").strip()
    api_key = ui_api_key or env_api_key
    if not api_key:
        raise ValueError("RunPod API key is required. Paste one in the UI or set LLM_STUDIO_RUNPOD_API_KEY.")

    cleanup_policy = target.get("cleanup_policy") if isinstance(target.get("cleanup_policy"), dict) else {}
    return ResolvedRunPodTarget(
        api_key=api_key,
        api_key_source="ui" if ui_api_key else "env",
        image_name=str(settings.runpod_training_image),
        gpu_type_id=str(target.get("gpu_type_id") or settings.runpod_default_gpu_type),
        gpu_count=_positive_int(target.get("gpu_count") or settings.runpod_default_gpu_count, "gpu_count"),
        cloud_type=str(target.get("cloud_type") or settings.runpod_default_cloud_type).upper(),
        data_center_id=optional_str(target.get("data_center_id") or settings.runpod_default_data_center_id),
        volume_size_gb=_positive_int(
            target.get("network_volume_size_gb") or settings.runpod_default_volume_size_gb,
            "network_volume_size_gb",
        ),
        container_disk_gb=_positive_int(settings.runpod_container_disk_gb, "container_disk_gb"),
        volume_mount_path=str(settings.runpod_volume_mount_path),
        agent_port=_positive_int(settings.runpod_agent_port, "agent_port"),
        agent_port_protocol=str(settings.runpod_agent_port_protocol or "tcp").lower(),
        cleanup_policy=cleanup_policy,
        interruptible=bool(target.get("interruptible", False)),
    )


def optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"RunPod {field_name} must be an integer.") from exc
    if parsed <= 0:
        raise ValueError(f"RunPod {field_name} must be positive.")
    return parsed
