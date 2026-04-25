from __future__ import annotations

import json
import ssl
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi


class RunPodClientError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, payload: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


@dataclass(slots=True)
class CreatePodRequest:
    name: str
    image_name: str
    gpu_type_id: str
    gpu_count: int
    cloud_type: str
    container_disk_gb: int
    volume_gb: int
    volume_mount_path: str
    ports: list[str]
    env: dict[str, str]
    data_center_id: str | None = None
    network_volume_id: str | None = None
    interruptible: bool = False
    support_public_ip: bool = True
    docker_entrypoint: list[str] = field(default_factory=list)
    docker_start_cmd: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "cloudType": self.cloud_type,
            "computeType": "GPU",
            "gpuTypeIds": [self.gpu_type_id],
            "gpuCount": self.gpu_count,
            "gpuTypePriority": "availability",
            "containerDiskInGb": self.container_disk_gb,
            "volumeInGb": self.volume_gb,
            "volumeMountPath": self.volume_mount_path,
            "ports": self.ports,
            "supportPublicIp": self.support_public_ip,
            "imageName": self.image_name,
            "dockerEntrypoint": self.docker_entrypoint,
            "dockerStartCmd": self.docker_start_cmd,
            "env": self.env,
            "interruptible": self.interruptible,
        }
        if self.data_center_id:
            payload["dataCenterIds"] = [self.data_center_id]
            payload["dataCenterPriority"] = "availability"
        if self.network_volume_id:
            payload["networkVolumeId"] = self.network_volume_id
        return payload


@dataclass(slots=True)
class CreateNetworkVolumeRequest:
    name: str
    size_gb: int
    data_center_id: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size_gb,
            "dataCenterId": self.data_center_id,
        }


class RunPodClient:
    def __init__(self, api_key: str, *, base_url: str = "https://rest.runpod.io/v1", timeout: float = 30.0) -> None:
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    def validate_key(self) -> dict[str, Any]:
        pods = self.list_pods()
        return {"pod_count": len(pods)}

    def list_pods(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        query = f"?{urlencode(filters)}" if filters else ""
        payload = self._request("GET", f"/pods{query}")
        return _extract_items(payload)

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/pods/{pod_id}")
        return _extract_object(payload)

    def create_pod(self, request: CreatePodRequest) -> dict[str, Any]:
        payload = self._request("POST", "/pods", json_body=request.to_payload())
        return _extract_object(payload)

    def stop_pod(self, pod_id: str) -> None:
        self._request("POST", f"/pods/{pod_id}/stop")

    def start_pod(self, pod_id: str) -> None:
        self._request("POST", f"/pods/{pod_id}/start")

    def delete_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"/pods/{pod_id}")

    def list_network_volumes(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/networkvolumes")
        return _extract_items(payload)

    def create_network_volume(self, request: CreateNetworkVolumeRequest) -> dict[str, Any]:
        payload = self._request("POST", "/networkvolumes", json_body=request.to_payload())
        return _extract_object(payload)

    def _request(self, method: str, path: str, *, json_body: dict[str, Any] | None = None) -> Any:
        body = None
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(f"{self._base_url}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self._timeout, context=self._ssl_context) as response:
                raw = response.read()
        except HTTPError as exc:
            payload = _decode_error_payload(exc)
            message = _error_message(payload) or f"RunPod request failed with HTTP {exc.code}."
            raise RunPodClientError(message, status_code=exc.code, payload=payload) from exc
        except URLError as exc:
            raise RunPodClientError(f"RunPod request failed: {exc.reason}") from exc
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RunPodClientError("RunPod returned a non-JSON response.") from exc


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("items", "pods", "networkVolumes", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _extract_object(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("pod", "networkVolume", "data"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        return payload
    raise RunPodClientError("RunPod returned an unexpected response shape.")


def _decode_error_payload(exc: HTTPError) -> Any:
    try:
        raw = exc.read()
    except Exception:
        return None
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return raw.decode("utf-8", errors="replace")


def _error_message(payload: Any) -> str | None:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
    return None
