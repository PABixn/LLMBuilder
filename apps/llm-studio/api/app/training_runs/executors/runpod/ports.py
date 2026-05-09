from __future__ import annotations

from typing import Any

from .config import optional_str


def build_agent_base_url(pod: dict[str, Any], port: int) -> str:
    pod_id = optional_str(pod.get("id")) or optional_str(pod.get("podId"))
    mappings = extract_port_mappings(pod)
    public_ip = optional_str(pod.get("publicIp"))
    for value in mappings.values():
        if isinstance(value, dict):
            url = value.get("url") or value.get("uri")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return url.rstrip("/")
    for key, value in mappings.items():
        if isinstance(value, dict):
            if pod_id and is_http_port_mapping(value, port):
                return f"https://{pod_id}-{port}.proxy.runpod.net"
            if not is_tcp_port_mapping(value):
                continue
            host = tcp_mapping_host(value, public_ip)
            mapped_port = value.get("publicPort") or value.get("externalPort") or value.get("port")
            if host and mapped_port:
                return f"http://{host}:{mapped_port}"
        elif public_ip and str(key) == str(port) and isinstance(value, (int, str)):
            try:
                mapped_port = int(value)
            except (TypeError, ValueError):
                continue
            return f"http://{public_ip}:{mapped_port}"
        elif isinstance(value, str) and value.startswith(("http://", "https://")):
            return value.rstrip("/")
    if pod_id and pod_exposes_http_port(pod, port):
        return f"https://{pod_id}-{port}.proxy.runpod.net"
    return ""


def extract_port_mappings(pod: dict[str, Any]) -> dict[str, Any]:
    value = pod.get("portMappings")
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {str(index): item for index, item in enumerate(value)}
    runtime = pod.get("runtime")
    if isinstance(runtime, dict):
        value = runtime.get("ports")
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {str(index): item for index, item in enumerate(value)}
    value = pod.get("ports")
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {str(index): item for index, item in enumerate(value)}
    return {}


def pod_exposes_http_port(pod: dict[str, Any], port: int) -> bool:
    expected = f"{port}/http"
    ports = pod.get("ports")
    if isinstance(ports, list) and any(item == expected for item in ports):
        return True
    mappings = extract_port_mappings(pod)
    return any(is_http_port_mapping(value, port) for value in mappings.values())


def is_http_port_mapping(value: Any, port: int) -> bool:
    if not isinstance(value, dict):
        return False
    mapping_type = str(value.get("type") or value.get("protocol") or "").lower()
    private_port = value.get("privatePort") or value.get("containerPort") or value.get("port")
    try:
        private_port_number = int(private_port)
    except (TypeError, ValueError):
        private_port_number = None
    return mapping_type == "http" and private_port_number == port


def is_tcp_port_mapping(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    mapping_type = str(value.get("type") or value.get("protocol") or "").lower()
    return mapping_type in {"", "tcp"}


def tcp_mapping_host(value: dict[str, Any], public_ip: str | None) -> Any:
    host = value.get("host") or value.get("ip")
    if value.get("isIpPublic") is False and public_ip:
        return public_ip
    return host or public_ip
