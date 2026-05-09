from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SENSITIVE_KEYS = {"api_key", "apikey", "authorization", "bearer", "password", "secret", "token"}
SENSITIVE_KEY_SUFFIXES = ("_api_key", "_apikey", "_authorization", "_password", "_secret", "_token")
ENV_KEYS_TO_REPORT = (
    "CUDA_VISIBLE_DEVICES",
    "HF_DATASETS_CACHE",
    "HF_HOME",
    "LLM_STUDIO_IMAGE_BUILT_AT",
    "LLM_STUDIO_IMAGE_REVISION",
    "LLM_STUDIO_REMOTE_AGENT_TOKEN",
    "LLM_STUDIO_REMOTE_JOB_ID",
    "LLM_STUDIO_REMOTE_WORKSPACE",
    "LLM_STUDIO_RUNPOD_AGENT_PORT",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
)


def workspace_root() -> Path:
    return Path(os.getenv("LLM_STUDIO_REMOTE_WORKSPACE", "/workspace/llm-studio")).resolve()


def logs_root() -> Path:
    path = workspace_root() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_file_path(name: str) -> Path:
    return logs_root() / name


def log_event(
    *,
    service: str,
    event: str,
    job_id: str | None = None,
    file_name: str,
    prefix: str,
    **fields: Any,
) -> dict[str, Any]:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "job_id": job_id,
        "service": service,
        **sanitize_log_fields(fields),
    }
    line = json.dumps(payload, ensure_ascii=True, default=str, sort_keys=True)
    print(f"[{prefix}] {line}", flush=True)
    try:
        with log_file_path(file_name).open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
    except Exception as exc:
        fallback = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "diagnostic_log_write_failed",
            "service": service,
            "target": str(workspace_root() / "logs" / file_name),
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"[{prefix}] {json.dumps(fallback, ensure_ascii=True, sort_keys=True)}", flush=True)
    return payload


def emit_startup_diagnostics() -> None:
    job_id = os.getenv("LLM_STUDIO_REMOTE_JOB_ID") or None
    log_startup("startup_begin", job_id=job_id, workspace=str(workspace_root()), cwd=os.getcwd())
    log_startup(
        "environment",
        job_id=job_id,
        env={key: os.getenv(key) for key in ENV_KEYS_TO_REPORT if os.getenv(key) is not None},
    )
    log_startup(
        "python",
        job_id=job_id,
        executable=sys.executable,
        version=sys.version,
        platform=platform.platform(),
        argv=sys.argv,
        path=sys.path[:12],
    )
    log_startup("filesystem", job_id=job_id, checks=filesystem_checks(), disk_usage=disk_usage_report())
    log_startup("system_tools", job_id=job_id, tools=system_tool_report())
    log_startup("nvidia_smi", job_id=job_id, **nvidia_smi_report())
    log_startup("torch_cuda", job_id=job_id, **torch_cuda_report())
    log_startup("module_imports", job_id=job_id, modules=module_import_report())
    log_startup("startup_done", job_id=job_id)


def log_startup(event: str, *, job_id: str | None = None, **fields: Any) -> None:
    log_event(
        service="llm-studio-remote-startup",
        event=event,
        job_id=job_id,
        file_name="startup.log",
        prefix="llm-studio-startup",
        **fields,
    )


def filesystem_checks() -> list[dict[str, Any]]:
    candidates = (
        "/opt/llm-builder",
        "/opt/llm-builder/training/runner.py",
        "/opt/llm-builder/training/local_text_data.py",
        "/opt/llm-builder/remote_agent/app.py",
        str(workspace_root()),
    )
    checks: list[dict[str, Any]] = []
    for candidate in candidates:
        path = Path(candidate)
        item: dict[str, Any] = {
            "path": candidate,
            "exists": path.exists(),
            "is_dir": path.is_dir(),
            "is_file": path.is_file(),
        }
        try:
            stat = path.stat()
            item["size_bytes"] = stat.st_size
            item["mode"] = oct(stat.st_mode & 0o777)
        except OSError as exc:
            item["stat_error"] = f"{type(exc).__name__}: {exc}"
        checks.append(item)
    return checks


def disk_usage_report() -> dict[str, Any]:
    report: dict[str, Any] = {}
    for label, path in (("workspace", workspace_root()), ("repo", Path("/opt/llm-builder")), ("root", Path("/"))):
        try:
            usage = shutil.disk_usage(path)
            report[label] = {
                "path": str(path),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            }
        except OSError as exc:
            report[label] = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return report


def nvidia_smi_report() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "error": "nvidia-smi is not on PATH"}
    command = [
        executable,
        "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except Exception as exc:
        return {"available": True, "command": command, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip().splitlines(),
        "stderr": completed.stderr.strip().splitlines(),
    }


def system_tool_report() -> list[dict[str, Any]]:
    tools = ("cc", "gcc", "g++", "make", "zstd")
    results: list[dict[str, Any]] = []
    for tool in tools:
        executable = shutil.which(tool)
        item: dict[str, Any] = {"tool": tool, "available": executable is not None, "path": executable}
        if executable is not None:
            try:
                completed = subprocess.run(
                    [executable, "--version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except Exception as exc:
                item["version_error"] = f"{type(exc).__name__}: {exc}"
            else:
                item["returncode"] = completed.returncode
                item["version"] = (completed.stdout or completed.stderr).strip().splitlines()[:1]
        results.append(item)
    return results


def torch_cuda_report() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "import_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
    devices: list[dict[str, Any]] = []
    try:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "capability": list(properties.major_minor) if hasattr(properties, "major_minor") else [properties.major, properties.minor],
                }
            )
    except Exception as exc:
        devices.append({"error": f"{type(exc).__name__}: {exc}"})
    return {
        "import_ok": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
    }


def module_import_report() -> list[dict[str, Any]]:
    modules = {
        "zstandard": True,
        "fastapi": True,
        "uvicorn": True,
        "torch": True,
        "datasets": True,
        "tokenizers": True,
        "training.local_text_data": True,
        "training.runner": True,
        "remote_agent.app": True,
    }
    results: list[dict[str, Any]] = []
    for module, required in modules.items():
        try:
            imported = importlib.import_module(module)
        except Exception as exc:
            results.append(
                {
                    "module": module,
                    "required": required,
                    "import_ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                }
            )
            continue
        results.append({"module": module, "required": required, "import_ok": True, "file": getattr(imported, "__file__", None)})
    return results


def sanitize_log_fields(fields: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in fields.items():
        if is_sensitive_key(key):
            sanitized[key] = "[redacted]"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_fields(value)
        elif isinstance(value, list):
            sanitized[key] = [sanitize_log_value(item) for item in value]
        else:
            sanitized[key] = sanitize_log_value(value)
    return sanitized


def sanitize_log_value(value: Any) -> Any:
    if isinstance(value, dict):
        return sanitize_log_fields(value)
    if isinstance(value, list):
        return [sanitize_log_value(item) for item in value]
    return value


def is_sensitive_key(key: str) -> bool:
    key_lower = key.lower()
    return key_lower in SENSITIVE_KEYS or key_lower.endswith(SENSITIVE_KEY_SUFFIXES)


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "startup"
    if command != "startup":
        print(f"Unsupported diagnostics command: {command}", file=sys.stderr, flush=True)
        return 2
    try:
        emit_startup_diagnostics()
    except Exception:
        log_startup(
            "startup_diagnostics_failed",
            job_id=os.getenv("LLM_STUDIO_REMOTE_JOB_ID") or None,
            traceback=traceback.format_exc(limit=12),
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
