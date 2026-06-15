from __future__ import annotations

import ipaddress
import os
import platform
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .runtime_paths import api_root, source_root, template_dir
from .storage_safety import (
    InsufficientStorageError,
    ManagedStorageError,
    ensure_directory,
    ensure_writable_directory,
)

APP_NAME = "LLMStudio"
API_ROOT = api_root()
TEMPLATE_DIR = template_dir()
REPO_ROOT = source_root()

MODEL_CONFIG_TEMPLATE_PATH = TEMPLATE_DIR / "model_config.json"
MODEL_SCHEMA_PATH = TEMPLATE_DIR / "model_config_schema.json"
TOKENIZER_CONFIG_TEMPLATE_PATH = TEMPLATE_DIR / "tok_config.json"
DATALOADER_CONFIG_TEMPLATE_PATH = TEMPLATE_DIR / "dataloader_config.json"
TOKENIZER_SCHEMA_PATH = TEMPLATE_DIR / "tokenizer_config_schema.json"
DATALOADER_SCHEMA_PATH = TEMPLATE_DIR / "dataloader_config_schema.json"
TRAINING_LOOP_TEMPLATE_PATH = REPO_ROOT / "training" / "training_config.json"
TRAINING_LOOP_SCHEMA_PATH = REPO_ROOT / "training" / "training_config_schema.json"
TRAINING_DATALOADER_TEMPLATE_PATH = REPO_ROOT / "training" / "dataloader_config.json"
TRAINING_DATALOADER_SCHEMA_PATH = REPO_ROOT / "training" / "dataloader_config_schema.json"


@dataclass(frozen=True)
class RuntimeSettings:
    desktop_mode: bool
    source_root: Path
    data_dir: Path
    cache_dir: Path
    log_dir: Path
    projects_dir: Path
    tokenizer_cache_dir: Path
    tokenizer_output_dir: Path
    tokenizer_upload_dir: Path
    tokenizer_database_path: Path
    tokenizer_logs_dir: Path
    tokenizer_hf_home: Path
    tokenizer_hf_datasets_cache: Path
    tokenizer_max_workers: int
    tokenizer_database_url: str
    training_jobs_dir: Path
    training_exports_dir: Path
    training_database_path: Path
    training_database_url: str
    runpod_api_key: str | None
    runpod_default_gpu_type: str
    runpod_default_gpu_count: int
    runpod_default_cloud_type: str
    runpod_default_data_center_id: str | None
    runpod_default_volume_size_gb: int
    runpod_container_disk_gb: int
    runpod_volume_mount_path: str
    runpod_training_image: str
    runpod_agent_port: int
    runpod_agent_port_protocol: str
    runpod_pod_ttl_minutes: int
    runpod_auto_delete_pod: bool
    runpod_auto_delete_volume: bool
    host: str
    port: int
    serve_web: bool
    web_dist_dir: Path
    cors_allowed_origins: tuple[str, ...]
    cors_allow_origin_regex: str | None
    runtime_token: str | None
    runtime_version: str
    startup_handshake_path: Path | None

    @property
    def web_index_path(self) -> Path:
        return self.web_dist_dir / "index.html"


def reset_settings_cache() -> None:
    get_settings.cache_clear()


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
    desktop_mode = _read_bool("LLM_STUDIO_DESKTOP", default=False)
    data_dir = _resolve_env_path(
        "LLM_STUDIO_DATA_DIR",
        _default_data_dir(),
        relative_base=_default_data_dir(),
    )
    projects_dir = _resolve_env_path(
        "LLM_STUDIO_PROJECTS_DIR",
        data_dir / "projects",
        relative_base=data_dir,
    )
    web_dist_dir = _resolve_env_path(
        "LLM_STUDIO_WEB_DIST_DIR",
        API_ROOT / "web-dist",
        relative_base=API_ROOT,
    )
    tokenizer_cache_dir = _resolve_compat_env_path(
        preferred_var_name="LLM_STUDIO_TOKENIZER_CACHE_DIR",
        fallback_var_name="TOKENIZER_STUDIO_CACHE_DIR",
        default_path=_default_cache_dir(),
        relative_base=data_dir,
    )
    tokenizer_output_dir = _resolve_compat_env_path(
        preferred_var_name="LLM_STUDIO_TOKENIZER_OUTPUT_DIR",
        fallback_var_name="TOKENIZER_STUDIO_OUTPUT_DIR",
        default_path=data_dir / "artifacts" / "tokenizers",
        relative_base=data_dir,
    )
    tokenizer_upload_dir = _resolve_compat_env_path(
        preferred_var_name="LLM_STUDIO_TOKENIZER_UPLOAD_DIR",
        fallback_var_name="TOKENIZER_STUDIO_UPLOAD_DIR",
        default_path=data_dir / "uploads",
        relative_base=data_dir,
    )
    tokenizer_database_path = _resolve_compat_env_path(
        preferred_var_name="LLM_STUDIO_TOKENIZER_DB_PATH",
        fallback_var_name="TOKENIZER_STUDIO_DB_PATH",
        default_path=data_dir / "db" / "llm_studio_tokenizer.db",
        relative_base=data_dir,
    )
    tokenizer_logs_dir = _resolve_compat_env_path(
        preferred_var_name="LLM_STUDIO_TOKENIZER_LOG_DIR",
        fallback_var_name="TOKENIZER_STUDIO_LOG_DIR",
        default_path=data_dir / "logs",
        relative_base=data_dir,
    )
    tokenizer_hf_home = _resolve_compat_env_path(
        preferred_var_name="LLM_STUDIO_TOKENIZER_HF_HOME",
        fallback_var_name="TOKENIZER_STUDIO_HF_HOME",
        default_path=data_dir / "cache" / "huggingface",
        relative_base=data_dir,
    )
    tokenizer_hf_datasets_cache = _resolve_compat_env_path(
        preferred_var_name="LLM_STUDIO_TOKENIZER_HF_DATASETS_CACHE",
        fallback_var_name="TOKENIZER_STUDIO_HF_DATASETS_CACHE",
        default_path=tokenizer_hf_home / "datasets",
        relative_base=data_dir,
    )
    tokenizer_database_url = _read_first_non_empty_env(
        "LLM_STUDIO_TOKENIZER_DATABASE_URL",
        "TOKENIZER_STUDIO_DATABASE_URL",
    ) or f"sqlite:///{tokenizer_database_path.resolve()}"
    training_jobs_dir = _resolve_compat_env_path(
        preferred_var_name="LLM_STUDIO_TRAINING_JOBS_DIR",
        fallback_var_name="TOKENIZER_STUDIO_TRAINING_JOBS_DIR",
        default_path=data_dir / "training" / "jobs",
        relative_base=data_dir,
    )
    training_exports_dir = _resolve_compat_env_path(
        preferred_var_name="LLM_STUDIO_TRAINING_EXPORT_DIR",
        fallback_var_name="TOKENIZER_STUDIO_TRAINING_EXPORT_DIR",
        default_path=data_dir / "training" / "exports",
        relative_base=data_dir,
    )
    training_database_path = _resolve_compat_env_path(
        preferred_var_name="LLM_STUDIO_TRAINING_DB_PATH",
        fallback_var_name="TOKENIZER_STUDIO_TRAINING_DB_PATH",
        default_path=data_dir / "db" / "llm_studio_training.db",
        relative_base=data_dir,
    )
    training_database_url = _read_first_non_empty_env(
        "LLM_STUDIO_TRAINING_DATABASE_URL",
        "TOKENIZER_STUDIO_TRAINING_DATABASE_URL",
    ) or f"sqlite:///{training_database_path.resolve()}"
    runpod_api_key = _read_first_non_empty_env("LLM_STUDIO_RUNPOD_API_KEY")
    runpod_default_gpu_type = (
        _read_first_non_empty_env("LLM_STUDIO_RUNPOD_DEFAULT_GPU_TYPE")
        or "NVIDIA GeForce RTX 4090"
    )
    runpod_default_gpu_count = _read_positive_int("LLM_STUDIO_RUNPOD_DEFAULT_GPU_COUNT", default=1)
    runpod_default_cloud_type = (
        _read_first_non_empty_env("LLM_STUDIO_RUNPOD_DEFAULT_CLOUD_TYPE")
        or "SECURE"
    ).upper()
    runpod_default_data_center_id = _read_first_non_empty_env("LLM_STUDIO_RUNPOD_DEFAULT_DATA_CENTER_ID")
    runpod_default_volume_size_gb = _read_positive_int("LLM_STUDIO_RUNPOD_DEFAULT_VOLUME_SIZE_GB", default=100)
    runpod_container_disk_gb = _read_positive_int("LLM_STUDIO_RUNPOD_CONTAINER_DISK_GB", default=50)
    runpod_volume_mount_path = (
        _read_first_non_empty_env("LLM_STUDIO_RUNPOD_VOLUME_MOUNT_PATH")
        or "/workspace"
    )
    runpod_training_image = (
        _read_first_non_empty_env("LLM_STUDIO_RUNPOD_TRAINING_IMAGE")
        or "ghcr.io/pabixn/llm-builder-training:latest"
    )
    runpod_agent_port = _read_port("LLM_STUDIO_RUNPOD_AGENT_PORT", default=8021)
    runpod_agent_port_protocol = _read_choice(
        "LLM_STUDIO_RUNPOD_AGENT_PORT_PROTOCOL",
        choices={"tcp", "http"},
        default="tcp",
    )
    runpod_pod_ttl_minutes = _read_non_negative_int("LLM_STUDIO_RUNPOD_POD_TTL_MINUTES", default=0)
    runpod_auto_delete_pod = _read_bool("LLM_STUDIO_RUNPOD_AUTO_DELETE_POD", default=True)
    runpod_auto_delete_volume = _read_bool("LLM_STUDIO_RUNPOD_AUTO_DELETE_VOLUME", default=False)

    host = _read_first_non_empty_env("LLM_STUDIO_HOST", "TOKENIZER_STUDIO_HOST") or "127.0.0.1"
    if desktop_mode and not _is_loopback_host(host):
        raise RuntimeError(
            f"Desktop mode requires a loopback bind address; received {host!r}."
        )
    port = _read_port(
        ("LLM_STUDIO_PORT", "TOKENIZER_STUDIO_PORT"),
        default=0 if desktop_mode else 8000,
    )
    serve_web = _read_bool(
        ("LLM_STUDIO_SERVE_WEB", "TOKENIZER_STUDIO_SERVE_WEB"),
        default=not desktop_mode,
    )
    runtime_token = _read_first_non_empty_env(
        "LLM_STUDIO_RUNTIME_TOKEN",
        "TOKENIZER_STUDIO_RUNTIME_TOKEN",
    )
    if desktop_mode and runtime_token is None:
        raise RuntimeError("Desktop mode requires LLM_STUDIO_RUNTIME_TOKEN.")
    cache_dir = _resolve_env_path(
        "LLM_STUDIO_CACHE_DIR",
        _default_cache_dir(),
        relative_base=data_dir,
    )
    log_dir = _resolve_env_path(
        "LLM_STUDIO_LOG_DIR",
        data_dir / "logs" / "backend",
        relative_base=data_dir,
    )
    runtime_version = _read_first_non_empty_env("LLM_STUDIO_RUNTIME_VERSION") or "source-tree"
    startup_handshake_path = (
        _resolve_env_path(
            "LLM_STUDIO_STARTUP_HANDSHAKE_PATH",
            data_dir / "startup-handshake.json",
            relative_base=data_dir,
        )
        if _read_first_non_empty_env("LLM_STUDIO_STARTUP_HANDSHAKE_PATH") is not None
        else None
    )
    tokenizer_max_workers = _read_positive_int(
        ("LLM_STUDIO_TOKENIZER_MAX_WORKERS", "TOKENIZER_STUDIO_MAX_WORKERS"),
        default=1,
    )

    return RuntimeSettings(
        desktop_mode=desktop_mode,
        source_root=REPO_ROOT,
        data_dir=data_dir,
        cache_dir=cache_dir,
        log_dir=log_dir,
        projects_dir=projects_dir,
        tokenizer_cache_dir=tokenizer_cache_dir,
        tokenizer_output_dir=tokenizer_output_dir,
        tokenizer_upload_dir=tokenizer_upload_dir,
        tokenizer_database_path=tokenizer_database_path,
        tokenizer_logs_dir=tokenizer_logs_dir,
        tokenizer_hf_home=tokenizer_hf_home,
        tokenizer_hf_datasets_cache=tokenizer_hf_datasets_cache,
        tokenizer_max_workers=tokenizer_max_workers,
        tokenizer_database_url=tokenizer_database_url,
        training_jobs_dir=training_jobs_dir,
        training_exports_dir=training_exports_dir,
        training_database_path=training_database_path,
        training_database_url=training_database_url,
        runpod_api_key=runpod_api_key,
        runpod_default_gpu_type=runpod_default_gpu_type,
        runpod_default_gpu_count=runpod_default_gpu_count,
        runpod_default_cloud_type=runpod_default_cloud_type,
        runpod_default_data_center_id=runpod_default_data_center_id,
        runpod_default_volume_size_gb=runpod_default_volume_size_gb,
        runpod_container_disk_gb=runpod_container_disk_gb,
        runpod_volume_mount_path=runpod_volume_mount_path,
        runpod_training_image=runpod_training_image,
        runpod_agent_port=runpod_agent_port,
        runpod_agent_port_protocol=runpod_agent_port_protocol,
        runpod_pod_ttl_minutes=runpod_pod_ttl_minutes,
        runpod_auto_delete_pod=runpod_auto_delete_pod,
        runpod_auto_delete_volume=runpod_auto_delete_volume,
        host=host,
        port=port,
        serve_web=serve_web,
        web_dist_dir=web_dist_dir,
        cors_allowed_origins=_read_origins(desktop_mode=desktop_mode),
        cors_allow_origin_regex=_read_origin_regex(desktop_mode=desktop_mode),
        runtime_token=runtime_token,
        runtime_version=runtime_version,
        startup_handshake_path=startup_handshake_path,
    )


def ensure_runtime_directories(settings: RuntimeSettings | None = None) -> None:
    selected = settings or get_settings()
    for path in (
        selected.data_dir,
        selected.cache_dir,
        selected.log_dir,
        selected.projects_dir,
        selected.tokenizer_cache_dir,
        selected.tokenizer_output_dir,
        selected.tokenizer_upload_dir,
        selected.tokenizer_database_path.parent,
        selected.tokenizer_logs_dir,
        selected.tokenizer_hf_home,
        selected.tokenizer_hf_datasets_cache,
        selected.training_jobs_dir,
        selected.training_exports_dir,
        selected.training_database_path.parent,
    ):
        ensure_directory(path, operation="desktop runtime initialization")


def validate_runtime_storage(
    settings: RuntimeSettings | None = None,
    *,
    minimum_free_bytes: int = 64 * 1024 * 1024,
) -> None:
    selected = settings or get_settings()
    failures: list[str] = []
    for path in (selected.data_dir, selected.cache_dir, selected.log_dir):
        try:
            ensure_writable_directory(path, operation="desktop runtime startup")
        except ManagedStorageError as exc:
            failures.append(str(exc))
            continue
        try:
            free = shutil.disk_usage(path).free
        except OSError:
            failures.append(f"{path}: could not inspect available disk space")
            continue
        if free < minimum_free_bytes:
            failures.append(
                f"{path}: insufficient free space ({free} bytes available, "
                f"{minimum_free_bytes} required)"
            )
    if failures:
        if any("insufficient free space" in failure for failure in failures):
            error_type = InsufficientStorageError
        else:
            error_type = ManagedStorageError
        raise error_type("Desktop runtime storage validation failed:\n- " + "\n- ".join(failures))


def apply_runtime_environment(settings: RuntimeSettings | None = None) -> None:
    selected = settings or get_settings()
    os.environ.setdefault("HF_HOME", str(selected.tokenizer_hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(selected.tokenizer_hf_datasets_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(selected.tokenizer_hf_home / "hub"))


def tokenizer_output_dir() -> Path:
    return get_settings().tokenizer_output_dir


def tokenizer_upload_dir() -> Path:
    return get_settings().tokenizer_upload_dir


def tokenizer_database_url() -> str:
    return get_settings().tokenizer_database_url


def tokenizer_max_job_workers() -> int:
    return get_settings().tokenizer_max_workers


def training_jobs_dir() -> Path:
    return get_settings().training_jobs_dir


def training_exports_dir() -> Path:
    return get_settings().training_exports_dir


def training_database_url() -> str:
    return get_settings().training_database_url


def _resolve_env_path(var_name: str, default_path: Path, *, relative_base: Path) -> Path:
    raw = os.getenv(var_name)
    if raw is None or raw.strip() == "":
        return default_path
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = relative_base / path
    return path


def _resolve_compat_env_path(
    *,
    preferred_var_name: str,
    fallback_var_name: str,
    default_path: Path,
    relative_base: Path,
) -> Path:
    raw = _read_first_non_empty_env(preferred_var_name, fallback_var_name)
    if raw is None:
        return default_path
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = relative_base / path
    return path


def _read_first_non_empty_env(*var_names: str) -> str | None:
    for var_name in var_names:
        raw = _read_non_empty_env(var_name)
        if raw is not None:
            return raw
    return None


def _read_non_empty_env(var_name: str) -> str | None:
    raw = os.getenv(var_name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped else None


def _read_bool(var_name: str | tuple[str, ...], *, default: bool) -> bool:
    var_names = var_name if isinstance(var_name, tuple) else (var_name,)
    raw = _read_first_non_empty_env(*var_names)
    if raw is None:
        return default
    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _read_positive_int(var_name: str | tuple[str, ...], *, default: int) -> int:
    var_names = var_name if isinstance(var_name, tuple) else (var_name,)
    raw = _read_first_non_empty_env(*var_names)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _read_non_negative_int(var_name: str | tuple[str, ...], *, default: int) -> int:
    var_names = var_name if isinstance(var_name, tuple) else (var_name,)
    raw = _read_first_non_empty_env(*var_names)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _read_port(var_name: str | tuple[str, ...], *, default: int) -> int:
    var_names = var_name if isinstance(var_name, tuple) else (var_name,)
    raw = _read_first_non_empty_env(*var_names)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < 0 or value > 65535:
        return default
    return value


def _read_choice(var_name: str | tuple[str, ...], *, choices: set[str], default: str) -> str:
    var_names = var_name if isinstance(var_name, tuple) else (var_name,)
    raw = _read_first_non_empty_env(*var_names)
    if raw is None:
        return default
    normalized = raw.lower()
    return normalized if normalized in choices else default


def _read_origins(*, desktop_mode: bool = False) -> tuple[str, ...]:
    raw = _read_first_non_empty_env("LLM_STUDIO_CORS_ORIGINS", "TOKENIZER_STUDIO_CORS_ORIGINS")
    if raw is None:
        if desktop_mode:
            return ("tauri://localhost", "https://tauri.localhost")
        return (
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        )
    values = [value.strip() for value in raw.split(",") if value.strip() != ""]
    return tuple(values)


def _read_origin_regex(*, desktop_mode: bool = False) -> str | None:
    raw = _read_first_non_empty_env(
        "LLM_STUDIO_CORS_ORIGIN_REGEX",
        "TOKENIZER_STUDIO_CORS_ORIGIN_REGEX",
    )
    if raw is not None:
        return raw
    if desktop_mode:
        return None
    return r"http://(localhost|127\\.0\\.0\\.1)(:\\d+)?"


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _default_data_dir() -> Path:
    if platform.system().lower() == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if os.name == "nt":
        appdata = os.getenv("APPDATA")
        if appdata:
            return Path(appdata) / APP_NAME
        return Path.home() / "AppData" / "Roaming" / APP_NAME
    xdg_home = os.getenv("XDG_DATA_HOME")
    if xdg_home:
        return Path(xdg_home) / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME


def _default_cache_dir() -> Path:
    if platform.system().lower() == "darwin":
        return Path.home() / "Library" / "Caches" / APP_NAME
    if os.name == "nt":
        local = os.getenv("LOCALAPPDATA")
        if local:
            return Path(local) / APP_NAME / "Cache"
        return Path.home() / "AppData" / "Local" / APP_NAME / "Cache"
    xdg_home = os.getenv("XDG_CACHE_HOME")
    if xdg_home:
        return Path(xdg_home) / APP_NAME
    return Path.home() / ".cache" / APP_NAME
