from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

APP_NAME = "LLMStudio"
API_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = API_ROOT / "templates"

MODEL_CONFIG_TEMPLATE_PATH = TEMPLATE_DIR / "model_config.json"
MODEL_SCHEMA_PATH = TEMPLATE_DIR / "model_config_schema.json"
TOKENIZER_CONFIG_TEMPLATE_PATH = TEMPLATE_DIR / "tok_config.json"
DATALOADER_CONFIG_TEMPLATE_PATH = TEMPLATE_DIR / "dataloader_config.json"
TOKENIZER_SCHEMA_PATH = TEMPLATE_DIR / "tokenizer_config_schema.json"
DATALOADER_SCHEMA_PATH = TEMPLATE_DIR / "dataloader_config_schema.json"


@dataclass(frozen=True)
class RuntimeSettings:
    data_dir: Path
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
    host: str
    port: int
    serve_web: bool
    web_dist_dir: Path
    cors_allowed_origins: tuple[str, ...]
    cors_allow_origin_regex: str | None
    runtime_token: str | None

    @property
    def web_index_path(self) -> Path:
        return self.web_dist_dir / "index.html"


def reset_settings_cache() -> None:
    get_settings.cache_clear()


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
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
        default_path=API_ROOT / "artifacts" / "tokenizers",
        relative_base=data_dir,
    )
    tokenizer_upload_dir = _resolve_compat_env_path(
        preferred_var_name="LLM_STUDIO_TOKENIZER_UPLOAD_DIR",
        fallback_var_name="TOKENIZER_STUDIO_UPLOAD_DIR",
        default_path=API_ROOT / "datasets" / "uploads",
        relative_base=data_dir,
    )
    tokenizer_database_path = _resolve_compat_env_path(
        preferred_var_name="LLM_STUDIO_TOKENIZER_DB_PATH",
        fallback_var_name="TOKENIZER_STUDIO_DB_PATH",
        default_path=data_dir / "db" / "tokenizer_studio.db",
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
        relative_base=tokenizer_hf_home,
    )
    tokenizer_database_url = _read_first_non_empty_env(
        "LLM_STUDIO_TOKENIZER_DATABASE_URL",
        "TOKENIZER_STUDIO_DATABASE_URL",
    ) or f"sqlite:///{tokenizer_database_path.resolve()}"

    host = _read_first_non_empty_env("LLM_STUDIO_HOST", "TOKENIZER_STUDIO_HOST") or "127.0.0.1"
    port = _read_port(("LLM_STUDIO_PORT", "TOKENIZER_STUDIO_PORT"), default=8000)
    serve_web = _read_bool(("LLM_STUDIO_SERVE_WEB", "TOKENIZER_STUDIO_SERVE_WEB"), default=True)
    runtime_token = _read_first_non_empty_env(
        "LLM_STUDIO_RUNTIME_TOKEN",
        "TOKENIZER_STUDIO_RUNTIME_TOKEN",
    )
    tokenizer_max_workers = _read_positive_int(
        ("LLM_STUDIO_TOKENIZER_MAX_WORKERS", "TOKENIZER_STUDIO_MAX_WORKERS"),
        default=1,
    )

    return RuntimeSettings(
        data_dir=data_dir,
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
        host=host,
        port=port,
        serve_web=serve_web,
        web_dist_dir=web_dist_dir,
        cors_allowed_origins=_read_origins(),
        cors_allow_origin_regex=_read_origin_regex(),
        runtime_token=runtime_token,
    )


def ensure_runtime_directories(settings: RuntimeSettings | None = None) -> None:
    selected = settings or get_settings()
    for path in (
        selected.data_dir,
        selected.projects_dir,
        selected.tokenizer_cache_dir,
        selected.tokenizer_output_dir,
        selected.tokenizer_upload_dir,
        selected.tokenizer_database_path.parent,
        selected.tokenizer_logs_dir,
        selected.tokenizer_hf_home,
        selected.tokenizer_hf_datasets_cache,
    ):
        path.mkdir(parents=True, exist_ok=True)


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


def _read_origins() -> tuple[str, ...]:
    raw = _read_first_non_empty_env("LLM_STUDIO_CORS_ORIGINS", "TOKENIZER_STUDIO_CORS_ORIGINS")
    if raw is None:
        return (
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        )
    values = [value.strip() for value in raw.split(",") if value.strip() != ""]
    return tuple(values)


def _read_origin_regex() -> str | None:
    raw = _read_first_non_empty_env(
        "LLM_STUDIO_CORS_ORIGIN_REGEX",
        "TOKENIZER_STUDIO_CORS_ORIGIN_REGEX",
    )
    if raw is not None:
        return raw
    return r"http://(localhost|127\\.0\\.0\\.1)(:\\d+)?"


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
