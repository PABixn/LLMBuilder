from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

APP_NAME = "TokenizerStudio"
API_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = API_ROOT / "templates"

TOKENIZER_CONFIG_TEMPLATE_PATH = TEMPLATE_DIR / "tok_config.json"
DATALOADER_CONFIG_TEMPLATE_PATH = TEMPLATE_DIR / "dataloader_config.json"
TOKENIZER_SCHEMA_PATH = TEMPLATE_DIR / "tokenizer_config_schema.json"
DATALOADER_SCHEMA_PATH = TEMPLATE_DIR / "dataloader_config_schema.json"


@dataclass(frozen=True)
class RuntimeSettings:
    data_dir: Path
    cache_dir: Path
    output_dir: Path
    upload_dir: Path
    database_path: Path
    logs_dir: Path
    hf_home: Path
    hf_datasets_cache: Path
    host: str
    port: int
    serve_web: bool
    web_dist_dir: Path
    cors_allowed_origins: tuple[str, ...]
    cors_allow_origin_regex: str | None
    runtime_token: str | None
    max_workers: int
    database_url: str

    @property
    def web_index_path(self) -> Path:
        return self.web_dist_dir / "index.html"


def reset_settings_cache() -> None:
    get_settings.cache_clear()


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
    data_dir = _resolve_env_path(
        "TOKENIZER_STUDIO_DATA_DIR",
        _default_data_dir(),
        relative_base=_default_data_dir(),
    )
    cache_dir = _resolve_env_path(
        "TOKENIZER_STUDIO_CACHE_DIR",
        _default_cache_dir(),
        relative_base=data_dir,
    )
    output_root = _resolve_env_path(
        "TOKENIZER_STUDIO_OUTPUT_DIR",
        data_dir / "artifacts" / "tokenizers",
        relative_base=data_dir,
    )
    upload_root = _resolve_env_path(
        "TOKENIZER_STUDIO_UPLOAD_DIR",
        data_dir / "uploads",
        relative_base=data_dir,
    )
    db_path = _resolve_env_path(
        "TOKENIZER_STUDIO_DB_PATH",
        data_dir / "db" / "tokenizer_studio.db",
        relative_base=data_dir,
    )
    logs_root = _resolve_env_path(
        "TOKENIZER_STUDIO_LOG_DIR",
        data_dir / "logs",
        relative_base=data_dir,
    )
    hf_root = _resolve_env_path(
        "TOKENIZER_STUDIO_HF_HOME",
        data_dir / "cache" / "huggingface",
        relative_base=data_dir,
    )
    hf_datasets_root = _resolve_env_path(
        "TOKENIZER_STUDIO_HF_DATASETS_CACHE",
        hf_root / "datasets",
        relative_base=hf_root,
    )
    web_dist_root = _resolve_env_path(
        "TOKENIZER_STUDIO_WEB_DIST_DIR",
        API_ROOT / "web-dist",
        relative_base=API_ROOT,
    )

    explicit_database_url = _read_non_empty_env("TOKENIZER_STUDIO_DATABASE_URL")
    database_url_value = explicit_database_url or f"sqlite:///{db_path.resolve()}"
    host_value = _read_non_empty_env("TOKENIZER_STUDIO_HOST") or "127.0.0.1"
    port_value = _read_port("TOKENIZER_STUDIO_PORT", default=8000)
    serve_web_value = _read_bool("TOKENIZER_STUDIO_SERVE_WEB", default=True)
    runtime_token_value = _read_non_empty_env("TOKENIZER_STUDIO_RUNTIME_TOKEN")
    max_workers_value = _read_positive_int("TOKENIZER_STUDIO_MAX_WORKERS", default=1)
    origins_value = _read_origins()
    origin_regex_value = _read_origin_regex()

    return RuntimeSettings(
        data_dir=data_dir,
        cache_dir=cache_dir,
        output_dir=output_root,
        upload_dir=upload_root,
        database_path=db_path,
        logs_dir=logs_root,
        hf_home=hf_root,
        hf_datasets_cache=hf_datasets_root,
        host=host_value,
        port=port_value,
        serve_web=serve_web_value,
        web_dist_dir=web_dist_root,
        cors_allowed_origins=origins_value,
        cors_allow_origin_regex=origin_regex_value,
        runtime_token=runtime_token_value,
        max_workers=max_workers_value,
        database_url=database_url_value,
    )


def ensure_runtime_directories(settings: RuntimeSettings | None = None) -> None:
    selected = settings or get_settings()
    for path in (
        selected.data_dir,
        selected.cache_dir,
        selected.output_dir,
        selected.upload_dir,
        selected.database_path.parent,
        selected.logs_dir,
        selected.hf_home,
        selected.hf_datasets_cache,
    ):
        path.mkdir(parents=True, exist_ok=True)


def apply_runtime_environment(settings: RuntimeSettings | None = None) -> None:
    selected = settings or get_settings()
    os.environ.setdefault("HF_HOME", str(selected.hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(selected.hf_datasets_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(selected.hf_home / "hub"))


def output_dir() -> Path:
    return get_settings().output_dir


def upload_dir() -> Path:
    return get_settings().upload_dir


def database_path() -> Path:
    return get_settings().database_path


def database_url() -> str:
    return get_settings().database_url


def max_job_workers() -> int:
    return get_settings().max_workers


def _resolve_env_path(var_name: str, default_path: Path, *, relative_base: Path) -> Path:
    raw = os.getenv(var_name)
    if raw is None or raw.strip() == "":
        return default_path
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = relative_base / path
    return path


def _read_non_empty_env(var_name: str) -> str | None:
    raw = os.getenv(var_name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped else None


def _read_bool(var_name: str, *, default: bool) -> bool:
    raw = _read_non_empty_env(var_name)
    if raw is None:
        return default
    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _read_positive_int(var_name: str, *, default: int) -> int:
    raw = _read_non_empty_env(var_name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _read_port(var_name: str, *, default: int) -> int:
    raw = _read_non_empty_env(var_name)
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
    raw = _read_non_empty_env("TOKENIZER_STUDIO_CORS_ORIGINS")
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
    raw = _read_non_empty_env("TOKENIZER_STUDIO_CORS_ORIGIN_REGEX")
    if raw is not None:
        return raw
    return r"http://(localhost|127\.0\.0\.1)(:\d+)?"


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
