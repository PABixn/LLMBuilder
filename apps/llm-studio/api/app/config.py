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


@dataclass(frozen=True)
class RuntimeSettings:
    data_dir: Path
    projects_dir: Path
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

    host = _read_non_empty_env("LLM_STUDIO_HOST") or "127.0.0.1"
    port = _read_port("LLM_STUDIO_PORT", default=8000)
    serve_web = _read_bool("LLM_STUDIO_SERVE_WEB", default=True)
    runtime_token = _read_non_empty_env("LLM_STUDIO_RUNTIME_TOKEN")

    return RuntimeSettings(
        data_dir=data_dir,
        projects_dir=projects_dir,
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
    for path in (selected.data_dir, selected.projects_dir):
        path.mkdir(parents=True, exist_ok=True)


def apply_runtime_environment(settings: RuntimeSettings | None = None) -> None:
    _ = settings or get_settings()


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
    raw = _read_non_empty_env("LLM_STUDIO_CORS_ORIGINS")
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
    raw = _read_non_empty_env("LLM_STUDIO_CORS_ORIGIN_REGEX")
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
