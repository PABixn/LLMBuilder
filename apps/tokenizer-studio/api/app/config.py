from __future__ import annotations

import os
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = API_ROOT / "artifacts" / "tokenizers"
DEFAULT_UPLOAD_DIR = API_ROOT / "datasets" / "uploads"
DEFAULT_DATABASE_PATH = API_ROOT / "data" / "tokenizer_studio.db"
TEMPLATE_DIR = API_ROOT / "templates"

TOKENIZER_CONFIG_TEMPLATE_PATH = TEMPLATE_DIR / "tok_config.json"
DATALOADER_CONFIG_TEMPLATE_PATH = TEMPLATE_DIR / "dataloader_config.json"
TOKENIZER_SCHEMA_PATH = TEMPLATE_DIR / "tokenizer_config_schema.json"
DATALOADER_SCHEMA_PATH = TEMPLATE_DIR / "dataloader_config_schema.json"


def _resolve_env_path(var_name: str, default_path: Path) -> Path:
    raw = os.getenv(var_name)
    if raw is None or raw.strip() == "":
        return default_path
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = API_ROOT / path
    return path


def output_dir() -> Path:
    return _resolve_env_path("TOKENIZER_STUDIO_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)


def upload_dir() -> Path:
    return _resolve_env_path("TOKENIZER_STUDIO_UPLOAD_DIR", DEFAULT_UPLOAD_DIR)


def database_path() -> Path:
    return _resolve_env_path("TOKENIZER_STUDIO_DB_PATH", DEFAULT_DATABASE_PATH)


def database_url() -> str:
    explicit = os.getenv("TOKENIZER_STUDIO_DATABASE_URL")
    if explicit is not None and explicit.strip() != "":
        return explicit.strip()
    path = database_path().resolve()
    return f"sqlite:///{path}"


def max_job_workers() -> int:
    value = os.getenv("TOKENIZER_STUDIO_MAX_WORKERS", "1")
    try:
        parsed = int(value)
    except ValueError:
        return 1
    return parsed if parsed > 0 else 1
