from __future__ import annotations

import os
from pathlib import Path

import pytest

from app import config


def test_settings_resolve_from_data_root(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "studio-data"

    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(data_root))
    monkeypatch.setenv("LLM_STUDIO_PROJECTS_DIR", "projects")
    monkeypatch.setenv("LLM_STUDIO_WEB_DIST_DIR", "web-dist")
    monkeypatch.setenv("LLM_STUDIO_PORT", "9052")
    monkeypatch.setenv("LLM_STUDIO_TOKENIZER_OUTPUT_DIR", "artifacts/tokenizers")
    monkeypatch.setenv("LLM_STUDIO_TOKENIZER_UPLOAD_DIR", "uploads")
    monkeypatch.setenv("LLM_STUDIO_TOKENIZER_DB_PATH", "db/tokenizer.db")
    monkeypatch.setenv("LLM_STUDIO_TOKENIZER_HF_HOME", "cache/hf")
    monkeypatch.setenv("LLM_STUDIO_TOKENIZER_HF_DATASETS_CACHE", "cache/hf/datasets")
    monkeypatch.setenv("LLM_STUDIO_TOKENIZER_MAX_WORKERS", "3")
    monkeypatch.setenv("LLM_STUDIO_TRAINING_JOBS_DIR", "training/jobs")
    monkeypatch.setenv("LLM_STUDIO_TRAINING_EXPORT_DIR", "training/exports")
    monkeypatch.setenv("LLM_STUDIO_TRAINING_DB_PATH", "db/training.db")

    config.reset_settings_cache()
    settings = config.get_settings()

    assert settings.data_dir == data_root
    assert settings.projects_dir == data_root / "projects"
    assert settings.web_dist_dir == config.API_ROOT / "web-dist"
    assert settings.port == 9052
    assert settings.tokenizer_output_dir == data_root / "artifacts" / "tokenizers"
    assert settings.tokenizer_upload_dir == data_root / "uploads"
    assert settings.tokenizer_database_path == data_root / "db" / "tokenizer.db"
    assert settings.tokenizer_hf_home == data_root / "cache" / "hf"
    assert settings.tokenizer_hf_datasets_cache == data_root / "cache" / "hf" / "datasets"
    assert settings.tokenizer_max_workers == 3
    assert settings.training_jobs_dir == data_root / "training" / "jobs"
    assert settings.training_exports_dir == data_root / "training" / "exports"
    assert settings.training_database_path == data_root / "db" / "training.db"


def test_runtime_directories(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "studio-data"
    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(data_root))
    config.reset_settings_cache()
    settings = config.get_settings()

    config.ensure_runtime_directories(settings)
    config.apply_runtime_environment(settings)

    assert settings.data_dir.exists()
    assert settings.projects_dir.exists()
    assert settings.tokenizer_output_dir.exists()
    assert settings.tokenizer_upload_dir.exists()
    assert settings.tokenizer_logs_dir.exists()
    assert settings.tokenizer_database_path.parent.exists()
    assert settings.tokenizer_hf_home.exists()
    assert settings.tokenizer_hf_datasets_cache.exists()
    assert settings.training_jobs_dir.exists()
    assert settings.training_exports_dir.exists()
    assert settings.training_database_path.parent.exists()
    assert Path(os.environ["HF_HOME"]) == settings.tokenizer_hf_home
    assert Path(os.environ["HF_DATASETS_CACHE"]) == settings.tokenizer_hf_datasets_cache
    assert Path(os.environ["HUGGINGFACE_HUB_CACHE"]) == settings.tokenizer_hf_home / "hub"


def test_runtime_storage_validation_reports_low_disk(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "studio-data"
    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(data_root))
    config.reset_settings_cache()
    settings = config.get_settings()
    config.ensure_runtime_directories(settings)
    monkeypatch.setattr(
        config.shutil,
        "disk_usage",
        lambda _path: config.shutil._ntuple_diskusage(total=100, used=99, free=1),
    )

    with pytest.raises(RuntimeError, match="insufficient free space"):
        config.validate_runtime_storage(settings, minimum_free_bytes=2)


def test_default_upload_and_artifact_dirs_use_managed_data_root(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "managed data"
    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(data_root))
    monkeypatch.delenv("LLM_STUDIO_TOKENIZER_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("TOKENIZER_STUDIO_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("LLM_STUDIO_TOKENIZER_UPLOAD_DIR", raising=False)
    monkeypatch.delenv("TOKENIZER_STUDIO_UPLOAD_DIR", raising=False)
    config.reset_settings_cache()
    settings = config.get_settings()

    assert settings.tokenizer_output_dir == data_root / "artifacts" / "tokenizers"
    assert settings.tokenizer_upload_dir == data_root / "uploads"


def test_default_database_names_use_llm_studio_namespace(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "managed data"
    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(data_root))
    monkeypatch.delenv("LLM_STUDIO_TOKENIZER_DB_PATH", raising=False)
    monkeypatch.delenv("TOKENIZER_STUDIO_DB_PATH", raising=False)
    monkeypatch.delenv("LLM_STUDIO_TRAINING_DB_PATH", raising=False)
    monkeypatch.delenv("TOKENIZER_STUDIO_TRAINING_DB_PATH", raising=False)
    config.reset_settings_cache()
    settings = config.get_settings()

    assert settings.tokenizer_database_path == data_root / "db" / "llm_studio_tokenizer.db"
    assert settings.training_database_path == data_root / "db" / "llm_studio_training.db"


def test_desktop_mode_requires_token_and_loopback(monkeypatch) -> None:
    monkeypatch.setenv("LLM_STUDIO_DESKTOP", "1")
    monkeypatch.delenv("LLM_STUDIO_RUNTIME_TOKEN", raising=False)
    config.reset_settings_cache()
    with pytest.raises(RuntimeError, match="requires LLM_STUDIO_RUNTIME_TOKEN"):
        config.get_settings()

    monkeypatch.setenv("LLM_STUDIO_RUNTIME_TOKEN", "test-runtime-token")
    monkeypatch.setenv("LLM_STUDIO_HOST", "0.0.0.0")
    config.reset_settings_cache()
    with pytest.raises(RuntimeError, match="loopback bind address"):
        config.get_settings()


def test_runpod_agent_port_protocol_defaults_to_tcp(monkeypatch) -> None:
    monkeypatch.delenv("LLM_STUDIO_RUNPOD_AGENT_PORT_PROTOCOL", raising=False)
    config.reset_settings_cache()

    assert config.get_settings().runpod_agent_port_protocol == "tcp"


def test_runpod_agent_port_protocol_can_use_http_proxy(monkeypatch) -> None:
    monkeypatch.setenv("LLM_STUDIO_RUNPOD_AGENT_PORT_PROTOCOL", "http")
    config.reset_settings_cache()

    assert config.get_settings().runpod_agent_port_protocol == "http"
