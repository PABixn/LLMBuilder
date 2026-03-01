from __future__ import annotations

import os
from pathlib import Path

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
    assert Path(os.environ["HF_HOME"]) == settings.tokenizer_hf_home
    assert Path(os.environ["HF_DATASETS_CACHE"]) == settings.tokenizer_hf_datasets_cache
    assert Path(os.environ["HUGGINGFACE_HUB_CACHE"]) == settings.tokenizer_hf_home / "hub"


def test_default_upload_dir_uses_repo_datasets_uploads(monkeypatch) -> None:
    monkeypatch.delenv("LLM_STUDIO_TOKENIZER_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("TOKENIZER_STUDIO_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("LLM_STUDIO_TOKENIZER_UPLOAD_DIR", raising=False)
    monkeypatch.delenv("TOKENIZER_STUDIO_UPLOAD_DIR", raising=False)
    config.reset_settings_cache()
    settings = config.get_settings()

    assert settings.tokenizer_output_dir == config.API_ROOT / "artifacts" / "tokenizers"
    assert settings.tokenizer_upload_dir == config.API_ROOT / "datasets" / "uploads"
