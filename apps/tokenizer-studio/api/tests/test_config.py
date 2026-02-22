from __future__ import annotations

import os
from pathlib import Path

from app import config


def test_settings_resolve_from_data_root(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "studio-data"

    monkeypatch.setenv("TOKENIZER_STUDIO_DATA_DIR", str(data_root))
    monkeypatch.setenv("TOKENIZER_STUDIO_OUTPUT_DIR", "artifacts/tokenizers")
    monkeypatch.setenv("TOKENIZER_STUDIO_UPLOAD_DIR", "uploads")
    monkeypatch.setenv("TOKENIZER_STUDIO_DB_PATH", "db/tokenizer.db")
    monkeypatch.setenv("TOKENIZER_STUDIO_HF_HOME", "cache/hf")
    monkeypatch.setenv("TOKENIZER_STUDIO_HF_DATASETS_CACHE", "cache/hf/datasets")
    monkeypatch.setenv("TOKENIZER_STUDIO_WEB_DIST_DIR", "web-dist")
    monkeypatch.setenv("TOKENIZER_STUDIO_PORT", "9051")

    config.reset_settings_cache()
    settings = config.get_settings()

    assert settings.data_dir == data_root
    assert settings.output_dir == data_root / "artifacts" / "tokenizers"
    assert settings.upload_dir == data_root / "uploads"
    assert settings.database_path == data_root / "db" / "tokenizer.db"
    assert settings.hf_home == data_root / "cache" / "hf"
    assert settings.hf_datasets_cache == data_root / "cache" / "hf" / "datasets"
    assert settings.web_dist_dir == config.API_ROOT / "web-dist"
    assert settings.port == 9051


def test_runtime_directories_and_hf_env(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "studio-data"
    monkeypatch.setenv("TOKENIZER_STUDIO_DATA_DIR", str(data_root))
    config.reset_settings_cache()
    settings = config.get_settings()

    config.ensure_runtime_directories(settings)
    assert settings.output_dir.exists()
    assert settings.upload_dir.exists()
    assert settings.logs_dir.exists()
    assert settings.database_path.parent.exists()
    assert settings.hf_home.exists()
    assert settings.hf_datasets_cache.exists()

    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    config.apply_runtime_environment(settings)

    assert Path(os.environ["HF_HOME"]) == settings.hf_home
    assert Path(os.environ["HF_DATASETS_CACHE"]) == settings.hf_datasets_cache
    assert Path(os.environ["HUGGINGFACE_HUB_CACHE"]) == settings.hf_home / "hub"
