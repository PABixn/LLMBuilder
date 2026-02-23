from __future__ import annotations

from pathlib import Path

from app import config


def test_settings_resolve_from_data_root(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "studio-data"

    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(data_root))
    monkeypatch.setenv("LLM_STUDIO_PROJECTS_DIR", "projects")
    monkeypatch.setenv("LLM_STUDIO_WEB_DIST_DIR", "web-dist")
    monkeypatch.setenv("LLM_STUDIO_PORT", "9052")

    config.reset_settings_cache()
    settings = config.get_settings()

    assert settings.data_dir == data_root
    assert settings.projects_dir == data_root / "projects"
    assert settings.web_dist_dir == config.API_ROOT / "web-dist"
    assert settings.port == 9052


def test_runtime_directories(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "studio-data"
    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(data_root))
    config.reset_settings_cache()
    settings = config.get_settings()

    config.ensure_runtime_directories(settings)
    config.apply_runtime_environment(settings)

    assert settings.data_dir.exists()
    assert settings.projects_dir.exists()
