from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app import runtime_paths


@pytest.fixture(autouse=True)
def reset_path_cache() -> None:
    runtime_paths.reset_runtime_paths_cache()
    yield
    runtime_paths.reset_runtime_paths_cache()


def test_source_tree_runtime_resources_are_complete(monkeypatch) -> None:
    monkeypatch.delenv(runtime_paths.SOURCE_ROOT_ENV, raising=False)

    resources = runtime_paths.validate_runtime_resources()

    assert resources.source_root.name == "LLMBuilder"
    assert resources.model_dir.is_dir()
    assert resources.tokenizer_dir.is_dir()
    assert resources.training_dir.is_dir()


def test_synthetic_packaged_runtime_resolves_without_repository_parents(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / f"Runtime With Spaces \u0142-{'long-' * 16}" / "source"
    repo_root = Path(__file__).resolve().parents[4]
    for relative in (
        Path("apps/llm-studio/api/app"),
        Path("apps/llm-studio/api/templates"),
        Path("model"),
        Path("tokenizer"),
        Path("training"),
    ):
        shutil.copytree(repo_root / relative, source_root / relative)

    monkeypatch.setenv(runtime_paths.SOURCE_ROOT_ENV, str(source_root))

    resources = runtime_paths.validate_runtime_resources()

    assert resources.source_root == source_root.resolve()
    assert runtime_paths.ensure_source_root_on_path() == source_root.resolve()


def test_missing_packaged_runtime_reports_every_required_resource(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "empty-source"
    source_root.mkdir()
    monkeypatch.setenv(runtime_paths.SOURCE_ROOT_ENV, str(source_root))

    with pytest.raises(runtime_paths.RuntimeLayoutError) as exc_info:
        runtime_paths.validate_runtime_resources()

    message = str(exc_info.value)
    assert "missing required immutable resources" in message
    assert str(source_root / "model") in message
    assert str(source_root / "training" / "training_config.json") in message
