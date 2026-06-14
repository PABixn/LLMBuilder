from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SOURCE_ROOT_ENV = "LLM_STUDIO_SOURCE_ROOT"


class RuntimeLayoutError(RuntimeError):
    """Raised when immutable packaged runtime resources are incomplete."""


@dataclass(frozen=True)
class RuntimeResources:
    source_root: Path
    api_root: Path
    template_dir: Path
    model_dir: Path
    tokenizer_dir: Path
    training_dir: Path

    def required_paths(self) -> tuple[Path, ...]:
        return (
            self.api_root / "app",
            self.template_dir,
            self.model_dir,
            self.tokenizer_dir,
            self.training_dir,
            self.template_dir / "model_config.json",
            self.template_dir / "model_config_schema.json",
            self.template_dir / "tok_config.json",
            self.template_dir / "tokenizer_config_schema.json",
            self.template_dir / "dataloader_config.json",
            self.template_dir / "dataloader_config_schema.json",
            self.training_dir / "training_config.json",
            self.training_dir / "training_config_schema.json",
            self.training_dir / "dataloader_config.json",
            self.training_dir / "dataloader_config_schema.json",
        )


def reset_runtime_paths_cache() -> None:
    runtime_resources.cache_clear()


@lru_cache(maxsize=1)
def runtime_resources() -> RuntimeResources:
    source_root = _resolve_source_root()
    api_root = source_root / "apps" / "llm-studio" / "api"
    return RuntimeResources(
        source_root=source_root,
        api_root=api_root,
        template_dir=api_root / "templates",
        model_dir=source_root / "model",
        tokenizer_dir=source_root / "tokenizer",
        training_dir=source_root / "training",
    )


def source_root() -> Path:
    return runtime_resources().source_root


def api_root() -> Path:
    return runtime_resources().api_root


def template_dir() -> Path:
    return runtime_resources().template_dir


def ensure_source_root_on_path() -> Path:
    root = source_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def validate_runtime_resources() -> RuntimeResources:
    resources = runtime_resources()
    missing = [path for path in resources.required_paths() if not path.exists()]
    if missing:
        rendered = "\n".join(f"- {path}" for path in missing)
        raise RuntimeLayoutError(
            "LLM Studio runtime is missing required immutable resources:\n"
            f"{rendered}\n"
            f"Resolved source root: {resources.source_root}"
        )
    return resources


def _resolve_source_root() -> Path:
    explicit = os.getenv(SOURCE_ROOT_ENV)
    if explicit is not None and explicit.strip():
        candidate = Path(explicit.strip()).expanduser().resolve()
        if not candidate.is_dir():
            raise RuntimeLayoutError(
                f"{SOURCE_ROOT_ENV} points to a missing directory: {candidate}"
            )
        return candidate

    return Path(__file__).resolve().parents[4]
