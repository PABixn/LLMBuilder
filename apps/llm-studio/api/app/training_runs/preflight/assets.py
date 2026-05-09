from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ...schemas import load_json
from ...tokenizer_storage import StudioStore as TokenizerStudioStore
from ..identifiers import JOB_ID_RE, PROJECT_ID_RE, validate_identifier
from ..schemas import TrainingAssetRef

IMPORT_ROOT = Path(__file__).resolve().parents[6]
if str(IMPORT_ROOT) not in sys.path:
    sys.path.append(str(IMPORT_ROOT))

from model.loader import LLMConfig


def load_project_asset(project_id: str, *, projects_dir: Path) -> tuple[TrainingAssetRef, dict[str, Any]]:
    validate_identifier(project_id, PROJECT_ID_RE)
    project_dir = projects_dir / project_id
    metadata_path = project_dir / "metadata.json"
    artifact_path = project_dir / "model_config.json"
    if not metadata_path.exists() or not artifact_path.exists():
        raise KeyError(project_id)
    metadata = load_json(metadata_path)
    model_config = load_json(artifact_path)
    model = load_config_dict(model_config)
    name = metadata.get("name") if isinstance(metadata.get("name"), str) and metadata.get("name") else f"Project {project_id[:8]}"
    return (
        TrainingAssetRef(
            id=project_id,
            name=name,
            artifact_path=str(artifact_path.resolve()),
            artifact_file=artifact_path.name,
            status="READY",
        ),
        model.model_dump(mode="json"),
    )


def load_tokenizer_asset(
    tokenizer_job_id: str,
    *,
    tokenizer_store: TokenizerStudioStore,
) -> tuple[TrainingAssetRef, Path]:
    validate_identifier(tokenizer_job_id, JOB_ID_RE)
    tokenizer_job = tokenizer_store.get_job(tokenizer_job_id)
    if tokenizer_job is None:
        raise KeyError(tokenizer_job_id)
    if tokenizer_job.status.value != "completed":
        raise ValueError("Tokenizer job must be completed before training can start.")
    artifact_path = require_tokenizer_artifact_path(tokenizer_job_id, tokenizer_store=tokenizer_store)
    return (
        TrainingAssetRef(
            id=tokenizer_job_id,
            name=tokenizer_display_name(tokenizer_job.tokenizer_config, tokenizer_job_id, tokenizer_job.artifact_file),
            artifact_path=str(artifact_path.resolve()),
            artifact_file=artifact_path.name,
            status=tokenizer_job.status.value,
        ),
        artifact_path,
    )


def require_tokenizer_artifact_path(tokenizer_job_id: str, *, tokenizer_store: TokenizerStudioStore) -> Path:
    tokenizer_job = tokenizer_store.get_job(tokenizer_job_id)
    if tokenizer_job is None or tokenizer_job.artifact_path is None:
        raise KeyError(tokenizer_job_id)
    artifact_path = Path(tokenizer_job.artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Tokenizer artifact missing: {artifact_path}")
    return artifact_path


def load_config_dict(payload: dict[str, Any]) -> LLMConfig:
    return LLMConfig.model_validate(payload)


def tokenizer_display_name(
    tokenizer_config: dict[str, Any],
    job_id: str,
    artifact_file: str | None,
) -> str:
    name = tokenizer_config.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    if isinstance(artifact_file, str) and artifact_file.strip():
        return artifact_file.strip()
    return f"Tokenizer {job_id[:8]}"
