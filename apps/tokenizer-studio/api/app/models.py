from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class TrainTokenizerRequest(BaseModel):
    tokenizer_config: dict[str, Any]
    dataloader_config: dict[str, Any]
    evaluation_thresholds: list[int] = Field(default_factory=lambda: [5, 10, 25])
    evaluation_text_path: str | None = None

    @field_validator("evaluation_thresholds")
    @classmethod
    def validate_thresholds(cls, values: list[int]) -> list[int]:
        if not values:
            raise ValueError("evaluation_thresholds must include at least one value")
        if any(v <= 0 for v in values):
            raise ValueError("evaluation_thresholds values must all be positive")
        deduped = sorted(set(values))
        return deduped


class ValidateConfigRequest(BaseModel):
    config: dict[str, Any]


class HealthResponse(BaseModel):
    ok: bool = True


class TokenizerStatsResponse(BaseModel):
    num_chars: int
    num_tokens: int
    token_per_char: float
    vocab_size: int
    num_used_tokens: int
    num_unused_tokens: int
    rare_tokens: dict[int, int]
    rare_token_fraction: dict[int, float]


class TrainingJobResponse(BaseModel):
    id: str
    status: JobStatus
    stage: str
    progress: float
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    tokenizer_config: dict[str, Any]
    dataloader_config: dict[str, Any]
    evaluation_thresholds: list[int]
    evaluation_text_path: str
    artifact_file: str | None = None
    artifact_path: str | None = None
    stats: TokenizerStatsResponse | None = None
    error: str | None = None


class TrainingJobsListResponse(BaseModel):
    jobs: list[TrainingJobResponse]


class ConfigTemplatesResponse(BaseModel):
    tokenizer_config_template: dict[str, Any]
    dataloader_config_template: dict[str, Any]


class ConfigSchemasResponse(BaseModel):
    tokenizer_schema: dict[str, Any]
    dataloader_schema: dict[str, Any]


class ValidateConfigResponse(BaseModel):
    valid: bool = True
    normalized_config: dict[str, Any]


class ArtifactMetadataResponse(BaseModel):
    job_id: str
    artifact_file: str
    artifact_path: str
    exists: bool
    size_bytes: int

    @classmethod
    def from_path(cls, job_id: str, path: Path) -> "ArtifactMetadataResponse":
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else 0
        return cls(
            job_id=job_id,
            artifact_file=path.name,
            artifact_path=str(path),
            exists=exists,
            size_bytes=size_bytes,
        )
