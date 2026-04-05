from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainingJobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TrainingJobState(str, Enum):
    queued = "queued"
    preflight = "preflight"
    estimating_memory = "estimating_memory"
    initializing_model = "initializing_model"
    building_dataloader = "building_dataloader"
    training = "training"
    checkpointing = "checkpointing"
    sampling = "sampling"
    finalizing = "finalizing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TrainingConfigTemplatesResponse(BaseModel):
    training_config_template: dict[str, Any]
    dataloader_config_template: dict[str, Any]


class TrainingConfigSchemasResponse(BaseModel):
    training_config_schema: dict[str, Any]
    dataloader_schema: dict[str, Any]


class ValidateConfigRequest(BaseModel):
    config: dict[str, Any]


class ValidateConfigResponse(BaseModel):
    valid: bool = True
    normalized_config: dict[str, Any]


class TrainingAssetRef(BaseModel):
    id: str
    name: str
    artifact_path: str | None = None
    artifact_file: str | None = None
    status: str | None = None


class TrainingIssue(BaseModel):
    code: str
    message: str
    path: str
    severity: str = "error"


class TrainingFixSuggestion(BaseModel):
    code: str
    label: str
    description: str
    path: str
    value: Any | None = None


class TrainingCompatibilitySummary(BaseModel):
    model_context_length: int
    model_vocab_size: int
    tokenizer_vocab_size: int
    seq_len: int
    scheduler_total_steps: int
    max_steps: int
    missing_special_tokens: list[str] = Field(default_factory=list)


class DerivedRuntimeSummary(BaseModel):
    device: str
    device_type: str
    micro_batch_size: int
    tokens_per_micro_step: int
    tokens_per_world_step: int
    grad_accum_steps: int
    max_batch_size_from_total: int
    max_batch_size_from_memory: int
    max_allowed_batch_size: int
    ddp: bool = False
    ddp_rank: int = 0
    ddp_world_size: int = 1


class TrainingPreflightRequest(StrictModel):
    project_id: str
    tokenizer_job_id: str
    training_config: dict[str, Any]
    dataloader_config: dict[str, Any]


class TrainingPreflightResponse(BaseModel):
    valid: bool
    model_project: TrainingAssetRef
    tokenizer: TrainingAssetRef
    normalized_training_config: dict[str, Any]
    normalized_dataloader_config: dict[str, Any]
    warnings: list[TrainingIssue] = Field(default_factory=list)
    errors: list[TrainingIssue] = Field(default_factory=list)
    recommended_fixes: list[TrainingFixSuggestion] = Field(default_factory=list)
    compatibility: TrainingCompatibilitySummary | None = None
    derived_runtime: DerivedRuntimeSummary | None = None
    memory_estimate: dict[str, Any] | None = None


class CreateTrainingJobRequest(TrainingPreflightRequest):
    name: str | None = Field(default=None, max_length=200)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class TrainingMetricPoint(BaseModel):
    step: int
    loss: float | None = None
    norm: float | None = None
    dt: float | None = None
    tok_per_sec: float | None = None
    lr: float | None = None


class TrainingMetricsResponse(BaseModel):
    job_id: str
    metrics: list[TrainingMetricPoint]


class TrainingSampleText(BaseModel):
    index: int
    prompt: str | None = None
    text: str


class TrainingSampleEntry(BaseModel):
    step: int
    samples: list[TrainingSampleText]


class TrainingSamplesResponse(BaseModel):
    job_id: str
    samples: list[TrainingSampleEntry]


class TrainingCheckpointEntry(BaseModel):
    step: int
    directory: str
    created_at: datetime | None = None
    size_bytes: int
    files: list[str] = Field(default_factory=list)


class TrainingCheckpointsResponse(BaseModel):
    job_id: str
    checkpoints: list[TrainingCheckpointEntry]


class TrainingLogsResponse(BaseModel):
    job_id: str
    stdout_lines: list[str] = Field(default_factory=list)
    stderr_lines: list[str] = Field(default_factory=list)


class TrainingJobResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    status: TrainingJobStatus
    state: TrainingJobState
    stage: str
    progress: float
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    project_id: str
    project_name: str
    tokenizer_job_id: str
    tokenizer_name: str
    model_payload: dict[str, Any] = Field(alias="model_config")
    training_config: dict[str, Any]
    dataloader_config: dict[str, Any]
    resolved_runtime: dict[str, Any] | None = None
    memory_estimate: dict[str, Any] | None = None
    artifact_dir: str
    artifact_bundle_file: str | None = None
    stats_path: str
    samples_path: str
    stdout_path: str
    stderr_path: str
    last_step: int = 0
    max_steps: int = 0
    latest_loss: float | None = None
    latest_grad_norm: float | None = None
    latest_lr: float | None = None
    latest_tokens_per_sec: float | None = None
    checkpoint_count: int = 0
    sample_count: int = 0
    error: str | None = None
    process_id: int | None = None
    output_size_bytes: int = 0


class TrainingJobsListResponse(BaseModel):
    jobs: list[TrainingJobResponse]
