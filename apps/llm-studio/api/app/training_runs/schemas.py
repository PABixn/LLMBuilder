from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

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


class TrainingExecutorKind(str, Enum):
    local = "local"
    runpod_pod = "runpod_pod"


class RunPodCloudType(str, Enum):
    secure = "SECURE"
    community = "COMMUNITY"


class RunPodCleanupPolicy(StrictModel):
    pod: Literal["delete_after_sync", "stop_after_sync", "keep"] = "delete_after_sync"
    network_volume: Literal["keep"] = "keep"

    @field_validator("network_volume", mode="before")
    @classmethod
    def normalize_network_volume_policy(cls, _value: Any) -> str:
        # The RunPod integration uses the Pod's attached volume and does not
        # create a separately managed network volume resource. Keep accepting
        # legacy payloads but normalize them to the only supported behavior.
        return "keep"


class TrainingExecutionTarget(StrictModel):
    kind: TrainingExecutorKind = TrainingExecutorKind.local
    api_key: str | None = Field(default=None, min_length=1)
    gpu_type_id: str | None = None
    gpu_count: int | None = Field(default=None, ge=1)
    cloud_type: RunPodCloudType | None = None
    data_center_id: str | None = None
    interruptible: bool = False
    network_volume_size_gb: int | None = Field(default=None, ge=1)
    cleanup_policy: RunPodCleanupPolicy = Field(default_factory=RunPodCleanupPolicy)


class RunPodProviderDefaults(BaseModel):
    gpu_type_id: str
    gpu_count: int
    cloud_type: RunPodCloudType
    data_center_id: str | None = None
    network_volume_size_gb: int
    container_disk_gb: int
    volume_mount_path: str
    training_image: str
    agent_port: int
    agent_port_protocol: Literal["tcp", "http"]
    cleanup_policy: RunPodCleanupPolicy


class RunPodProviderStatus(BaseModel):
    configured: bool
    validated: bool
    source: Literal["environment", "memory", "none"]
    defaults: RunPodProviderDefaults


class RunPodGpuCatalogItem(BaseModel):
    id: str
    display_name: str
    memory_gb: int | None = None


class RunPodProviderCatalog(BaseModel):
    gpu_options: list[RunPodGpuCatalogItem] = Field(default_factory=list)


class RunPodValidateKeyRequest(StrictModel):
    api_key: str = Field(min_length=1)


class RunPodValidateKeyResponse(BaseModel):
    valid: bool
    message: str
    account: dict[str, Any] | None = None


class RunPodResourceListResponse(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)


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


class TrainingBatchLrRecommendationOption(BaseModel):
    key: str
    label: str
    description: str
    tone: Literal["recommended", "neutral"] = "neutral"
    total_batch_size: int
    micro_batch_size: int
    grad_accum_steps: int
    learning_rate: float
    estimated_tokens_per_run: int
    recommended_max_steps: int
    estimated_tokens_per_recommended_run: int
    estimated_local_passes_at_recommended_steps: float | None = None
    clear_manual_micro_batch: bool = True


class TrainingBatchLrRecommendationFactor(BaseModel):
    code: str
    label: str
    detail: str
    tone: Literal["good", "neutral", "warning"] = "neutral"


class TrainingBatchLrRecommendationSignals(BaseModel):
    device: str
    device_type: str
    total_parameters: int
    parameter_memory_bytes_bf16: int
    estimated_kv_cache_bytes_for_context_fp16: int
    block_count: int
    attention_component_count: int
    max_mlp_multiplier: float
    dataset_count: int
    local_dataset_count: int
    streaming_dataset_count: int
    local_file_count: int
    local_total_size_bytes: int | None = None
    approx_local_tokens: int | None = None
    dominant_dataset_weight: float
    dataset_scale: str
    schedule_peak_factor: float
    warmup_fraction: float
    max_memory_micro_batch_size: int
    recommended_batch_target: int
    recommended_run_token_budget: int
    parameter_scaled_run_token_target: int


class TrainingBatchLrRecommendation(BaseModel):
    headline: str
    summary: str
    confidence: Literal["high", "medium", "low"] = "medium"
    current_total_batch_size: int
    current_learning_rate: float
    current_micro_batch_size: int | None = None
    current_grad_accum_steps: int | None = None
    recommended_option_key: str
    options: list[TrainingBatchLrRecommendationOption] = Field(default_factory=list)
    factors: list[TrainingBatchLrRecommendationFactor] = Field(default_factory=list)
    signals: TrainingBatchLrRecommendationSignals


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
    batch_and_lr_recommendation: TrainingBatchLrRecommendation | None = None


class CreateTrainingJobRequest(TrainingPreflightRequest):
    name: str | None = Field(default=None, max_length=200)
    hf_token: str | None = Field(default=None, min_length=1)
    execution_target: TrainingExecutionTarget = Field(default_factory=TrainingExecutionTarget)

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


class TrainingGenerateRequest(StrictModel):
    prompt: str = Field(min_length=1, max_length=50_000)
    checkpoint_step: int | None = Field(default=None, ge=0)
    max_tokens: int = Field(default=64, ge=1, le=1024)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_k: int | None = Field(default=50, ge=1, le=50_000)
    seed: int = Field(default=42, ge=0)
    repetition_penalty: float = Field(default=1.0, gt=0.0, le=5.0)


class TrainingGenerateResponse(BaseModel):
    job_id: str
    checkpoint_step: int
    checkpoint_path: str
    tokenizer_job_id: str
    prompt: str
    completion: str
    text: str
    prompt_token_count: int
    generated_token_count: int
    generated_token_ids: list[int]


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
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    latest_loss: float | None = None
    latest_grad_norm: float | None = None
    latest_lr: float | None = None
    latest_tokens_per_sec: float | None = None
    checkpoint_count: int = 0
    sample_count: int = 0
    error: str | None = None
    process_id: int | None = None
    output_size_bytes: int = 0
    executor_kind: TrainingExecutorKind = TrainingExecutorKind.local
    executor_status: str | None = None
    runpod_pod_id: str | None = None
    runpod_pod_name: str | None = None
    runpod_network_volume_id: str | None = None
    runpod_data_center_id: str | None = None
    runpod_gpu_type_id: str | None = None
    runpod_gpu_count: int = 1
    runpod_cloud_type: str | None = None
    runpod_interruptible: bool = False
    runpod_cost_per_hr: float | None = None
    runpod_public_ip: str | None = None
    runpod_port_mappings: dict[str, Any] | None = None
    runpod_agent_base_url: str | None = None
    runpod_last_heartbeat_at: datetime | None = None
    runpod_last_sync_at: datetime | None = None
    runpod_cleanup_policy: dict[str, Any] | None = None
    remote_workspace_path: str | None = None
    remote_error: str | None = None


class TrainingJobsListResponse(BaseModel):
    jobs: list[TrainingJobResponse]
