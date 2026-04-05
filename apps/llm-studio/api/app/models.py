from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    ok: bool = True


class ConfigTemplatesResponse(BaseModel):
    model_config_template: dict[str, Any]


class ConfigSchemasResponse(BaseModel):
    model_config_schema: dict[str, Any]


class ValidateModelRequest(BaseModel):
    config: dict[str, Any]


class ValidationIssue(BaseModel):
    code: str
    message: str
    path: str


class ValidateModelResponse(BaseModel):
    valid: bool = True
    normalized_config: dict[str, Any]
    warnings: list[ValidationIssue] = Field(default_factory=list)
    errors: list[ValidationIssue] = Field(default_factory=list)


class AnalyzeModelRequest(BaseModel):
    config: dict[str, Any]


class ParameterBreakdownEntry(BaseModel):
    key: str
    label: str
    parameters: int
    trainable_parameters: int
    module_count: int
    percentage: float
    trainable_percentage: float


class ModelAnalysisSummary(BaseModel):
    total_parameters: int
    trainable_parameters: int
    parameter_memory_bytes_fp32: int
    parameter_memory_bytes_bf16: int
    estimated_kv_cache_bytes_per_token_fp16: int
    estimated_kv_cache_bytes_for_context_fp16: int
    block_count: int
    component_count: int
    attention_component_count: int
    mlp_component_count: int
    norm_component_count: int
    activation_component_count: int
    mlp_activation_step_count: int
    min_head_dim: int | None = None
    max_head_dim: int | None = None
    instantiation_time_ms: float
    module_counts: dict[str, int] = Field(default_factory=dict)
    parameter_breakdown: list[ParameterBreakdownEntry] = Field(default_factory=list)


class AnalyzeModelResponse(BaseModel):
    valid: bool = True
    normalized_config: dict[str, Any]
    warnings: list[ValidationIssue] = Field(default_factory=list)
    errors: list[ValidationIssue] = Field(default_factory=list)
    instantiated: bool = False
    analysis: ModelAnalysisSummary | None = None
    instantiation_error: str | None = None


class CreateProjectRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_payload: dict[str, Any] = Field(alias="model_config")
    name: str | None = Field(default=None, max_length=200)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class ProjectSummaryResponse(BaseModel):
    id: str
    name: str | None = None
    created_at: datetime
    artifact_file: str
    artifact_path: str
    size_bytes: int


class ProjectDetailResponse(ProjectSummaryResponse):
    model_config = ConfigDict(populate_by_name=True)

    model_payload: dict[str, Any] = Field(alias="model_config")
    valid: bool = True
    warnings: list[ValidationIssue] = Field(default_factory=list)
    errors: list[ValidationIssue] = Field(default_factory=list)


class ProjectsListResponse(BaseModel):
    projects: list[ProjectSummaryResponse]
