from __future__ import annotations

import codecs
from collections import Counter, defaultdict
import re
import shutil
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
import torch.nn as nn

from .config import (
    DATALOADER_CONFIG_TEMPLATE_PATH,
    DATALOADER_SCHEMA_PATH,
    MODEL_CONFIG_TEMPLATE_PATH,
    MODEL_SCHEMA_PATH,
    TOKENIZER_CONFIG_TEMPLATE_PATH,
    TOKENIZER_SCHEMA_PATH,
    apply_runtime_environment,
    ensure_runtime_directories,
    get_settings,
    tokenizer_upload_dir,
)
from .models import (
    AnalyzeModelRequest,
    AnalyzeModelResponse,
    ConfigSchemasResponse,
    ConfigTemplatesResponse,
    CreateProjectRequest,
    HealthResponse,
    ModelAnalysisSummary,
    ParameterBreakdownEntry,
    ProjectDetailResponse,
    ProjectsListResponse,
    ProjectSummaryResponse,
    ValidateModelRequest,
    ValidateModelResponse,
    ValidationIssue,
)
from .tokenizer_jobs import TrainingJobManager
from .tokenizer_models import (
    ArtifactMetadataResponse,
    ConfigSchemasResponse as TokenizerConfigSchemasResponse,
    ConfigTemplatesResponse as TokenizerConfigTemplatesResponse,
    TokenizerPreviewRequest,
    TokenizerPreviewResponse,
    TrainTokenizerRequest,
    TrainingJobResponse,
    TrainingJobsListResponse,
    UploadedTrainFileResponse,
    ValidateConfigRequest,
    ValidateConfigResponse,
)
from .tokenizer_storage import StudioStore as TokenizerStudioStore
from .schemas import load_json, write_json

IMPORT_ROOT = Path(__file__).resolve().parents[4]
if str(IMPORT_ROOT) not in sys.path:
    sys.path.append(str(IMPORT_ROOT))

from model.loader import (
    ActivationComponent,
    AttentionComponent,
    LLMConfig,
    MLPComponent,
    NormComponent,
)
from model.model import (
    CausalSelfAttention,
    ConfigurableGPT,
    ConfigurableMLP,
    LearnableRMSNorm,
    StaticRMSNorm,
)
from tokenizer.dataloader_config import DataloaderConfig
from tokenizer.loader import TokenizerConfig

_PROJECT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{6,64}$")
_METADATA_FILE = "metadata.json"
_ARTIFACT_FILE = "model_config.json"
_FILENAME_SANITIZER = re.compile(r"[^a-zA-Z0-9._-]+")

_PARAMETER_BREAKDOWN_LABELS: dict[str, str] = {
    "embeddings": "Token Embeddings",
    "embeddings_tied": "Token Embedding / LM Head (Tied)",
    "attention_projections": "Attention Projections",
    "attention_norms": "Attention Norms",
    "mlp_projections": "MLP Projections",
    "mlp_norms": "MLP Norms",
    "model_norms": "Model Norms",
    "output_projection": "LM Head Projection",
    "linear_layers": "Other Linear Layers",
    "other": "Other Parameters",
}


class RuntimeTokenMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        path = request.url.path
        if not path.startswith("/api/v1"):
            return await call_next(request)
        if path in {"/api/v1/health", "/api/v1/tokenizer/health"}:
            return await call_next(request)

        provided = (
            request.headers.get("X-Studio-Token")
            or request.headers.get("X-LLM-Studio-Token")
            or request.headers.get("X-Tokenizer-Studio-Token")
        )
        if provided != self._token:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    apply_runtime_environment(settings)
    ensure_runtime_directories(settings)
    tokenizer_store = TokenizerStudioStore()
    tokenizer_store.initialize()
    tokenizer_store.mark_incomplete_jobs_failed(
        "Training was interrupted because the API restarted before completion."
    )

    app.state.settings = settings
    app.state.tokenizer_store = tokenizer_store
    app.state.tokenizer_jobs = TrainingJobManager(store=tokenizer_store)
    yield
    app.state.tokenizer_jobs.shutdown()
    app.state.tokenizer_store.dispose()


app = FastAPI(
    title="LLM Builder Studio API",
    version="0.1.0",
    description="Local API for model architecture design and tokenizer training.",
    lifespan=lifespan,
)

_settings = get_settings()
if _settings.runtime_token is not None:
    app.add_middleware(RuntimeTokenMiddleware, token=_settings.runtime_token)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_settings.cors_allowed_origins),
    allow_origin_regex=_settings.cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter(prefix="/api/v1", tags=["llm-studio"])
tokenizer_api = APIRouter(prefix="/api/v1/tokenizer", tags=["tokenizer-workspace"])


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_root() -> HealthResponse:
    return HealthResponse()


@api.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@api.get("/config/templates", response_model=ConfigTemplatesResponse)
def config_templates() -> ConfigTemplatesResponse:
    return ConfigTemplatesResponse(model_config_template=load_json(MODEL_CONFIG_TEMPLATE_PATH))


@api.get("/config/schemas", response_model=ConfigSchemasResponse)
def config_schemas() -> ConfigSchemasResponse:
    return ConfigSchemasResponse(model_config_schema=load_json(MODEL_SCHEMA_PATH))


def _sanitize_uploaded_filename(name: str) -> str:
    base_name = Path(name).name
    sanitized = _FILENAME_SANITIZER.sub("-", base_name).strip("-")
    return sanitized if sanitized else "train.txt"


async def _store_uploaded_text_file(file: UploadFile) -> UploadedTrainFileResponse:
    original_name = (file.filename or "").strip()
    if original_name == "":
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")

    target_dir = tokenizer_upload_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _sanitize_uploaded_filename(original_name)
    target_path = target_dir / f"{uuid4().hex[:12]}-{safe_name}"
    bytes_written = 0
    chars_written = 0
    utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    try:
        with target_path.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                chars_written += len(utf8_decoder.decode(chunk))
                handle.write(chunk)
            chars_written += len(utf8_decoder.decode(b"", final=True))
    except OSError as exc:
        if target_path.exists():
            target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to store uploaded file") from exc
    finally:
        await file.close()

    if bytes_written == 0:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return UploadedTrainFileResponse(
        file_name=target_path.name,
        file_path=str(target_path.resolve()),
        size_bytes=bytes_written,
        size_chars=chars_written,
    )


def _read_text_file_stats(file_path: str) -> UploadedTrainFileResponse:
    raw_path = file_path.strip()
    if raw_path == "":
        raise HTTPException(status_code=400, detail="file_path is required")

    path = Path(raw_path).expanduser()
    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Invalid file path") from exc

    if not resolved_path.is_file():
        raise HTTPException(status_code=400, detail="Path must point to a file")

    bytes_read = 0
    chars_read = 0
    utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        with resolved_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                bytes_read += len(chunk)
                chars_read += len(utf8_decoder.decode(chunk))
            chars_read += len(utf8_decoder.decode(b"", final=True))
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to read local file") from exc

    return UploadedTrainFileResponse(
        file_name=resolved_path.name,
        file_path=str(resolved_path),
        size_bytes=bytes_read,
        size_chars=chars_read,
    )


@tokenizer_api.get("/health", response_model=HealthResponse)
def tokenizer_health() -> HealthResponse:
    return HealthResponse()


@tokenizer_api.get("/config/templates", response_model=TokenizerConfigTemplatesResponse)
def tokenizer_config_templates() -> TokenizerConfigTemplatesResponse:
    return TokenizerConfigTemplatesResponse(
        tokenizer_config_template=load_json(TOKENIZER_CONFIG_TEMPLATE_PATH),
        dataloader_config_template=load_json(DATALOADER_CONFIG_TEMPLATE_PATH),
    )


@tokenizer_api.get("/config/schemas", response_model=TokenizerConfigSchemasResponse)
def tokenizer_config_schemas() -> TokenizerConfigSchemasResponse:
    return TokenizerConfigSchemasResponse(
        tokenizer_schema=load_json(TOKENIZER_SCHEMA_PATH),
        dataloader_schema=load_json(DATALOADER_SCHEMA_PATH),
    )


@tokenizer_api.post("/validate/tokenizer", response_model=ValidateConfigResponse)
def validate_tokenizer(payload: ValidateConfigRequest) -> ValidateConfigResponse:
    try:
        normalized = TokenizerConfig.model_validate(payload.config).model_dump(mode="json")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return ValidateConfigResponse(normalized_config=normalized)


@tokenizer_api.post("/validate/dataloader", response_model=ValidateConfigResponse)
def validate_dataloader(payload: ValidateConfigRequest) -> ValidateConfigResponse:
    try:
        normalized = DataloaderConfig.model_validate(payload.config).model_dump(mode="json")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return ValidateConfigResponse(normalized_config=normalized)


@tokenizer_api.post("/files/train", response_model=UploadedTrainFileResponse, status_code=201)
async def upload_train_file(file: UploadFile = File(...)) -> UploadedTrainFileResponse:
    uploaded = await _store_uploaded_text_file(file)
    app.state.tokenizer_store.record_uploaded_file(
        "train",
        uploaded.file_name,
        uploaded.file_path,
        uploaded.size_bytes,
    )
    return uploaded


@tokenizer_api.get("/files/stats", response_model=UploadedTrainFileResponse)
def get_local_file_stats(file_path: str) -> UploadedTrainFileResponse:
    return _read_text_file_stats(file_path)


@tokenizer_api.post("/files/validation", response_model=UploadedTrainFileResponse, status_code=201)
async def upload_validation_file(file: UploadFile = File(...)) -> UploadedTrainFileResponse:
    uploaded = await _store_uploaded_text_file(file)
    app.state.tokenizer_store.record_uploaded_file(
        "validation",
        uploaded.file_name,
        uploaded.file_path,
        uploaded.size_bytes,
    )
    return uploaded


@tokenizer_api.post("/jobs", response_model=TrainingJobResponse, status_code=201)
def create_tokenizer_job(payload: TrainTokenizerRequest) -> TrainingJobResponse:
    manager = app.state.tokenizer_jobs
    try:
        return manager.create_job(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@tokenizer_api.get("/jobs", response_model=TrainingJobsListResponse)
def list_tokenizer_jobs() -> TrainingJobsListResponse:
    manager = app.state.tokenizer_jobs
    return TrainingJobsListResponse(jobs=manager.list_jobs())


@tokenizer_api.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def get_tokenizer_job(job_id: str) -> TrainingJobResponse:
    manager = app.state.tokenizer_jobs
    try:
        return manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job id: {job_id}") from exc


@tokenizer_api.delete("/jobs/{job_id}", status_code=204)
def delete_tokenizer_job(job_id: str) -> Response:
    manager = app.state.tokenizer_jobs
    try:
        manager.delete_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job id: {job_id}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(status_code=204)


@tokenizer_api.post("/jobs/{job_id}/preview", response_model=TokenizerPreviewResponse)
def preview_job_tokenizer(job_id: str, payload: TokenizerPreviewRequest) -> TokenizerPreviewResponse:
    manager = app.state.tokenizer_jobs
    try:
        return manager.preview_tokens(job_id, payload.text)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job id: {job_id}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Tokenizer artifact is not ready for job {job_id}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@tokenizer_api.get("/jobs/{job_id}/artifact/meta", response_model=ArtifactMetadataResponse)
def tokenizer_artifact_meta(job_id: str) -> ArtifactMetadataResponse:
    manager = app.state.tokenizer_jobs
    try:
        path = manager.get_artifact_path(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job id: {job_id}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ArtifactMetadataResponse.from_path(job_id, path)


@tokenizer_api.get("/jobs/{job_id}/artifact")
def tokenizer_artifact_download(job_id: str) -> FileResponse:
    manager = app.state.tokenizer_jobs
    try:
        path = manager.get_artifact_path(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job id: {job_id}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, filename=path.name, media_type="application/json")


def _validation_issue(code: str, message: str, path: str) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, path=path)


def _semantic_validate_model(config: LLMConfig) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
    warnings: list[ValidationIssue] = []
    errors: list[ValidationIssue] = []

    for block_index, block in enumerate(config.blocks):
        for component_index, component in enumerate(block.components):
            attention = getattr(component, "attention", None)
            if attention is None:
                continue

            path = f"blocks[{block_index}].components[{component_index}].attention"
            n_head = attention.n_head
            n_kv_head = attention.n_kv_head

            if config.n_embd % n_head != 0:
                errors.append(
                    _validation_issue(
                        "n_embd_not_divisible_by_n_head",
                        f"n_embd ({config.n_embd}) must be divisible by n_head ({n_head}).",
                        path,
                    )
                )
            else:
                head_dim = config.n_embd // n_head
                if head_dim % 2 != 0:
                    warnings.append(
                        _validation_issue(
                            "head_dim_not_even",
                            f"head_dim ({head_dim}) should be even for better kernel compatibility.",
                            path,
                        )
                    )

            if n_kv_head > n_head:
                errors.append(
                    _validation_issue(
                        "n_kv_head_gt_n_head",
                        f"n_kv_head ({n_kv_head}) must be <= n_head ({n_head}).",
                        path,
                    )
                )

            if n_head % n_kv_head != 0:
                errors.append(
                    _validation_issue(
                        "n_head_not_divisible_by_n_kv_head",
                        f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head}).",
                        path,
                    )
                )

    return warnings, errors


def _parse_and_validate_payload_model(
    config_payload: dict[str, object],
) -> tuple[LLMConfig, dict[str, object], list[ValidationIssue], list[ValidationIssue]]:
    try:
        parsed = LLMConfig.model_validate(config_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    normalized = parsed.model_dump(mode="json")
    warnings, errors = _semantic_validate_model(parsed)
    return parsed, normalized, warnings, errors


def _parameter_layer_key(
    *,
    module_name: str,
    module: nn.Module | None,
    component_owner: nn.Module | None,
    weight_tying: bool,
) -> str:
    if module_name == "transformer.wte":
        return "embeddings_tied" if weight_tying else "embeddings"
    if module_name == "lm_head":
        return "output_projection"
    if module_name in {"in_norm", "out_norm"}:
        return "model_norms"

    if isinstance(module, nn.Linear):
        if isinstance(component_owner, CausalSelfAttention):
            return "attention_projections"
        if isinstance(component_owner, ConfigurableMLP):
            return "mlp_projections"
        return "linear_layers"

    if isinstance(module, (LearnableRMSNorm, StaticRMSNorm, nn.LayerNorm)):
        if isinstance(component_owner, CausalSelfAttention):
            return "attention_norms"
        if isinstance(component_owner, ConfigurableMLP):
            return "mlp_norms"
        return "model_norms"

    if isinstance(module, nn.Embedding):
        return "embeddings"

    return "other"


def _find_component_owner(module_name: str, module_lookup: dict[str, nn.Module]) -> nn.Module | None:
    ancestor_name = module_name
    while ancestor_name:
        ancestor_name = ancestor_name.rsplit(".", 1)[0] if "." in ancestor_name else ""
        if not ancestor_name:
            return None
        ancestor_module = module_lookup.get(ancestor_name)
        if isinstance(ancestor_module, (CausalSelfAttention, ConfigurableMLP)):
            return ancestor_module
    return None


def _build_parameter_breakdown(
    *,
    model: ConfigurableGPT,
    total_parameters: int,
    trainable_parameters: int,
    weight_tying: bool,
) -> list[ParameterBreakdownEntry]:
    if total_parameters <= 0:
        return []

    module_lookup = dict(model.named_modules())
    layer_parameter_counts: defaultdict[str, int] = defaultdict(int)
    layer_trainable_parameter_counts: defaultdict[str, int] = defaultdict(int)
    layer_module_names: defaultdict[str, set[str]] = defaultdict(set)

    for parameter_name, parameter in model.named_parameters():
        parameter_count = int(parameter.numel())
        if parameter_count <= 0:
            continue

        module_name = parameter_name.rsplit(".", 1)[0] if "." in parameter_name else ""
        module = module_lookup.get(module_name)
        component_owner = _find_component_owner(module_name, module_lookup)

        key = _parameter_layer_key(
            module_name=module_name,
            module=module,
            component_owner=component_owner,
            weight_tying=weight_tying,
        )
        layer_parameter_counts[key] += parameter_count
        if parameter.requires_grad:
            layer_trainable_parameter_counts[key] += parameter_count
        if module_name:
            layer_module_names[key].add(module_name)

    sorted_items = sorted(
        layer_parameter_counts.items(),
        key=lambda item: (-item[1], _PARAMETER_BREAKDOWN_LABELS.get(item[0], item[0])),
    )

    breakdown: list[ParameterBreakdownEntry] = []
    for key, parameters in sorted_items:
        if parameters <= 0:
            continue
        breakdown.append(
            ParameterBreakdownEntry(
                key=key,
                label=_PARAMETER_BREAKDOWN_LABELS.get(key, key.replace("_", " ").title()),
                parameters=parameters,
                trainable_parameters=layer_trainable_parameter_counts[key],
                module_count=len(layer_module_names[key]),
                percentage=round((parameters / total_parameters) * 100.0, 4),
                trainable_percentage=round(
                    (layer_trainable_parameter_counts[key] / trainable_parameters) * 100.0, 4
                )
                if trainable_parameters > 0
                else 0.0,
            )
        )
    return breakdown


def _build_model_analysis(config: LLMConfig) -> ModelAnalysisSummary:
    attention_count = 0
    mlp_count = 0
    norm_count = 0
    activation_count = 0
    mlp_activation_step_count = 0
    component_count = 0
    kv_cache_bytes_per_token_fp16 = 0
    head_dims: list[int] = []

    for block in config.blocks:
        for component in block.components:
            component_count += 1
            if isinstance(component, AttentionComponent):
                attention_count += 1
                if config.n_embd % component.attention.n_head == 0:
                    head_dim = config.n_embd // component.attention.n_head
                    head_dims.append(head_dim)
                    # K and V cache for batch=1, fp16 values (2 bytes each).
                    kv_cache_bytes_per_token_fp16 += (
                        2 * component.attention.n_kv_head * head_dim * 2
                    )
            elif isinstance(component, MLPComponent):
                mlp_count += 1
                for step in component.mlp.sequence:
                    if isinstance(step, ActivationComponent):
                        mlp_activation_step_count += 1
            elif isinstance(component, NormComponent):
                norm_count += 1
            elif isinstance(component, ActivationComponent):
                activation_count += 1

    started = time.perf_counter()
    model = ConfigurableGPT(config)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_breakdown = _build_parameter_breakdown(
        model=model,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        weight_tying=config.weight_tying,
    )
    module_counts = dict(sorted(Counter(type(module).__name__ for module in model.modules()).items()))

    del model

    return ModelAnalysisSummary(
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        parameter_memory_bytes_fp32=total_parameters * 4,
        parameter_memory_bytes_bf16=total_parameters * 2,
        estimated_kv_cache_bytes_per_token_fp16=kv_cache_bytes_per_token_fp16,
        estimated_kv_cache_bytes_for_context_fp16=kv_cache_bytes_per_token_fp16
        * config.context_length,
        block_count=len(config.blocks),
        component_count=component_count,
        attention_component_count=attention_count,
        mlp_component_count=mlp_count,
        norm_component_count=norm_count,
        activation_component_count=activation_count,
        mlp_activation_step_count=mlp_activation_step_count,
        min_head_dim=min(head_dims) if head_dims else None,
        max_head_dim=max(head_dims) if head_dims else None,
        instantiation_time_ms=round(elapsed_ms, 3),
        module_counts=module_counts,
        parameter_breakdown=parameter_breakdown,
    )


@api.post("/validate/model", response_model=ValidateModelResponse)
def validate_model(payload: ValidateModelRequest) -> ValidateModelResponse:
    _, normalized, warnings, errors = _parse_and_validate_payload_model(payload.config)
    return ValidateModelResponse(
        valid=not errors,
        normalized_config=normalized,
        warnings=warnings,
        errors=errors,
    )


@api.post("/analyze/model", response_model=AnalyzeModelResponse)
def analyze_model(payload: AnalyzeModelRequest) -> AnalyzeModelResponse:
    parsed, normalized, warnings, errors = _parse_and_validate_payload_model(payload.config)
    if errors:
        return AnalyzeModelResponse(
            valid=False,
            normalized_config=normalized,
            warnings=warnings,
            errors=errors,
            instantiated=False,
            analysis=None,
            instantiation_error=None,
        )

    try:
        analysis = _build_model_analysis(parsed)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        instantiation_issue = _validation_issue(
            "model_instantiation_failed",
            f"Config parsed successfully but ConfigurableGPT failed to initialize: {exc}",
            "$",
        )
        return AnalyzeModelResponse(
            valid=False,
            normalized_config=normalized,
            warnings=warnings,
            errors=[*errors, instantiation_issue],
            instantiated=False,
            analysis=None,
            instantiation_error=str(exc),
        )

    return AnalyzeModelResponse(
        valid=True,
        normalized_config=normalized,
        warnings=warnings,
        errors=errors,
        instantiated=True,
        analysis=analysis,
        instantiation_error=None,
    )


def _projects_root() -> Path:
    return get_settings().projects_dir


def _assert_valid_project_id(project_id: str) -> None:
    if not _PROJECT_ID_RE.fullmatch(project_id):
        raise HTTPException(status_code=404, detail="Project not found")


def _project_dir(project_id: str) -> Path:
    _assert_valid_project_id(project_id)
    return _projects_root() / project_id


def _project_metadata_path(project_id: str) -> Path:
    return _project_dir(project_id) / _METADATA_FILE


def _project_artifact_path(project_id: str) -> Path:
    return _project_dir(project_id) / _ARTIFACT_FILE


def _project_summary_from_metadata(metadata: dict[str, object], artifact_path: Path) -> ProjectSummaryResponse:
    if not artifact_path.exists() or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Project artifact not found")

    return ProjectSummaryResponse(
        id=str(metadata["id"]),
        name=metadata.get("name") if isinstance(metadata.get("name"), str) else None,
        created_at=metadata["created_at"],
        artifact_file=artifact_path.name,
        artifact_path=str(artifact_path.resolve()),
        size_bytes=artifact_path.stat().st_size,
    )


def _load_project_summary(project_id: str) -> ProjectSummaryResponse:
    metadata_path = _project_metadata_path(project_id)
    artifact_path = _project_artifact_path(project_id)
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    metadata = load_json(metadata_path)
    return _project_summary_from_metadata(metadata, artifact_path)


def _load_project_detail(project_id: str) -> ProjectDetailResponse:
    summary = _load_project_summary(project_id)
    artifact_path = _project_artifact_path(project_id)
    model_config = load_json(artifact_path)
    _, normalized, warnings, errors = _parse_and_validate_payload_model(model_config)
    return ProjectDetailResponse(
        **summary.model_dump(),
        model_payload=normalized,
        valid=not errors,
        warnings=warnings,
        errors=errors,
    )


@api.post("/projects", response_model=ProjectDetailResponse, status_code=201)
def create_project(payload: CreateProjectRequest) -> ProjectDetailResponse:
    _, normalized, warnings, errors = _parse_and_validate_payload_model(payload.model_payload)

    project_id = uuid4().hex[:12]
    created_at = datetime.now(timezone.utc).isoformat()
    project_dir = _project_dir(project_id)
    project_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "id": project_id,
        "name": payload.name,
        "created_at": created_at,
        "artifact_file": _ARTIFACT_FILE,
    }
    artifact_path = project_dir / _ARTIFACT_FILE

    write_json(project_dir / _METADATA_FILE, metadata)
    write_json(artifact_path, normalized)

    summary = _project_summary_from_metadata(metadata, artifact_path)
    return ProjectDetailResponse(
        **summary.model_dump(),
        model_payload=normalized,
        valid=not errors,
        warnings=warnings,
        errors=errors,
    )


@api.get("/projects", response_model=ProjectsListResponse)
def list_projects() -> ProjectsListResponse:
    root = _projects_root()
    root.mkdir(parents=True, exist_ok=True)

    projects: list[ProjectSummaryResponse] = []
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        project_id = candidate.name
        try:
            projects.append(_load_project_summary(project_id))
        except HTTPException:
            continue
        except (OSError, ValueError, KeyError):
            continue

    projects.sort(key=lambda item: item.created_at, reverse=True)
    return ProjectsListResponse(projects=projects)


@api.get("/projects/{project_id}", response_model=ProjectDetailResponse)
def get_project(project_id: str) -> ProjectDetailResponse:
    try:
        return _load_project_detail(project_id)
    except HTTPException:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc


@api.get("/projects/{project_id}/artifact")
def download_project_artifact(project_id: str) -> FileResponse:
    path = _project_artifact_path(project_id)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Project artifact not found")
    return FileResponse(path, filename=path.name, media_type="application/json")


@api.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: str) -> Response:
    project_dir = _project_dir(project_id)
    if not project_dir.exists() or not project_dir.is_dir():
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        shutil.rmtree(project_dir)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to delete project") from exc
    return Response(status_code=204)


app.include_router(api)
app.include_router(tokenizer_api)

if _settings.serve_web and _settings.web_index_path.exists():

    @app.get("/", include_in_schema=False)
    async def web_index() -> FileResponse:
        return FileResponse(_settings.web_index_path)

    @app.get("/{asset_path:path}", include_in_schema=False)
    async def web_assets(asset_path: str) -> FileResponse:
        if asset_path.startswith("api/") or asset_path == "health":
            raise HTTPException(status_code=404, detail="Not Found")

        if asset_path == "":
            return FileResponse(_settings.web_index_path)

        normalized = Path(asset_path)
        if ".." in normalized.parts:
            raise HTTPException(status_code=404, detail="Not Found")

        candidate = _settings.web_dist_dir / normalized
        if candidate.is_file():
            return FileResponse(candidate)
        if normalized.suffix != "":
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(_settings.web_index_path)
