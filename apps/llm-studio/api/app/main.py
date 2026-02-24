from __future__ import annotations

from collections import Counter
import re
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from .config import (
    MODEL_CONFIG_TEMPLATE_PATH,
    MODEL_SCHEMA_PATH,
    apply_runtime_environment,
    ensure_runtime_directories,
    get_settings,
)
from .models import (
    AnalyzeModelRequest,
    AnalyzeModelResponse,
    ConfigSchemasResponse,
    ConfigTemplatesResponse,
    CreateProjectRequest,
    HealthResponse,
    ModelAnalysisSummary,
    ProjectDetailResponse,
    ProjectsListResponse,
    ProjectSummaryResponse,
    ValidateModelRequest,
    ValidateModelResponse,
    ValidationIssue,
)
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
from model.model import ConfigurableGPT

_PROJECT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{6,64}$")
_METADATA_FILE = "metadata.json"
_ARTIFACT_FILE = "model_config.json"


class RuntimeTokenMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        path = request.url.path
        if not path.startswith("/api/v1"):
            return await call_next(request)
        if path == "/api/v1/health":
            return await call_next(request)

        provided = request.headers.get("X-LLM-Studio-Token")
        if provided != self._token:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    apply_runtime_environment(settings)
    ensure_runtime_directories(settings)
    app.state.settings = settings
    yield


app = FastAPI(
    title="LLM Studio API",
    version="0.1.0",
    description="Local API for validating and storing LLM model configs.",
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


app.include_router(api)

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
