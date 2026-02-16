from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import ValidationError

from .config import (
    DATALOADER_CONFIG_TEMPLATE_PATH,
    DATALOADER_SCHEMA_PATH,
    TOKENIZER_CONFIG_TEMPLATE_PATH,
    TOKENIZER_SCHEMA_PATH,
)
from .jobs import TrainingJobManager
from .models import (
    ArtifactMetadataResponse,
    ConfigSchemasResponse,
    ConfigTemplatesResponse,
    HealthResponse,
    TrainTokenizerRequest,
    TrainingJobResponse,
    TrainingJobsListResponse,
    ValidateConfigRequest,
    ValidateConfigResponse,
)
from .schemas import load_json

IMPORT_ROOT = Path(__file__).resolve().parents[4]
if str(IMPORT_ROOT) not in sys.path:
    sys.path.append(str(IMPORT_ROOT))

from tokenizer.dataloader_config import DataloaderConfig
from tokenizer.loader import TokenizerConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.jobs = TrainingJobManager()
    yield
    app.state.jobs.shutdown()


app = FastAPI(
    title="Tokenizer Studio API",
    version="0.1.0",
    description="Local API for configuring and training tokenizer artifacts.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter(prefix="/api/v1", tags=["tokenizer-studio"])


@api.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@api.get("/config/templates", response_model=ConfigTemplatesResponse)
def config_templates() -> ConfigTemplatesResponse:
    return ConfigTemplatesResponse(
        tokenizer_config_template=load_json(TOKENIZER_CONFIG_TEMPLATE_PATH),
        dataloader_config_template=load_json(DATALOADER_CONFIG_TEMPLATE_PATH),
    )


@api.get("/config/schemas", response_model=ConfigSchemasResponse)
def config_schemas() -> ConfigSchemasResponse:
    return ConfigSchemasResponse(
        tokenizer_schema=load_json(TOKENIZER_SCHEMA_PATH),
        dataloader_schema=load_json(DATALOADER_SCHEMA_PATH),
    )


@api.post("/validate/tokenizer", response_model=ValidateConfigResponse)
def validate_tokenizer(payload: ValidateConfigRequest) -> ValidateConfigResponse:
    try:
        normalized = TokenizerConfig.model_validate(payload.config).model_dump(mode="json")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return ValidateConfigResponse(normalized_config=normalized)


@api.post("/validate/dataloader", response_model=ValidateConfigResponse)
def validate_dataloader(payload: ValidateConfigRequest) -> ValidateConfigResponse:
    try:
        normalized = DataloaderConfig.model_validate(payload.config).model_dump(mode="json")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return ValidateConfigResponse(normalized_config=normalized)


@api.post("/jobs", response_model=TrainingJobResponse, status_code=201)
def create_job(payload: TrainTokenizerRequest) -> TrainingJobResponse:
    manager = app.state.jobs
    try:
        return manager.create_job(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.get("/jobs", response_model=TrainingJobsListResponse)
def list_jobs() -> TrainingJobsListResponse:
    manager = app.state.jobs
    return TrainingJobsListResponse(jobs=manager.list_jobs())


@api.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def get_job(job_id: str) -> TrainingJobResponse:
    manager = app.state.jobs
    try:
        return manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job id: {job_id}") from exc


@api.get("/jobs/{job_id}/artifact/meta", response_model=ArtifactMetadataResponse)
def artifact_meta(job_id: str) -> ArtifactMetadataResponse:
    manager = app.state.jobs
    try:
        path = manager.get_artifact_path(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job id: {job_id}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ArtifactMetadataResponse.from_path(job_id, path)


@api.get("/jobs/{job_id}/artifact")
def artifact_download(job_id: str) -> FileResponse:
    manager = app.state.jobs
    try:
        path = manager.get_artifact_path(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job id: {job_id}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, filename=path.name, media_type="application/json")


app.include_router(api)
