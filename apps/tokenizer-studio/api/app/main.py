from __future__ import annotations

import codecs
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import ValidationError

from .config import (
    DATALOADER_CONFIG_TEMPLATE_PATH,
    DATALOADER_SCHEMA_PATH,
    TOKENIZER_CONFIG_TEMPLATE_PATH,
    TOKENIZER_SCHEMA_PATH,
    apply_runtime_environment,
    ensure_runtime_directories,
    get_settings,
    upload_dir,
)
from .jobs import TrainingJobManager
from .models import (
    ArtifactMetadataResponse,
    ConfigSchemasResponse,
    ConfigTemplatesResponse,
    HealthResponse,
    TokenizerPreviewRequest,
    TokenizerPreviewResponse,
    TrainTokenizerRequest,
    TrainingJobResponse,
    TrainingJobsListResponse,
    UploadedTrainFileResponse,
    ValidateConfigRequest,
    ValidateConfigResponse,
)
from .schemas import load_json
from .storage import StudioStore

IMPORT_ROOT = Path(__file__).resolve().parents[4]
if str(IMPORT_ROOT) not in sys.path:
    sys.path.append(str(IMPORT_ROOT))

from tokenizer.dataloader_config import DataloaderConfig
from tokenizer.loader import TokenizerConfig

_FILENAME_SANITIZER = re.compile(r"[^a-zA-Z0-9._-]+")


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

        provided = request.headers.get("X-Tokenizer-Studio-Token")
        if provided != self._token:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)


def _sanitize_uploaded_filename(name: str) -> str:
    base_name = Path(name).name
    sanitized = _FILENAME_SANITIZER.sub("-", base_name).strip("-")
    return sanitized if sanitized else "train.txt"


async def _store_uploaded_text_file(file: UploadFile) -> UploadedTrainFileResponse:
    original_name = (file.filename or "").strip()
    if original_name == "":
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")

    target_dir = upload_dir()
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    apply_runtime_environment(settings)
    ensure_runtime_directories(settings)

    store = StudioStore()
    store.initialize()
    store.mark_incomplete_jobs_failed(
        "Training was interrupted because the API restarted before completion."
    )

    app.state.settings = settings
    app.state.store = store
    app.state.jobs = TrainingJobManager(store=store)
    yield
    app.state.jobs.shutdown()
    app.state.store.dispose()


app = FastAPI(
    title="Tokenizer Studio API",
    version="0.1.0",
    description="Local API for configuring and training tokenizer artifacts.",
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

api = APIRouter(prefix="/api/v1", tags=["tokenizer-studio"])


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_root() -> HealthResponse:
    return HealthResponse()


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


@api.post("/files/train", response_model=UploadedTrainFileResponse, status_code=201)
async def upload_train_file(file: UploadFile = File(...)) -> UploadedTrainFileResponse:
    uploaded = await _store_uploaded_text_file(file)
    app.state.store.record_uploaded_file(
        "train",
        uploaded.file_name,
        uploaded.file_path,
        uploaded.size_bytes,
    )
    return uploaded


@api.get("/files/stats", response_model=UploadedTrainFileResponse)
def get_local_file_stats(file_path: str) -> UploadedTrainFileResponse:
    return _read_text_file_stats(file_path)


@api.post("/files/validation", response_model=UploadedTrainFileResponse, status_code=201)
async def upload_validation_file(file: UploadFile = File(...)) -> UploadedTrainFileResponse:
    uploaded = await _store_uploaded_text_file(file)
    app.state.store.record_uploaded_file(
        "validation",
        uploaded.file_name,
        uploaded.file_path,
        uploaded.size_bytes,
    )
    return uploaded


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


@api.post("/jobs/{job_id}/preview", response_model=TokenizerPreviewResponse)
def preview_job_tokenizer(job_id: str, payload: TokenizerPreviewRequest) -> TokenizerPreviewResponse:
    manager = app.state.jobs
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
