from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
import json
import logging
from pathlib import Path
from typing import Any

import anyio
import torch
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import ValidationError
from tokenizers import Tokenizer

from ..config import (
    TRAINING_DATALOADER_SCHEMA_PATH,
    TRAINING_DATALOADER_TEMPLATE_PATH,
    TRAINING_LOOP_SCHEMA_PATH,
    TRAINING_LOOP_TEMPLATE_PATH,
    get_settings,
)
from ..dataset_credentials import strip_hf_tokens
from ..logging_config import redact_value
from ..models import HealthResponse
from ..runtime_paths import ensure_source_root_on_path
from ..schemas import load_json
from ..storage_safety import InsufficientStorageError
from .executors.runpod.client import RunPodClient, RunPodClientError
from .runpod_catalog import build_runpod_provider_catalog
from .schemas import (
    CreateTrainingJobRequest,
    RunPodCleanupPolicy,
    RunPodProviderCatalog,
    RunPodProviderDefaults,
    RunPodProviderStatus,
    RunPodResourceListResponse,
    RunPodValidateKeyRequest,
    RunPodValidateKeyResponse,
    TrainingCheckpointsResponse,
    TrainingConfigSchemasResponse,
    TrainingConfigTemplatesResponse,
    TrainingGenerateRequest,
    TrainingGenerateResponse,
    TrainingJobResponse,
    TrainingJobsListResponse,
    TrainingJobStatus,
    TrainingLogsResponse,
    TrainingMetricsResponse,
    TrainingPreflightRequest,
    TrainingPreflightResponse,
    TrainingSamplesResponse,
    ValidateConfigRequest,
    ValidateConfigResponse,
)

IMPORT_ROOT = ensure_source_root_on_path()

from model.loader import LLMConfig
from model.model import ConfigurableGPT
from training.dataloader_config import TrainingDataloaderConfig
from training.training_config import TrainingConfig
from training.utils import resolve_inference_device_type

training_api = APIRouter(prefix="/api/v1/training", tags=["training-workspace"])
logger = logging.getLogger("llm_studio.training_routes")
_GENERATION_DONE = object()


def register_training_routes(app: FastAPI) -> APIRouter:
    app.include_router(training_api)
    return training_api


@training_api.get("/health", response_model=HealthResponse)
def training_health() -> HealthResponse:
    return HealthResponse()


@training_api.get("/config/templates", response_model=TrainingConfigTemplatesResponse)
def training_config_templates() -> TrainingConfigTemplatesResponse:
    return TrainingConfigTemplatesResponse(
        training_config_template=load_json(TRAINING_LOOP_TEMPLATE_PATH),
        dataloader_config_template=load_json(TRAINING_DATALOADER_TEMPLATE_PATH),
    )


@training_api.get("/config/schemas", response_model=TrainingConfigSchemasResponse)
def training_config_schemas() -> TrainingConfigSchemasResponse:
    return TrainingConfigSchemasResponse(
        training_config_schema=load_json(TRAINING_LOOP_SCHEMA_PATH),
        dataloader_schema=load_json(TRAINING_DATALOADER_SCHEMA_PATH),
    )


@training_api.post("/validate/dataloader", response_model=ValidateConfigResponse)
def validate_training_dataloader(payload: ValidateConfigRequest) -> ValidateConfigResponse:
    try:
        normalized = TrainingDataloaderConfig.model_validate(payload.config).model_dump(mode="json")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return ValidateConfigResponse(normalized_config=strip_hf_tokens(normalized))


@training_api.post("/validate/training-config", response_model=ValidateConfigResponse)
def validate_training_config(payload: ValidateConfigRequest) -> ValidateConfigResponse:
    try:
        normalized = TrainingConfig.model_validate(payload.config).model_dump(mode="json")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return ValidateConfigResponse(normalized_config=normalized)


@training_api.post("/validate/preflight", response_model=TrainingPreflightResponse)
def validate_training_preflight(payload: TrainingPreflightRequest, request: Request) -> TrainingPreflightResponse:
    manager = request.app.state.training_jobs
    try:
        return manager.build_preflight(payload)
    except KeyError as exc:
        missing_id = str(exc).strip("'")
        raise HTTPException(status_code=404, detail=f"Unknown asset id: {missing_id}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InsufficientStorageError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc


def _runpod_defaults() -> RunPodProviderDefaults:
    settings = get_settings()
    return RunPodProviderDefaults(
        gpu_type_id=settings.runpod_default_gpu_type,
        gpu_count=settings.runpod_default_gpu_count,
        cloud_type=settings.runpod_default_cloud_type,  # type: ignore[arg-type]
        data_center_id=settings.runpod_default_data_center_id,
        network_volume_size_gb=settings.runpod_default_volume_size_gb,
        container_disk_gb=settings.runpod_container_disk_gb,
        volume_mount_path=settings.runpod_volume_mount_path,
        training_image=settings.runpod_training_image,
        agent_port=settings.runpod_agent_port,
        agent_port_protocol=settings.runpod_agent_port_protocol,  # type: ignore[arg-type]
        cleanup_policy=RunPodCleanupPolicy(
            pod="delete_after_sync" if settings.runpod_auto_delete_pod else "stop_after_sync",
            network_volume="keep",
        ),
    )


def _runpod_api_key(request: Request) -> tuple[str | None, str]:
    override = getattr(request.app.state, "runpod_api_key_override", None)
    if isinstance(override, str) and override.strip():
        return override.strip(), "memory"
    settings_key = get_settings().runpod_api_key
    if settings_key:
        return settings_key, "environment"
    return None, "none"


@training_api.get("/providers/runpod/defaults", response_model=RunPodProviderDefaults)
def get_runpod_defaults() -> RunPodProviderDefaults:
    return _runpod_defaults()


@training_api.get("/providers/runpod/status", response_model=RunPodProviderStatus)
def get_runpod_status(request: Request) -> RunPodProviderStatus:
    api_key, source = _runpod_api_key(request)
    return RunPodProviderStatus(
        configured=api_key is not None,
        validated=api_key is not None,
        source=source,  # type: ignore[arg-type]
        defaults=_runpod_defaults(),
    )


@training_api.get("/providers/runpod/catalog", response_model=RunPodProviderCatalog)
def get_runpod_catalog() -> RunPodProviderCatalog:
    return build_runpod_provider_catalog(_runpod_defaults().gpu_type_id)


@training_api.post("/providers/runpod/validate-key", response_model=RunPodValidateKeyResponse)
def validate_runpod_key(payload: RunPodValidateKeyRequest, request: Request) -> RunPodValidateKeyResponse:
    api_key = payload.api_key.strip()
    try:
        account = RunPodClient(api_key).validate_key()
    except RunPodClientError as exc:
        _log_provider_event("runpod.key.validation_failed", "RunPod API key validation failed.")
        return RunPodValidateKeyResponse(valid=False, message=str(exc), account=None)
    request.app.state.runpod_api_key_override = api_key
    _log_provider_event("runpod.key.validated", "RunPod API key validated.", source="memory")
    return RunPodValidateKeyResponse(valid=True, message="RunPod API key validated.", account=account)


@training_api.get("/providers/runpod/pods", response_model=RunPodResourceListResponse)
def list_runpod_pods(request: Request) -> RunPodResourceListResponse:
    api_key, source = _runpod_api_key(request)
    if api_key is None:
        _log_provider_event("runpod.pods.unavailable", "RunPod pods unavailable.", source=source)
        raise HTTPException(status_code=409, detail="RunPod API key is not configured.")
    try:
        items = RunPodClient(api_key).list_pods()
    except RunPodClientError as exc:
        _log_provider_event("runpod.pods.list_failed", "RunPod pod listing failed.", source=source)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    _log_provider_event("runpod.pods.listed", "RunPod pods listed.", source=source, count=len(items))
    sanitized_items = redact_value(items)
    return RunPodResourceListResponse(items=sanitized_items if isinstance(sanitized_items, list) else [])


@training_api.get("/providers/runpod/network-volumes", response_model=RunPodResourceListResponse)
def list_runpod_network_volumes(request: Request) -> RunPodResourceListResponse:
    api_key, source = _runpod_api_key(request)
    if api_key is None:
        _log_provider_event("runpod.volumes.unavailable", "RunPod volumes unavailable.", source=source)
        raise HTTPException(status_code=409, detail="RunPod API key is not configured.")
    try:
        items = RunPodClient(api_key).list_network_volumes()
    except RunPodClientError as exc:
        _log_provider_event("runpod.volumes.list_failed", "RunPod volume listing failed.", source=source)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    _log_provider_event(
        "runpod.volumes.listed",
        "RunPod volumes listed.",
        source=source,
        count=len(items),
    )
    sanitized_items = redact_value(items)
    return RunPodResourceListResponse(items=sanitized_items if isinstance(sanitized_items, list) else [])


def _log_provider_event(
    event_id: str,
    message: str,
    *,
    source: str | None = None,
    count: int | None = None,
) -> None:
    fields: dict[str, Any] = {}
    if source is not None:
        fields["source"] = source
    if count is not None:
        fields["count"] = count
    logger.info(message, extra={"event_id": event_id, "event_fields": fields})


@training_api.post("/jobs", response_model=TrainingJobResponse, status_code=201)
def create_training_job(payload: CreateTrainingJobRequest, request: Request) -> TrainingJobResponse:
    manager = request.app.state.training_jobs
    try:
        if payload.execution_target.kind.value == "runpod_pod" and payload.execution_target.api_key is None:
            api_key, _ = _runpod_api_key(request)
            payload.execution_target.api_key = api_key
        return manager.create_job(payload)
    except KeyError as exc:
        missing_id = str(exc).strip("'")
        raise HTTPException(status_code=404, detail=f"Unknown asset id: {missing_id}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@training_api.get("/jobs", response_model=TrainingJobsListResponse)
def list_training_jobs(request: Request) -> TrainingJobsListResponse:
    manager = request.app.state.training_jobs
    return TrainingJobsListResponse(jobs=manager.list_jobs())


@training_api.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def get_training_job(job_id: str, request: Request) -> TrainingJobResponse:
    manager = request.app.state.training_jobs
    try:
        return manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown training job id: {job_id}") from exc


@training_api.delete("/jobs/{job_id}", status_code=204)
def delete_training_job(job_id: str, request: Request) -> Response:
    manager = request.app.state.training_jobs
    try:
        manager.delete_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown training job id: {job_id}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=204)


@training_api.get("/jobs/{job_id}/metrics", response_model=TrainingMetricsResponse)
def get_training_metrics(job_id: str, request: Request, limit: int | None = None) -> TrainingMetricsResponse:
    manager = request.app.state.training_jobs
    try:
        return manager.get_metrics(job_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown training job id: {job_id}") from exc


@training_api.get("/jobs/{job_id}/samples", response_model=TrainingSamplesResponse)
def get_training_samples(job_id: str, request: Request, limit: int = 50) -> TrainingSamplesResponse:
    manager = request.app.state.training_jobs
    try:
        return manager.get_samples(job_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown training job id: {job_id}") from exc


@training_api.get("/jobs/{job_id}/logs", response_model=TrainingLogsResponse)
def get_training_logs(job_id: str, request: Request, lines: int | None = None) -> TrainingLogsResponse:
    manager = request.app.state.training_jobs
    try:
        return manager.get_logs(job_id, lines=lines)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown training job id: {job_id}") from exc


@training_api.get("/jobs/{job_id}/checkpoints", response_model=TrainingCheckpointsResponse)
def get_training_checkpoints(job_id: str, request: Request) -> TrainingCheckpointsResponse:
    manager = request.app.state.training_jobs
    try:
        return manager.get_checkpoints(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown training job id: {job_id}") from exc


def _prepare_training_generation(
    job_id: str,
    payload: TrainingGenerateRequest,
    request: Request,
):
    manager = request.app.state.training_jobs
    tokenizer_manager = request.app.state.tokenizer_jobs

    try:
        job = manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown training job id: {job_id}") from exc

    if job.status != TrainingJobStatus.completed:
        raise HTTPException(status_code=409, detail="Only completed training jobs can be used for inference.")

    try:
        checkpoints = manager.get_checkpoints(job_id).checkpoints
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown training job id: {job_id}") from exc

    if payload.checkpoint_step is None:
        checkpoint = max(checkpoints, key=lambda item: item.step, default=None)
    else:
        checkpoint = next(
            (item for item in checkpoints if item.step == payload.checkpoint_step),
            None,
        )
    if checkpoint is None:
        if payload.checkpoint_step is None:
            raise HTTPException(status_code=409, detail="Training job has no saved checkpoints.")
        raise HTTPException(
            status_code=404,
            detail=f"Training job does not have a checkpoint at step {payload.checkpoint_step}.",
        )

    checkpoint_path = _checkpoint_model_path(checkpoint.directory, checkpoint.files)
    if checkpoint_path is None:
        raise HTTPException(status_code=409, detail=f"Checkpoint step {checkpoint.step} does not contain a model weights file.")
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise HTTPException(status_code=404, detail=f"Checkpoint weights missing: {checkpoint_path}")

    try:
        tokenizer_path = tokenizer_manager.get_artifact_path(job.tokenizer_job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown tokenizer job id: {job.tokenizer_job_id}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        prompt_token_ids = tokenizer.encode(payload.prompt).ids
        if not prompt_token_ids:
            raise HTTPException(status_code=422, detail="Prompt did not produce any tokens.")
        model = ConfigurableGPT(LLMConfig.model_validate(job.model_payload))
        device = _default_inference_device()
        model.to(device)
        model.load_state_dict(_torch_load_state_dict(checkpoint_path, device))
        model.eval()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(exc).__name__}: {exc}") from exc

    return job, checkpoint, checkpoint_path, tokenizer, prompt_token_ids, model


@training_api.post("/jobs/{job_id}/generate", response_model=TrainingGenerateResponse)
def generate_from_training_job(
    job_id: str,
    payload: TrainingGenerateRequest,
    request: Request,
) -> TrainingGenerateResponse:
    job, checkpoint, checkpoint_path, tokenizer, prompt_token_ids, model = _prepare_training_generation(
        job_id,
        payload,
        request,
    )
    try:
        generated_token_ids = list(
            model.generate(
                tokens=prompt_token_ids,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                top_k=payload.top_k,
                seed=payload.seed,
                repetition_penalty=payload.repetition_penalty,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(exc).__name__}: {exc}") from exc

    full_token_ids = prompt_token_ids + generated_token_ids
    completion = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    text = tokenizer.decode(full_token_ids, skip_special_tokens=False)
    return TrainingGenerateResponse(
        job_id=job_id,
        checkpoint_step=checkpoint.step,
        checkpoint_path=str(checkpoint_path),
        tokenizer_job_id=job.tokenizer_job_id,
        prompt=payload.prompt,
        completion=completion,
        text=text,
        prompt_token_count=len(prompt_token_ids),
        generated_token_count=len(generated_token_ids),
        generated_token_ids=generated_token_ids,
    )


@training_api.post("/jobs/{job_id}/generate/stream")
def stream_generate_from_training_job(
    job_id: str,
    payload: TrainingGenerateRequest,
    request: Request,
) -> StreamingResponse:
    job, checkpoint, checkpoint_path, tokenizer, prompt_token_ids, model = _prepare_training_generation(
        job_id,
        payload,
        request,
    )

    return StreamingResponse(
        _stream_generation_events(
            request=request,
            job_id=job_id,
            checkpoint_step=checkpoint.step,
            checkpoint_path=checkpoint_path,
            tokenizer_job_id=job.tokenizer_job_id,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            model=model,
            payload=payload,
        ),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )


async def _stream_generation_events(
    *,
    request: Request,
    job_id: str,
    checkpoint_step: int,
    checkpoint_path: Path,
    tokenizer_job_id: str,
    tokenizer: Tokenizer,
    prompt_token_ids: list[int],
    model: ConfigurableGPT,
    payload: TrainingGenerateRequest,
) -> AsyncIterator[str]:
    generated_token_ids: list[int] = []
    generation: Iterator[int] | None = None
    terminal_event = False
    cancellation_logged = False
    _log_inference_stream_event(
        "training.inference.stream.started",
        "Streamed inference started.",
        job_id=job_id,
        checkpoint_step=checkpoint_step,
    )
    try:
        yield _encode_stream_event(
            {
                "type": "start",
                "job_id": job_id,
                "checkpoint_step": checkpoint_step,
                "checkpoint_path": str(checkpoint_path),
                "tokenizer_job_id": tokenizer_job_id,
                "prompt": payload.prompt,
                "prompt_token_count": len(prompt_token_ids),
            }
        )
        generation = iter(
            model.generate(
                tokens=prompt_token_ids,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                top_k=payload.top_k,
                seed=payload.seed,
                repetition_penalty=payload.repetition_penalty,
            )
        )
        while True:
            if await request.is_disconnected():
                cancellation_logged = True
                _log_inference_stream_event(
                    "training.inference.stream.cancelled",
                    "Streamed inference stopped after the caller disconnected.",
                    job_id=job_id,
                    checkpoint_step=checkpoint_step,
                    generated_token_count=len(generated_token_ids),
                )
                return
            token_id = await anyio.to_thread.run_sync(_next_generation_item, generation)
            if token_id is _GENERATION_DONE:
                break
            generated_token_ids.append(token_id)
            yield _encode_stream_event(
                {
                    "type": "token",
                    "index": len(generated_token_ids),
                    "token_id": token_id,
                    "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                }
            )
        full_token_ids = prompt_token_ids + generated_token_ids
        terminal_event = True
        _log_inference_stream_event(
            "training.inference.stream.completed",
            "Streamed inference completed.",
            job_id=job_id,
            checkpoint_step=checkpoint_step,
            generated_token_count=len(generated_token_ids),
        )
        yield _encode_stream_event(
            {
                "type": "done",
                "completion": tokenizer.decode(generated_token_ids, skip_special_tokens=False),
                "text": tokenizer.decode(full_token_ids, skip_special_tokens=False),
                "generated_token_count": len(generated_token_ids),
                "generated_token_ids": generated_token_ids,
            }
        )
    except Exception as exc:
        terminal_event = True
        _log_inference_stream_event(
            "training.inference.stream.failed",
            "Streamed inference failed.",
            job_id=job_id,
            checkpoint_step=checkpoint_step,
            generated_token_count=len(generated_token_ids),
            error_type=type(exc).__name__,
        )
        yield _encode_stream_event(
            {
                "type": "error",
                "detail": f"Inference failed: {type(exc).__name__}: {exc}",
            }
        )
    finally:
        close = getattr(generation, "close", None) if generation is not None else None
        if callable(close):
            close()
        if not terminal_event and not cancellation_logged:
            _log_inference_stream_event(
                "training.inference.stream.cancelled",
                "Streamed inference response was cancelled.",
                job_id=job_id,
                checkpoint_step=checkpoint_step,
                generated_token_count=len(generated_token_ids),
            )


def _next_generation_item(generation: Iterator[int]) -> int | object:
    return next(generation, _GENERATION_DONE)


def _encode_stream_event(event_payload: dict[str, object]) -> str:
    return json.dumps(event_payload, ensure_ascii=False) + "\n"


def _log_inference_stream_event(
    event_id: str,
    message: str,
    *,
    job_id: str,
    checkpoint_step: int,
    generated_token_count: int | None = None,
    error_type: str | None = None,
) -> None:
    fields: dict[str, object] = {
        "job_id": job_id,
        "checkpoint_step": checkpoint_step,
    }
    if generated_token_count is not None:
        fields["generated_token_count"] = generated_token_count
    if error_type is not None:
        fields["error_type"] = error_type
    logger.info(message, extra={"event_id": event_id, "event_fields": fields})


@training_api.post("/jobs/{job_id}/stop", response_model=TrainingJobResponse)
def stop_training_job(job_id: str, request: Request) -> TrainingJobResponse:
    manager = request.app.state.training_jobs
    try:
        return manager.stop_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown training job id: {job_id}") from exc


@training_api.post("/jobs/{job_id}/remote/resync", response_model=TrainingJobResponse)
def resync_remote_training_job(job_id: str, request: Request) -> TrainingJobResponse:
    manager = request.app.state.training_jobs
    try:
        return manager.resync_remote_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown training job id: {job_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@training_api.post("/jobs/{job_id}/remote/cleanup", response_model=TrainingJobResponse)
def cleanup_remote_training_job(job_id: str, request: Request) -> TrainingJobResponse:
    manager = request.app.state.training_jobs
    try:
        return manager.cleanup_remote_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown training job id: {job_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@training_api.post("/jobs/{job_id}/remote/reattach", response_model=TrainingJobResponse)
def reattach_remote_training_job(job_id: str, request: Request) -> TrainingJobResponse:
    manager = request.app.state.training_jobs
    try:
        return manager.reattach_remote_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown training job id: {job_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@training_api.get("/jobs/{job_id}/artifact")
def download_training_job_artifact(job_id: str, request: Request) -> FileResponse:
    manager = request.app.state.training_jobs
    try:
        path = manager.build_artifact_bundle(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown training job id: {job_id}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InsufficientStorageError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    return FileResponse(path, filename=path.name, media_type="application/zip")


def _checkpoint_model_path(directory: str, files: list[str]) -> Path | None:
    checkpoint_dir = Path(directory)
    for file_name in files:
        path = Path(file_name)
        if path.name.startswith("model-") and path.suffix == ".pt":
            return path if path.is_absolute() else checkpoint_dir / path
    return None


def _torch_load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        loaded = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location=device)
    if not isinstance(loaded, dict):
        raise ValueError("Checkpoint weights payload must be a state dict.")
    return loaded


def _default_inference_device() -> torch.device:
    return torch.device(resolve_inference_device_type())
