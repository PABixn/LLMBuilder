from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..schemas import write_json
from ..dataset_credentials import split_dataset_hf_tokens, strip_hf_tokens
from .executors import TrainingJobBundle
from .preflight import ResolvedPreflightContext
from .runtime_files import directory_size
from .schemas import CreateTrainingJobRequest, TrainingJobState, TrainingJobStatus
from .store import StoredTrainingJob

FILENAME_SANITIZER = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass(slots=True)
class PreparedTrainingJob:
    stored: StoredTrainingJob
    bundle: TrainingJobBundle
    job_dir: Path
    preflight_payload: dict[str, Any]


def sanitized_file_stem(value: str) -> str:
    stem = FILENAME_SANITIZER.sub("-", value).strip("-")
    return stem if stem else "training-run"


def build_artifact_archive(*, artifact_dir: Path, exports_root: Path, name: str, job_id: str) -> Path:
    exports_root.mkdir(parents=True, exist_ok=True)
    base_name = sanitized_file_stem(f"{name}-{job_id[:8]}")
    archive_base = exports_root / base_name
    return Path(shutil.make_archive(str(archive_base), "zip", root_dir=artifact_dir))


def prepare_training_job(
    *,
    request: CreateTrainingJobRequest,
    context: ResolvedPreflightContext,
    preflight_payload: dict[str, Any],
    tokenizer_source_path: Path,
    training_jobs_root: Path,
    runpod_default_gpu_count: int,
    created_at: datetime,
    job_id: str | None = None,
) -> PreparedTrainingJob:
    resolved_job_id = job_id or uuid4().hex
    job_dir = training_jobs_root / resolved_job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    sanitized_dataloader_config, _ = split_dataset_hf_tokens(context.normalized_dataloader_config)
    _, dataset_hf_tokens = split_dataset_hf_tokens(
        request.dataloader_config,
        fallback_token=request.hf_token,
    )
    sanitized_context = replace(
        context,
        normalized_dataloader_config=sanitized_dataloader_config,
    )
    sanitized_preflight_payload = strip_hf_tokens(preflight_payload)
    bundle = build_training_job_bundle(
        job_id=resolved_job_id,
        job_dir=job_dir,
        execution_target_payload=request.execution_target.model_dump(mode="json"),
        dataset_hf_tokens=dataset_hf_tokens,
    )
    write_training_job_inputs(
        bundle=bundle,
        context=sanitized_context,
        preflight_payload=sanitized_preflight_payload,
        tokenizer_source_path=tokenizer_source_path,
    )
    stored = build_stored_training_job(
        request=request,
        context=sanitized_context,
        job_id=resolved_job_id,
        job_dir=job_dir,
        bundle=bundle,
        runpod_default_gpu_count=runpod_default_gpu_count,
        created_at=created_at,
    )
    return PreparedTrainingJob(
        stored=stored,
        bundle=bundle,
        job_dir=job_dir,
        preflight_payload=sanitized_preflight_payload,
    )


def build_training_job_bundle(
    *,
    job_id: str,
    job_dir: Path,
    execution_target_payload: dict[str, Any],
    dataset_hf_tokens: list[str | None] | None = None,
) -> TrainingJobBundle:
    return TrainingJobBundle(
        job_id=job_id,
        job_dir=job_dir,
        model_config_path=job_dir / "model_config.json",
        tokenizer_path=job_dir / "tokenizer_artifact.json",
        training_config_path=job_dir / "training_config.json",
        dataloader_config_path=job_dir / "dataloader_config.json",
        resolved_preflight_path=job_dir / "resolved_preflight.json",
        stdout_path=job_dir / "stdout.log",
        stderr_path=job_dir / "stderr.log",
        stats_path=job_dir / "stats.jsonl",
        samples_path=job_dir / "samples.jsonl",
        manifest={
            "format": "llm-studio-training-bundle-v1",
            "job_id": job_id,
            "execution_target": execution_target_payload,
            "dataset_hf_tokens": list(dataset_hf_tokens or []),
        },
    )


def write_training_job_inputs(
    *,
    bundle: TrainingJobBundle,
    context: ResolvedPreflightContext,
    preflight_payload: dict[str, Any],
    tokenizer_source_path: Path,
) -> None:
    write_json(bundle.model_config_path, context.model_config)
    write_json(bundle.training_config_path, context.normalized_training_config)
    write_json(bundle.dataloader_config_path, context.normalized_dataloader_config)
    write_json(bundle.resolved_preflight_path, preflight_payload)
    shutil.copy2(tokenizer_source_path, bundle.tokenizer_path)


def build_stored_training_job(
    *,
    request: CreateTrainingJobRequest,
    context: ResolvedPreflightContext,
    job_id: str,
    job_dir: Path,
    bundle: TrainingJobBundle,
    runpod_default_gpu_count: int,
    created_at: datetime,
) -> StoredTrainingJob:
    execution_target = request.execution_target
    cleanup_policy = execution_target.cleanup_policy.model_dump(mode="json")
    name = request.name or f"{context.model_project.name} x {context.tokenizer.name}"
    return StoredTrainingJob(
        id=job_id,
        name=name,
        status=TrainingJobStatus.pending,
        state=TrainingJobState.queued,
        stage="Queued",
        progress=0.0,
        created_at=created_at,
        started_at=None,
        finished_at=None,
        project_id=request.project_id,
        project_name=context.model_project.name,
        tokenizer_job_id=request.tokenizer_job_id,
        tokenizer_name=context.tokenizer.name,
        model_config=context.model_config,
        training_config=context.normalized_training_config,
        dataloader_config=context.normalized_dataloader_config,
        resolved_runtime=context.derived_runtime.model_dump(mode="json")
        if context.derived_runtime is not None
        else None,
        memory_estimate=context.memory_estimate,
        artifact_dir=str(job_dir),
        artifact_bundle_file=None,
        stats_path=str(bundle.stats_path),
        samples_path=str(bundle.samples_path),
        stdout_path=str(bundle.stdout_path),
        stderr_path=str(bundle.stderr_path),
        last_step=0,
        max_steps=int(context.normalized_training_config.get("max_steps", 0)),
        latest_loss=None,
        latest_grad_norm=None,
        latest_lr=None,
        latest_tokens_per_sec=None,
        checkpoint_count=0,
        sample_count=0,
        error=None,
        process_id=None,
        output_size_bytes=directory_size(job_dir),
        executor_kind=execution_target.kind.value,
        executor_status="queued",
        runpod_data_center_id=execution_target.data_center_id,
        runpod_gpu_type_id=execution_target.gpu_type_id,
        runpod_gpu_count=execution_target.gpu_count or runpod_default_gpu_count,
        runpod_cloud_type=execution_target.cloud_type.value if execution_target.cloud_type is not None else None,
        runpod_interruptible=execution_target.interruptible,
        runpod_cleanup_policy=cleanup_policy,
    )
