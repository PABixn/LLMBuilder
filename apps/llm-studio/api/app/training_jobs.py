from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from pydantic import ValidationError
from tokenizers import Tokenizer

from .config import get_settings, training_exports_dir, training_jobs_dir
from .schemas import load_json, write_json
from .tokenizer_storage import StudioStore as TokenizerStudioStore
from .training_models import (
    TrainingBatchLrRecommendation,
    CreateTrainingJobRequest,
    DerivedRuntimeSummary,
    TrainingAssetRef,
    TrainingCheckpointEntry,
    TrainingCheckpointsResponse,
    TrainingCompatibilitySummary,
    TrainingFixSuggestion,
    TrainingIssue,
    TrainingJobResponse,
    TrainingJobState,
    TrainingJobStatus,
    TrainingLogsResponse,
    TrainingMetricPoint,
    TrainingMetricsResponse,
    TrainingPreflightRequest,
    TrainingPreflightResponse,
    TrainingSampleEntry,
    TrainingSampleText,
    TrainingSamplesResponse,
)
from .training_recommendations import build_batch_and_lr_recommendation
from .training_storage import StoredTrainingJob, TrainingStudioStore
from .training_executors import LocalSubprocessExecutor, RunPodPodExecutor, TrainingExecutor, TrainingJobBundle

IMPORT_ROOT = Path(__file__).resolve().parents[4]
if str(IMPORT_ROOT) not in sys.path:
    sys.path.append(str(IMPORT_ROOT))

from model.loader import LLMConfig
from model.model import ConfigurableGPT
from training.dataloader_config import TrainingDataloaderConfig
from training.memory_estimator import MemoryEstimator
from training.training_config import TrainingConfig, derive_batch_runtime_plan

_JOB_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{6,64}$")
_PROJECT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{6,64}$")
_FILENAME_SANITIZER = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass(slots=True)
class ResolvedPreflightContext:
    valid: bool
    model_project: TrainingAssetRef
    tokenizer: TrainingAssetRef
    model_config: dict[str, Any]
    normalized_training_config: dict[str, Any]
    normalized_dataloader_config: dict[str, Any]
    warnings: list[TrainingIssue]
    errors: list[TrainingIssue]
    recommended_fixes: list[TrainingFixSuggestion]
    compatibility: TrainingCompatibilitySummary | None
    derived_runtime: DerivedRuntimeSummary | None
    memory_estimate: dict[str, Any] | None
    batch_and_lr_recommendation: TrainingBatchLrRecommendation | None


class TrainingRunManager:
    def __init__(
        self,
        *,
        store: TrainingStudioStore,
        tokenizer_store: TokenizerStudioStore,
    ) -> None:
        self._store = store
        self._tokenizer_store = tokenizer_store
        self._local_executor = LocalSubprocessExecutor()
        self._runpod_executor = RunPodPodExecutor()
        self._executors: dict[str, TrainingExecutor] = {
            self._local_executor.kind: self._local_executor,
            self._runpod_executor.kind: self._runpod_executor,
        }

    def build_preflight(self, request: TrainingPreflightRequest) -> TrainingPreflightResponse:
        context = self._resolve_preflight_context(request)
        return self._preflight_response_from_context(context)

    def _preflight_response_from_context(
        self,
        context: ResolvedPreflightContext,
    ) -> TrainingPreflightResponse:
        return TrainingPreflightResponse(
            valid=context.valid,
            model_project=context.model_project,
            tokenizer=context.tokenizer,
            normalized_training_config=context.normalized_training_config,
            normalized_dataloader_config=context.normalized_dataloader_config,
            warnings=context.warnings,
            errors=context.errors,
            recommended_fixes=context.recommended_fixes,
            compatibility=context.compatibility,
            derived_runtime=context.derived_runtime,
            memory_estimate=context.memory_estimate,
            batch_and_lr_recommendation=context.batch_and_lr_recommendation,
        )

    def create_job(self, request: CreateTrainingJobRequest) -> TrainingJobResponse:
        context = self._resolve_preflight_context(request)
        if not context.valid:
            raise ValueError("Training preflight failed. Resolve the reported issues before launching.")
        preflight_payload = self._preflight_response_from_context(context).model_dump(mode="json")

        settings = get_settings()
        now = utc_now()
        job_id = uuid4().hex
        job_dir = settings.training_jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)

        model_config_path = job_dir / "model_config.json"
        tokenizer_path = job_dir / "tokenizer_artifact.json"
        training_config_path = job_dir / "training_config.json"
        dataloader_config_path = job_dir / "dataloader_config.json"
        resolved_preflight_path = job_dir / "resolved_preflight.json"
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        stats_path = job_dir / "stats.jsonl"
        samples_path = job_dir / "samples.jsonl"

        write_json(model_config_path, context.model_config)
        write_json(training_config_path, context.normalized_training_config)
        write_json(dataloader_config_path, context.normalized_dataloader_config)
        write_json(resolved_preflight_path, preflight_payload)

        tokenizer_source_path = self._require_tokenizer_artifact_path(request.tokenizer_job_id)
        shutil.copy2(tokenizer_source_path, tokenizer_path)

        name = request.name or f"{context.model_project.name} x {context.tokenizer.name}"
        execution_target = request.execution_target
        cleanup_policy = execution_target.cleanup_policy.model_dump(mode="json")
        stored = StoredTrainingJob(
            id=job_id,
            name=name,
            status=TrainingJobStatus.pending,
            state=TrainingJobState.queued,
            stage="Queued",
            progress=0.0,
            created_at=now,
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
            stats_path=str(stats_path),
            samples_path=str(samples_path),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
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
            runpod_gpu_count=execution_target.gpu_count or get_settings().runpod_default_gpu_count,
            runpod_cloud_type=execution_target.cloud_type.value if execution_target.cloud_type is not None else None,
            runpod_interruptible=execution_target.interruptible,
            runpod_cleanup_policy=cleanup_policy,
        )
        self._store.create_job(stored)

        bundle = TrainingJobBundle(
            job_id=job_id,
            job_dir=job_dir,
            model_config_path=model_config_path,
            tokenizer_path=tokenizer_path,
            training_config_path=training_config_path,
            dataloader_config_path=dataloader_config_path,
            resolved_preflight_path=resolved_preflight_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stats_path=stats_path,
            samples_path=samples_path,
            manifest={
                "format": "llm-studio-training-bundle-v1",
                "job_id": job_id,
                "execution_target": execution_target.model_dump(mode="json"),
            },
        )
        executor = self._executor_for_kind(request.execution_target.kind.value)

        if executor.kind == self._runpod_executor.kind:
            started_at = utc_now()
            self._store.update_job(
                job_id,
                status=TrainingJobStatus.running,
                state=TrainingJobState.preflight,
                stage="Provisioning RunPod pod",
                started_at=started_at,
                executor_status="provisioning",
            )
            thread = threading.Thread(
                target=self._submit_remote_job,
                args=(job_id, stored, bundle),
                name=f"llm-studio-runpod-{job_id[:12]}",
                daemon=True,
            )
            thread.start()
            return self.get_job(job_id)

        try:
            handle = executor.submit(
                stored,
                bundle,
            )
        except Exception:
            self._store.delete_job(job_id)
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

        self._store.update_job(
            job_id,
            status=TrainingJobStatus.running,
            state=TrainingJobState.preflight,
            stage="Launching training process" if handle.executor_kind == "local" else "Provisioning RunPod pod",
            started_at=handle.started_at or utc_now(),
            process_id=handle.process_id,
            **handle.updates,
        )
        return self.get_job(job_id)

    def _submit_remote_job(self, job_id: str, stored: StoredTrainingJob, bundle: TrainingJobBundle) -> None:
        try:
            handle = self._runpod_executor.submit(stored, bundle)
        except Exception as exc:
            self._store.update_job(
                job_id,
                status=TrainingJobStatus.failed,
                state=TrainingJobState.failed,
                stage="RunPod launch failed",
                progress=1.0,
                finished_at=utc_now(),
                error=str(exc),
                executor_status="failed",
                remote_error=str(exc),
            )
            return

        self._store.update_job(
            job_id,
            status=TrainingJobStatus.running,
            state=TrainingJobState.preflight,
            stage="Training started on RunPod",
            started_at=handle.started_at or utc_now(),
            process_id=handle.process_id,
            **handle.updates,
        )

    def list_jobs(self) -> list[TrainingJobResponse]:
        jobs = self._store.list_jobs()
        return [self._to_response(self._refresh_job(job.id) or job) for job in jobs]

    def get_job(self, job_id: str) -> TrainingJobResponse:
        job = self._refresh_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return self._to_response(job)

    def delete_job(self, job_id: str) -> None:
        job = self._refresh_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status in {TrainingJobStatus.pending, TrainingJobStatus.running}:
            raise RuntimeError("Stop the training job before deleting it.")

        artifact_dir = Path(job.artifact_dir)
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=False)
        if job.artifact_bundle_file:
            bundle_path = training_exports_dir() / job.artifact_bundle_file
            if bundle_path.exists():
                bundle_path.unlink()
        deleted = self._store.delete_job(job_id)
        if deleted is None:
            raise KeyError(job_id)

    def stop_job(self, job_id: str) -> TrainingJobResponse:
        job = self._refresh_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status not in {TrainingJobStatus.pending, TrainingJobStatus.running}:
            return self._to_response(job)

        finished_at = utc_now()
        snapshot = self._executor_for_job(job).stop(job)
        runtime_state_path = Path(job.artifact_dir) / "runtime_state.json"
        runtime_state = load_optional_json(runtime_state_path) or {}
        if isinstance(runtime_state, dict):
            runtime_state.update(
                {
                    "status": TrainingJobStatus.cancelled.value,
                    "state": TrainingJobState.cancelled.value,
                    "stage": "Cancelled",
                    "progress": 1.0,
                    "finished_at": finished_at.isoformat(),
                    "error": "Training was cancelled by the user.",
                }
            )
            write_json(runtime_state_path, runtime_state)

        updated = self._store.update_job(
            job_id,
            status=snapshot.status or TrainingJobStatus.cancelled,
            state=snapshot.state or TrainingJobState.cancelled,
            stage=snapshot.stage or "Cancelled",
            finished_at=snapshot.finished_at or finished_at,
            progress=1.0 if snapshot.progress is None else snapshot.progress,
            error=snapshot.error or "Training was cancelled by the user.",
            **snapshot.updates,
        )
        if updated is None:
            raise KeyError(job_id)
        return self._to_response(updated)

    def get_metrics(self, job_id: str, *, limit: int | None = None) -> TrainingMetricsResponse:
        job = self.get_job(job_id)
        payloads = read_jsonl(Path(job.stats_path))
        if limit is not None and limit > 0:
            payloads = payloads[-limit:]
        metrics = [TrainingMetricPoint.model_validate(item) for item in payloads]
        return TrainingMetricsResponse(job_id=job_id, metrics=metrics)

    def get_samples(self, job_id: str, *, limit: int = 50) -> TrainingSamplesResponse:
        job = self.get_job(job_id)
        payloads = read_jsonl(Path(job.samples_path))[-max(limit, 1) :]
        samples: list[TrainingSampleEntry] = []
        for item in payloads:
            entries = [
                TrainingSampleText.model_validate(sample)
                for sample in item.get("samples", [])
                if isinstance(sample, dict)
            ]
            samples.append(
                TrainingSampleEntry(
                    step=int(item.get("step", 0)),
                    samples=entries,
                )
            )
        return TrainingSamplesResponse(job_id=job_id, samples=samples)

    def get_logs(self, job_id: str, *, lines: int | None = None) -> TrainingLogsResponse:
        job = self.get_job(job_id)
        return TrainingLogsResponse(
            job_id=job_id,
            stdout_lines=tail_lines(Path(job.stdout_path), lines),
            stderr_lines=tail_lines(Path(job.stderr_path), lines),
        )

    def get_data_preview(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        preview_path = Path(job.artifact_dir) / "training_data_preview.json"
        payload = load_optional_json(preview_path)
        if payload is None:
            raise FileNotFoundError(preview_path)
        return payload

    def get_checkpoints(self, job_id: str) -> TrainingCheckpointsResponse:
        job = self.get_job(job_id)
        checkpoints_root = Path(job.artifact_dir) / "checkpoints"
        checkpoints: list[TrainingCheckpointEntry] = []
        if checkpoints_root.exists():
            for candidate in sorted(
                (path for path in checkpoints_root.iterdir() if path.is_dir()),
                key=lambda path: int(path.name) if path.name.isdigit() else -1,
                reverse=True,
            ):
                files = sorted(str(path.name) for path in candidate.iterdir() if path.is_file())
                checkpoints.append(
                    TrainingCheckpointEntry(
                        step=int(candidate.name) if candidate.name.isdigit() else 0,
                        directory=str(candidate),
                        created_at=datetime.fromtimestamp(candidate.stat().st_mtime, tz=timezone.utc),
                        size_bytes=directory_size(candidate),
                        files=files,
                    )
                )
        return TrainingCheckpointsResponse(job_id=job_id, checkpoints=checkpoints)

    def build_artifact_bundle(self, job_id: str) -> Path:
        job = self.get_job(job_id)
        artifact_dir = Path(job.artifact_dir)
        exports_root = training_exports_dir()
        exports_root.mkdir(parents=True, exist_ok=True)
        base_name = sanitized_file_stem(f"{job.name}-{job.id[:8]}")
        archive_base = exports_root / base_name
        archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=artifact_dir)
        bundle_name = Path(archive_path).name
        self._store.update_job(job_id, artifact_bundle_file=bundle_name)
        return Path(archive_path)

    def resync_remote_job(self, job_id: str) -> TrainingJobResponse:
        job = self._refresh_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.executor_kind != "runpod_pod":
            raise ValueError("Remote resync is only available for RunPod jobs.")
        return self._to_response(job)

    def cleanup_remote_job(self, job_id: str) -> TrainingJobResponse:
        job = self._refresh_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.executor_kind != "runpod_pod":
            raise ValueError("Remote cleanup is only available for RunPod jobs.")
        policy_payload = job.runpod_cleanup_policy or {}
        from .training_executors import CleanupPolicy

        self._executor_for_job(job).cleanup(
            job,
            CleanupPolicy(
                pod=str(policy_payload.get("pod") or "delete_after_sync"),
                network_volume=str(policy_payload.get("network_volume") or "keep"),
            ),
        )
        updated = self._store.update_job(job_id, executor_status="cleaned_up") or job
        return self._to_response(updated)

    def reattach_remote_job(self, job_id: str) -> TrainingJobResponse:
        job = self._store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.executor_kind != "runpod_pod":
            raise ValueError("Remote reattach is only available for RunPod jobs.")
        updated = self._store.update_job(
            job_id,
            remote_error=(
                "Automatic reattach cannot recover the raw pod-agent token because only its hash is stored. "
                "Stop the Pod or launch a new run."
            ),
        ) or job
        return self._to_response(updated)

    def shutdown(self) -> None:
        self._local_executor.shutdown()

    def _refresh_job(self, job_id: str) -> StoredTrainingJob | None:
        validate_identifier(job_id, _JOB_ID_RE)
        job = self._store.get_job(job_id)
        if job is None:
            return None

        updates: dict[str, Any] = {}
        runtime_state = load_optional_json(Path(job.artifact_dir) / "runtime_state.json")
        metadata = load_optional_json(Path(job.artifact_dir) / "metadata.json")
        if runtime_state is not None:
            runtime_updates = self._state_updates_from_runtime(runtime_state)
            if job.status not in {TrainingJobStatus.pending, TrainingJobStatus.running}:
                for key in ("status", "state", "stage", "progress", "started_at"):
                    runtime_updates.pop(key, None)
                if job.finished_at is not None:
                    runtime_updates.pop("finished_at", None)
                if job.error is not None:
                    runtime_updates.pop("error", None)
            updates.update(runtime_updates)
        if metadata is not None:
            if isinstance(metadata.get("error"), str):
                updates["error"] = metadata["error"]

        snapshot = self._executor_for_job(job).refresh(job)
        if snapshot.status is not None:
            status = runtime_state.get("status") if isinstance(runtime_state, dict) else None
            if status == TrainingJobStatus.completed.value:
                updates.setdefault("status", TrainingJobStatus.completed)
                updates.setdefault("state", TrainingJobState.completed)
                updates.setdefault("stage", "Completed")
                updates.setdefault("progress", 1.0)
            else:
                updates.setdefault("status", snapshot.status)
                if snapshot.state is not None:
                    updates.setdefault("state", snapshot.state)
                if snapshot.stage is not None:
                    updates.setdefault("stage", snapshot.stage)
                if snapshot.progress is not None:
                    updates.setdefault("progress", snapshot.progress)
                if snapshot.error is not None:
                    updates.setdefault("error", snapshot.error)
            if snapshot.finished_at is not None:
                updates.setdefault("finished_at", snapshot.finished_at)
        updates.update(snapshot.updates)

        updates["output_size_bytes"] = directory_size(Path(job.artifact_dir))

        if updates:
            job = self._store.update_job(job_id, **updates) or job
        return job

    def _state_updates_from_runtime(self, payload: dict[str, Any]) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        if "status" in payload and isinstance(payload["status"], str):
            updates["status"] = TrainingJobStatus(payload["status"])
        if "state" in payload and isinstance(payload["state"], str):
            updates["state"] = TrainingJobState(payload["state"])
        if "stage" in payload and isinstance(payload["stage"], str):
            updates["stage"] = payload["stage"]
        if "progress" in payload:
            updates["progress"] = float(payload["progress"])
        if "started_at" in payload and isinstance(payload["started_at"], str):
            updates["started_at"] = parse_datetime(payload["started_at"])
        if "finished_at" in payload and isinstance(payload["finished_at"], str):
            updates["finished_at"] = parse_datetime(payload["finished_at"])
        if "last_step" in payload:
            updates["last_step"] = int(payload["last_step"])
        if "max_steps" in payload:
            updates["max_steps"] = int(payload["max_steps"])
        if "latest_loss" in payload:
            updates["latest_loss"] = None if payload["latest_loss"] is None else float(payload["latest_loss"])
        if "latest_grad_norm" in payload:
            updates["latest_grad_norm"] = (
                None if payload["latest_grad_norm"] is None else float(payload["latest_grad_norm"])
            )
        if "latest_lr" in payload:
            updates["latest_lr"] = None if payload["latest_lr"] is None else float(payload["latest_lr"])
        if "latest_tokens_per_sec" in payload:
            updates["latest_tokens_per_sec"] = (
                None if payload["latest_tokens_per_sec"] is None else float(payload["latest_tokens_per_sec"])
            )
        if "checkpoint_count" in payload:
            updates["checkpoint_count"] = int(payload["checkpoint_count"])
        if "sample_count" in payload:
            updates["sample_count"] = int(payload["sample_count"])
        if "resolved_runtime" in payload and isinstance(payload["resolved_runtime"], dict):
            updates["resolved_runtime"] = payload["resolved_runtime"]
        if "memory_estimate" in payload and isinstance(payload["memory_estimate"], dict):
            updates["memory_estimate"] = payload["memory_estimate"]
        if "error" in payload and isinstance(payload["error"], str):
            updates["error"] = payload["error"]
        return updates

    def _resolve_preflight_context(self, request: TrainingPreflightRequest) -> ResolvedPreflightContext:
        warnings: list[TrainingIssue] = []
        errors: list[TrainingIssue] = []
        fixes: list[TrainingFixSuggestion] = []

        project_ref, model_config = self._load_project_asset(request.project_id)
        tokenizer_ref, tokenizer_path = self._load_tokenizer_asset(request.tokenizer_job_id)

        try:
            parsed_training_config = TrainingConfig.model_validate(request.training_config)
            normalized_training_config = parsed_training_config.model_dump(mode="json")
        except ValidationError as exc:
            parsed_training_config = None
            normalized_training_config = request.training_config
            errors.extend(validation_issues("training_config_invalid", "$.training_config", exc))
        except Exception as exc:
            parsed_training_config = None
            normalized_training_config = request.training_config
            errors.append(issue("training_config_invalid", str(exc), "$.training_config"))

        try:
            parsed_dataloader_config = TrainingDataloaderConfig.model_validate(request.dataloader_config)
            normalized_dataloader_config = parsed_dataloader_config.model_dump(mode="json")
        except ValidationError as exc:
            parsed_dataloader_config = None
            normalized_dataloader_config = request.dataloader_config
            errors.extend(validation_issues("dataloader_config_invalid", "$.dataloader_config", exc))
        except Exception as exc:
            parsed_dataloader_config = None
            normalized_dataloader_config = request.dataloader_config
            errors.append(issue("dataloader_config_invalid", str(exc), "$.dataloader_config"))

        compatibility = None
        derived_runtime = None
        memory_estimate = None
        batch_and_lr_recommendation = None

        add_scheduler_step_fix(fixes, request.training_config)

        if parsed_training_config is not None and parsed_dataloader_config is not None:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer_vocab_size = tokenizer.get_vocab_size()
            missing_special_tokens = collect_missing_special_tokens(tokenizer, parsed_dataloader_config)
            compatibility = TrainingCompatibilitySummary(
                model_context_length=int(model_config["context_length"]),
                model_vocab_size=int(model_config["vocab_size"]),
                tokenizer_vocab_size=int(tokenizer_vocab_size),
                seq_len=int(parsed_training_config.seq_len),
                scheduler_total_steps=sum(item.steps for item in parsed_training_config.lr_scheduler.schedulers),
                max_steps=int(parsed_training_config.max_steps),
                missing_special_tokens=missing_special_tokens,
            )

            if int(model_config["vocab_size"]) != int(tokenizer_vocab_size):
                errors.append(
                    issue(
                        "vocab_size_mismatch",
                        f"Model vocab_size ({model_config['vocab_size']}) must match tokenizer vocab_size ({tokenizer_vocab_size}).",
                        "$.model_config.vocab_size",
                    )
                )

            if parsed_training_config.seq_len > int(model_config["context_length"]):
                errors.append(
                    issue(
                        "seq_len_exceeds_context_length",
                        f"seq_len ({parsed_training_config.seq_len}) exceeds model context_length ({model_config['context_length']}).",
                        "$.training_config.seq_len",
                    )
                )
                fixes.append(
                    TrainingFixSuggestion(
                        code="set_seq_len_to_context_length",
                        label="Set seq_len to model context_length",
                        description="Clamp sequence length to the selected model's context window.",
                        path="training_config.seq_len",
                        value=int(model_config["context_length"]),
                    )
                )

            for missing_token in missing_special_tokens:
                errors.append(
                    issue(
                        "missing_special_token",
                        f"Tokenizer is missing required special token {missing_token}.",
                        "$.dataloader_config",
                    )
                )

            data_file_errors = validate_local_data_files(parsed_dataloader_config)
            errors.extend(data_file_errors)

            sparse_checkpoint_threshold = cadence_for_fraction(parsed_training_config.max_steps, 0.2)
            if parsed_training_config.save_every > sparse_checkpoint_threshold:
                warnings.append(
                    issue(
                        "save_every_sparse",
                        f"save_every is larger than 20% of max_steps ({sparse_checkpoint_threshold}), so checkpoints will be sparse.",
                        "$.training_config.save_every",
                        severity="warning",
                    )
                )
                suggested_save_every = cadence_for_fraction(parsed_training_config.max_steps, 0.1)
                fixes.append(
                    TrainingFixSuggestion(
                        code="set_save_every_to_periodic_cadence",
                        label="Save every 10% of the run",
                        description=f"Set save_every to {suggested_save_every}, creating roughly 10 checkpoints across this run.",
                        path="training_config.save_every",
                        value=int(suggested_save_every),
                    )
                )
            if parsed_training_config.sample_every > parsed_training_config.max_steps:
                warnings.append(
                    issue(
                        "sample_every_exceeds_max_steps",
                        "sample_every is larger than max_steps, so no intermediate samples will be generated.",
                        "$.training_config.sample_every",
                        severity="warning",
                    )
                )
                suggested_sample_every = cadence_for_fraction(parsed_training_config.max_steps, 0.2)
                fixes.append(
                    TrainingFixSuggestion(
                        code="set_sample_every_to_run_cadence",
                        label="Sample during the run",
                        description=f"Set sample_every to {suggested_sample_every} so samples appear before training completes.",
                        path="training_config.sample_every",
                        value=int(suggested_sample_every),
                    )
                )

            if parsed_training_config.optimizer.lr >= 0.01:
                warnings.append(
                    issue(
                        "optimizer_lr_high",
                        "Learning rate is unusually high for transformer training and may destabilize the run.",
                        "$.training_config.optimizer.lr",
                        severity="warning",
                    )
                )
                fixes.append(
                    TrainingFixSuggestion(
                        code="set_optimizer_lr_to_starter_safe_value",
                        label="Use a safer learning rate",
                        description="Set optimizer.lr to 3e-4, a conservative starter value for this trainer.",
                        path="training_config.optimizer.lr",
                        value=0.0003,
                    )
                )

            if parsed_training_config.seq_len <= int(model_config["context_length"]):
                model = ConfigurableGPT(load_config_dict(model_config))
                optimizer = model.setup_optimizer(
                    lr=parsed_training_config.optimizer.lr,
                    weight_decay=parsed_training_config.optimizer.weight_decay,
                    betas=parsed_training_config.optimizer.betas,
                    eps=parsed_training_config.optimizer.eps,
                )
                device = default_training_device()
                memory_estimator = MemoryEstimator(
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    token_dtype=parsed_dataloader_config.token_dtype,
                )
                estimate = memory_estimator.estimate(
                    seq_len=parsed_training_config.seq_len,
                    batch_size=None,
                )
                recommendation_estimate = estimate
                batch_plan = None
                try:
                    batch_plan = derive_batch_runtime_plan(
                        total_batch_size=parsed_training_config.total_batch_size,
                        seq_len=parsed_training_config.seq_len,
                        max_memory_batch_size=estimate.max_batch_size,
                        world_size=1,
                        requested_micro_batch_size=parsed_training_config.micro_batch_size,
                    )
                except ValueError as exc:
                    runtime_error = str(exc)
                    errors.append(
                        issue(
                            "invalid_micro_batch_size",
                            runtime_error,
                            "$.training_config.total_batch_size",
                        )
                    )
                    fixes.extend(runtime_batch_fixes(runtime_error, parsed_training_config, estimate.max_batch_size))
                    memory_estimate = estimate.to_dict()
                else:
                    configured_estimate = memory_estimator.estimate(
                        seq_len=parsed_training_config.seq_len,
                        batch_size=batch_plan.micro_batch_size,
                    )
                    recommendation_estimate = configured_estimate
                    memory_estimate = configured_estimate.to_dict()
                    derived_runtime = DerivedRuntimeSummary(
                        device=str(device),
                        device_type=device.type,
                        micro_batch_size=batch_plan.micro_batch_size,
                        tokens_per_micro_step=batch_plan.tokens_per_micro_step,
                        tokens_per_world_step=batch_plan.tokens_per_world_step,
                        grad_accum_steps=batch_plan.grad_accum_steps,
                        max_batch_size_from_total=batch_plan.max_batch_size_from_total,
                        max_batch_size_from_memory=batch_plan.max_batch_size_from_memory,
                        max_allowed_batch_size=batch_plan.max_allowed_batch_size,
                        ddp=False,
                        ddp_rank=0,
                        ddp_world_size=1,
                    )
                tokenizer_job = self._tokenizer_store.get_job(request.tokenizer_job_id)
                batch_and_lr_recommendation = build_batch_and_lr_recommendation(
                    model_config=load_config_dict(model_config),
                    model=model,
                    training_config=parsed_training_config,
                    dataloader_config=parsed_dataloader_config,
                    tokenizer_stats=tokenizer_job.stats if tokenizer_job is not None else None,
                    memory_estimate=recommendation_estimate,
                    current_batch_plan=batch_plan,
                )

        return ResolvedPreflightContext(
            valid=not errors,
            model_project=project_ref,
            tokenizer=tokenizer_ref,
            model_config=model_config,
            normalized_training_config=normalized_training_config,
            normalized_dataloader_config=normalized_dataloader_config,
            warnings=warnings,
            errors=errors,
            recommended_fixes=fixes,
            compatibility=compatibility,
            derived_runtime=derived_runtime,
            memory_estimate=memory_estimate,
            batch_and_lr_recommendation=batch_and_lr_recommendation,
        )

    def _load_project_asset(self, project_id: str) -> tuple[TrainingAssetRef, dict[str, Any]]:
        validate_identifier(project_id, _PROJECT_ID_RE)
        project_dir = get_settings().projects_dir / project_id
        metadata_path = project_dir / "metadata.json"
        artifact_path = project_dir / "model_config.json"
        if not metadata_path.exists() or not artifact_path.exists():
            raise KeyError(project_id)
        metadata = load_json(metadata_path)
        model_config = load_json(artifact_path)
        model = load_config_dict(model_config)
        name = metadata.get("name") if isinstance(metadata.get("name"), str) and metadata.get("name") else f"Project {project_id[:8]}"
        return (
            TrainingAssetRef(
                id=project_id,
                name=name,
                artifact_path=str(artifact_path.resolve()),
                artifact_file=artifact_path.name,
                status="READY",
            ),
            model.model_dump(mode="json"),
        )

    def _load_tokenizer_asset(self, tokenizer_job_id: str) -> tuple[TrainingAssetRef, Path]:
        validate_identifier(tokenizer_job_id, _JOB_ID_RE)
        tokenizer_job = self._tokenizer_store.get_job(tokenizer_job_id)
        if tokenizer_job is None:
            raise KeyError(tokenizer_job_id)
        if tokenizer_job.status.value != "completed":
            raise ValueError("Tokenizer job must be completed before training can start.")
        artifact_path = self._require_tokenizer_artifact_path(tokenizer_job_id)
        return (
            TrainingAssetRef(
                id=tokenizer_job_id,
                name=tokenizer_display_name(tokenizer_job.tokenizer_config, tokenizer_job_id, tokenizer_job.artifact_file),
                artifact_path=str(artifact_path.resolve()),
                artifact_file=artifact_path.name,
                status=tokenizer_job.status.value,
            ),
            artifact_path,
        )

    def _require_tokenizer_artifact_path(self, tokenizer_job_id: str) -> Path:
        tokenizer_job = self._tokenizer_store.get_job(tokenizer_job_id)
        if tokenizer_job is None or tokenizer_job.artifact_path is None:
            raise KeyError(tokenizer_job_id)
        artifact_path = Path(tokenizer_job.artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Tokenizer artifact missing: {artifact_path}")
        return artifact_path

    def _executor_for_kind(self, kind: str) -> TrainingExecutor:
        executor = self._executors.get(kind)
        if executor is None:
            raise ValueError(f"Unsupported training execution target: {kind}")
        return executor

    def _executor_for_job(self, job: StoredTrainingJob) -> TrainingExecutor:
        return self._executor_for_kind(job.executor_kind or "local")

    def _to_response(self, job: StoredTrainingJob) -> TrainingJobResponse:
        runtime_state = load_optional_json(Path(job.artifact_dir) / "runtime_state.json")
        return TrainingJobResponse(
            id=job.id,
            name=job.name,
            status=job.status,
            state=job.state,
            stage=job.stage,
            progress=job.progress,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            project_id=job.project_id,
            project_name=job.project_name,
            tokenizer_job_id=job.tokenizer_job_id,
            tokenizer_name=job.tokenizer_name,
            model_config=job.model_config,
            training_config=job.training_config,
            dataloader_config=job.dataloader_config,
            resolved_runtime=job.resolved_runtime,
            memory_estimate=job.memory_estimate,
            artifact_dir=job.artifact_dir,
            artifact_bundle_file=job.artifact_bundle_file,
            stats_path=job.stats_path,
            samples_path=job.samples_path,
            stdout_path=job.stdout_path,
            stderr_path=job.stderr_path,
            last_step=job.last_step,
            max_steps=job.max_steps,
            elapsed_seconds=optional_float_from_payload(runtime_state, "elapsed_seconds"),
            eta_seconds=optional_float_from_payload(runtime_state, "eta_seconds"),
            latest_loss=job.latest_loss,
            latest_grad_norm=job.latest_grad_norm,
            latest_lr=job.latest_lr,
            latest_tokens_per_sec=job.latest_tokens_per_sec,
            checkpoint_count=job.checkpoint_count,
            sample_count=job.sample_count,
            error=job.error,
            process_id=job.process_id,
            output_size_bytes=job.output_size_bytes,
            executor_kind=job.executor_kind,
            executor_status=job.executor_status,
            runpod_pod_id=job.runpod_pod_id,
            runpod_pod_name=job.runpod_pod_name,
            runpod_network_volume_id=job.runpod_network_volume_id,
            runpod_data_center_id=job.runpod_data_center_id,
            runpod_gpu_type_id=job.runpod_gpu_type_id,
            runpod_gpu_count=job.runpod_gpu_count,
            runpod_cloud_type=job.runpod_cloud_type,
            runpod_interruptible=job.runpod_interruptible,
            runpod_cost_per_hr=job.runpod_cost_per_hr,
            runpod_public_ip=job.runpod_public_ip,
            runpod_port_mappings=job.runpod_port_mappings,
            runpod_agent_base_url=job.runpod_agent_base_url,
            runpod_last_heartbeat_at=job.runpod_last_heartbeat_at,
            runpod_last_sync_at=job.runpod_last_sync_at,
            runpod_cleanup_policy=job.runpod_cleanup_policy,
            remote_workspace_path=job.remote_workspace_path,
            remote_error=job.remote_error,
        )


def issue(code: str, message: str, path: str, *, severity: str = "error") -> TrainingIssue:
    return TrainingIssue(code=code, message=message, path=path, severity=severity)


def optional_float_from_payload(payload: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def validation_issues(code: str, base_path: str, exc: ValidationError) -> list[TrainingIssue]:
    items = exc.errors(include_url=False, include_input=False, include_context=False)
    if not items:
        return [issue(code, "Validation failed.", base_path)]
    return [
        issue(
            code,
            humanize_validation_message(str(item.get("msg") or "Validation failed.")),
            json_path(base_path, item.get("loc")),
        )
        for item in items
    ]


def humanize_validation_message(message: str) -> str:
    cleaned = message.strip()
    if cleaned.startswith("Value error, "):
        cleaned = cleaned.removeprefix("Value error, ").strip()
    if cleaned == "sum of lr_scheduler steps must equal max_steps":
        return (
            "LR scheduler steps must add up to max_steps. "
            "Use the suggested fix below or edit training_config.lr_scheduler."
        )
    return cleaned[:1].upper() + cleaned[1:] if cleaned else "Validation failed."


def json_path(base_path: str, loc: object) -> str:
    if not isinstance(loc, (list, tuple)) or not loc:
        return base_path

    path = base_path
    for part in loc:
        if isinstance(part, int):
            path += f"[{part}]"
        elif isinstance(part, str) and part:
            path += f".{part}"
    return path


def load_config_dict(payload: dict[str, Any]) -> LLMConfig:
    return LLMConfig.model_validate(payload)


def tokenizer_display_name(
    tokenizer_config: dict[str, Any],
    job_id: str,
    artifact_file: str | None,
) -> str:
    name = tokenizer_config.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    if isinstance(artifact_file, str) and artifact_file.strip():
        return artifact_file.strip()
    return f"Tokenizer {job_id[:8]}"


def collect_missing_special_tokens(tokenizer: Tokenizer, config: TrainingDataloaderConfig) -> list[str]:
    required_tokens: list[str] = []
    if config.add_bos and config.bos_token is not None:
        required_tokens.append(config.bos_token)
    if config.add_eos and config.eos_token is not None:
        required_tokens.append(config.eos_token)
    if not config.drop_last:
        if config.pad_token is not None:
            required_tokens.append(config.pad_token)
        elif config.eos_token is not None:
            required_tokens.append(config.eos_token)

    missing = []
    for token in required_tokens:
        if tokenizer.token_to_id(token) is None:
            missing.append(token)
    return missing


def validate_local_data_files(config: TrainingDataloaderConfig) -> list[TrainingIssue]:
    errors: list[TrainingIssue] = []
    for dataset_index, dataset in enumerate(config.datasets):
        if dataset.data_files is None:
            continue
        paths = flatten_data_files(dataset.data_files)
        for raw_path in paths:
            if "://" in raw_path:
                continue
            resolved = resolve_data_path(raw_path)
            if has_glob_magic(raw_path):
                if not glob.glob(str(resolved), recursive=True):
                    errors.append(
                        issue(
                            "local_dataset_file_missing",
                            f"Local dataset glob did not match any files: {raw_path}",
                            f"$.dataloader_config.datasets[{dataset_index}].data_files",
                        )
                    )
            elif not resolved.exists():
                errors.append(
                    issue(
                        "local_dataset_file_missing",
                        f"Local dataset file does not exist: {raw_path}",
                        f"$.dataloader_config.datasets[{dataset_index}].data_files",
                    )
                )
    return errors


def add_scheduler_step_fix(
    fixes: list[TrainingFixSuggestion],
    training_config: dict[str, Any],
) -> None:
    max_steps = training_config.get("max_steps")
    scheduler = training_config.get("lr_scheduler")
    schedulers = scheduler.get("schedulers") if isinstance(scheduler, dict) else None
    if not isinstance(max_steps, int) or max_steps <= 0 or not isinstance(schedulers, list):
        return

    current_total = 0
    for item in schedulers:
        if isinstance(item, dict) and isinstance(item.get("steps"), int):
            current_total += int(item["steps"])
    if current_total == max_steps:
        return

    fixes.append(
        TrainingFixSuggestion(
            code="match_scheduler_steps_to_max_steps",
            label="Match scheduler to max_steps",
            description=f"Replace the LR scheduler with a warmup/cosine schedule totaling {max_steps} steps.",
            path="training_config.lr_scheduler",
            value=build_default_scheduler(max_steps),
        )
    )


def build_default_scheduler(max_steps: int) -> dict[str, Any]:
    if max_steps <= 1:
        return {
            "type": "sequential",
            "schedulers": [{"type": "constant", "steps": 1, "factor": 1.0}],
        }

    warmup_steps = max(1, min(50, max_steps // 10))
    decay_steps = max_steps - warmup_steps
    if decay_steps <= 0:
        return {
            "type": "sequential",
            "schedulers": [
                {
                    "type": "linear",
                    "steps": max_steps,
                    "start_factor": 0.1,
                    "end_factor": 1.0,
                }
            ],
        }

    return {
        "type": "sequential",
        "schedulers": [
            {
                "type": "linear",
                "steps": warmup_steps,
                "start_factor": 0.1,
                "end_factor": 1.0,
            },
            {
                "type": "cosine_annealing",
                "steps": decay_steps,
                "eta_min": 1e-5,
            },
        ],
    }


def runtime_batch_fixes(
    error_message: str,
    training_config: TrainingConfig,
    max_memory_batch_size: int,
) -> list[TrainingFixSuggestion]:
    fixes: list[TrainingFixSuggestion] = []
    if training_config.micro_batch_size is not None:
        fixes.append(
            TrainingFixSuggestion(
                code="auto_select_micro_batch_size",
                label="Auto-select micro batch size",
                description="Remove micro_batch_size so preflight can choose the largest valid value for memory and accumulation.",
                path="training_config.micro_batch_size",
                value=None,
            )
        )

    if "total_batch_size must be divisible by seq_len" in error_message:
        nearest_total_batch_size = nearest_divisible_total_batch_size(
            training_config.total_batch_size,
            training_config.seq_len,
        )
        if nearest_total_batch_size != training_config.total_batch_size:
            fixes.append(
                TrainingFixSuggestion(
                    code="make_total_batch_size_divisible",
                    label="Make total batch size divisible",
                    description=f"Set total_batch_size to {nearest_total_batch_size}, the nearest multiple of seq_len.",
                    path="training_config.total_batch_size",
                    value=nearest_total_batch_size,
                )
            )

    if "micro_batch_size exceeds the memory-estimated maximum" in error_message and max_memory_batch_size > 0:
        fixes.append(
            TrainingFixSuggestion(
                code="cap_micro_batch_size_to_memory",
                label="Fit micro batch size to memory",
                description=f"Set micro_batch_size to the memory-estimated maximum ({max_memory_batch_size}).",
                path="training_config.micro_batch_size",
                value=max_memory_batch_size,
            )
        )

    return fixes


def nearest_divisible_total_batch_size(total_batch_size: int, seq_len: int) -> int:
    if seq_len <= 0:
        return total_batch_size
    lower = max(seq_len, (total_batch_size // seq_len) * seq_len)
    upper = lower if lower == total_batch_size else lower + seq_len
    if abs(total_batch_size - lower) <= abs(upper - total_batch_size):
        return lower
    return upper


def cadence_for_fraction(max_steps: int, fraction: float) -> int:
    return max(1, min(max_steps, round(max_steps * fraction)))


def flatten_data_files(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        results: list[str] = []
        for item in value:
            results.extend(flatten_data_files(item))
        return results
    if isinstance(value, dict):
        results: list[str] = []
        for item in value.values():
            results.extend(flatten_data_files(item))
        return results
    return []


def resolve_data_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return IMPORT_ROOT / candidate


def has_glob_magic(value: str) -> bool:
    return any(ch in value for ch in "*?[")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            items.append(parsed)
    return items


def tail_lines(path: Path, lines: int | None) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    if lines is None or lines <= 0:
        return text.splitlines()
    return text.splitlines()[-max(lines, 1) :]


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for candidate in path.rglob("*"):
        if candidate.is_file():
            total += candidate.stat().st_size
    return total


def sanitized_file_stem(value: str) -> str:
    stem = _FILENAME_SANITIZER.sub("-", value).strip("-")
    return stem if stem else "training-run"


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_identifier(value: str, pattern: re.Pattern[str]) -> None:
    if not pattern.fullmatch(value):
        raise KeyError(value)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def default_training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
