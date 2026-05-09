from __future__ import annotations

import re
import shutil
import threading
from pathlib import Path
from typing import Any

from .config import get_settings, training_exports_dir
from .schemas import write_json
from .tokenizer_storage import StudioStore as TokenizerStudioStore
from .training_runs.schemas import (
    CreateTrainingJobRequest,
    TrainingAssetRef,
    TrainingCheckpointsResponse,
    TrainingJobResponse,
    TrainingJobState,
    TrainingJobStatus,
    TrainingLogsResponse,
    TrainingMetricsResponse,
    TrainingPreflightRequest,
    TrainingPreflightResponse,
    TrainingSamplesResponse,
)
from .training_runs.store import StoredTrainingJob, TrainingStudioStore
from .training_runs.executors import (
    ExecutionSnapshot,
    LocalSubprocessExecutor,
    RunPodPodExecutor,
    TrainingExecutor,
    TrainingJobBundle,
)
from .training_runs.artifacts import build_artifact_archive, prepare_training_job
from .training_runs.preflight import ResolvedPreflightContext, TrainingPreflightService
from .training_runs.responses import job_to_response
from .training_runs.refresh import ExecutorRefreshThrottle
from .training_runs.runtime_files import (
    directory_size,
    latest_metric_updates,
    list_checkpoint_entries,
    load_optional_json,
    parse_datetime,
    read_metric_points,
    read_sample_entries,
    read_training_log_lines,
    utc_now,
)

_JOB_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{6,64}$")
_RUNPOD_REFRESH_CACHE_SECONDS = 1.5


class TrainingRunManager:
    def __init__(
        self,
        *,
        store: TrainingStudioStore,
        tokenizer_store: TokenizerStudioStore,
    ) -> None:
        self._store = store
        self._tokenizer_store = tokenizer_store
        self._preflight_service = TrainingPreflightService(tokenizer_store=tokenizer_store)
        self._local_executor = LocalSubprocessExecutor()
        self._runpod_executor = RunPodPodExecutor()
        self._executors: dict[str, TrainingExecutor] = {
            self._local_executor.kind: self._local_executor,
            self._runpod_executor.kind: self._runpod_executor,
        }
        self._refresh_locks: dict[str, threading.RLock] = {}
        self._refresh_locks_guard = threading.Lock()
        self._refresh_throttle = ExecutorRefreshThrottle(
            runpod_kind=self._runpod_executor.kind,
            interval_seconds=_RUNPOD_REFRESH_CACHE_SECONDS,
        )
        self._last_runpod_refresh_at = self._refresh_throttle.last_runpod_refresh_at

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
        tokenizer_source_path = self._require_tokenizer_artifact_path(request.tokenizer_job_id)
        prepared = prepare_training_job(
            request=request,
            context=context,
            preflight_payload=preflight_payload,
            tokenizer_source_path=tokenizer_source_path,
            training_jobs_root=settings.training_jobs_dir,
            runpod_default_gpu_count=settings.runpod_default_gpu_count,
            created_at=utc_now(),
        )
        stored = prepared.stored
        bundle = prepared.bundle
        job_id = stored.id
        job_dir = prepared.job_dir
        self._store.create_job(stored)
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

        if executor.kind == self._local_executor.kind and "_spawn_process" in self.__dict__:
            custom_spawn = self.__dict__["_spawn_process"]

            def spawn_from_bundle(local_bundle: TrainingJobBundle):
                return custom_spawn(
                    job_id=local_bundle.job_id,
                    model_config_path=local_bundle.model_config_path,
                    tokenizer_path=local_bundle.tokenizer_path,
                    training_config_path=local_bundle.training_config_path,
                    dataloader_config_path=local_bundle.dataloader_config_path,
                    output_dir=local_bundle.job_dir,
                )

            self._local_executor._spawn_process = spawn_from_bundle  # type: ignore[method-assign]

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
        return TrainingMetricsResponse(
            job_id=job_id,
            metrics=read_metric_points(Path(job.stats_path), limit=limit),
        )

    def get_samples(self, job_id: str, *, limit: int = 50) -> TrainingSamplesResponse:
        job = self.get_job(job_id)
        return TrainingSamplesResponse(
            job_id=job_id,
            samples=read_sample_entries(Path(job.samples_path), limit=limit),
        )

    def get_logs(self, job_id: str, *, lines: int | None = None) -> TrainingLogsResponse:
        job = self.get_job(job_id)
        stdout_lines, stderr_lines = read_training_log_lines(
            artifact_dir=Path(job.artifact_dir),
            stdout_path=Path(job.stdout_path),
            stderr_path=Path(job.stderr_path),
            executor_kind=job.executor_kind,
            lines=lines,
        )
        return TrainingLogsResponse(
            job_id=job_id,
            stdout_lines=stdout_lines,
            stderr_lines=stderr_lines,
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
        return TrainingCheckpointsResponse(
            job_id=job_id,
            checkpoints=list_checkpoint_entries(Path(job.artifact_dir)),
        )

    def build_artifact_bundle(self, job_id: str) -> Path:
        job = self.get_job(job_id)
        artifact_dir = Path(job.artifact_dir)
        archive_path = build_artifact_archive(
            artifact_dir=artifact_dir,
            exports_root=training_exports_dir(),
            name=job.name,
            job_id=job.id,
        )
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
        from .training_runs.executors import CleanupPolicy

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
        """Compatibility action that records why automatic RunPod reattach is unavailable."""
        job = self._store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.executor_kind != "runpod_pod":
            raise ValueError("Remote reattach is only available for RunPod jobs.")
        updated = self._store.update_job(
            job_id,
            remote_error=(
                "Remote reattach is unavailable in this version because the raw pod-agent token is not "
                "persisted. Stop the Pod from RunPod or launch a new run."
            ),
        ) or job
        return self._to_response(updated)

    def shutdown(self) -> None:
        self._local_executor.shutdown()

    def _refresh_job(self, job_id: str) -> StoredTrainingJob | None:
        validate_identifier(job_id, _JOB_ID_RE)
        with self._refresh_lock_for_job(job_id):
            return self._refresh_job_locked(job_id)

    def _refresh_job_locked(self, job_id: str) -> StoredTrainingJob | None:
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

        snapshot = ExecutionSnapshot()
        if self._should_refresh_executor(job):
            snapshot = self._executor_for_job(job).refresh(job)
            self._record_executor_refresh(job)
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

        updates.update(latest_metric_updates(Path(job.stats_path)))
        updates["output_size_bytes"] = directory_size(Path(job.artifact_dir))

        if updates:
            job = self._store.update_job(job_id, **updates) or job
        return job

    def _refresh_lock_for_job(self, job_id: str) -> threading.RLock:
        with self._refresh_locks_guard:
            lock = self._refresh_locks.get(job_id)
            if lock is None:
                lock = threading.RLock()
                self._refresh_locks[job_id] = lock
            return lock

    def _should_refresh_executor(self, job: StoredTrainingJob) -> bool:
        return self._get_refresh_throttle().should_refresh(job)

    def _record_executor_refresh(self, job: StoredTrainingJob) -> None:
        self._get_refresh_throttle().record_refresh(job)

    def _get_refresh_throttle(self) -> ExecutorRefreshThrottle:
        throttle = getattr(self, "_refresh_throttle", None)
        if throttle is None:
            throttle = ExecutorRefreshThrottle(
                runpod_kind=self._runpod_executor.kind,
                interval_seconds=_RUNPOD_REFRESH_CACHE_SECONDS,
            )
            legacy_refresh_state = getattr(self, "_last_runpod_refresh_at", None)
            if isinstance(legacy_refresh_state, dict):
                throttle.last_runpod_refresh_at.update(legacy_refresh_state)
            self._refresh_throttle = throttle
            self._last_runpod_refresh_at = throttle.last_runpod_refresh_at
        return throttle

    def _state_updates_from_runtime(self, payload: dict[str, Any]) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        if "status" in payload and isinstance(payload["status"], str):
            status = TrainingJobStatus(payload["status"])
            updates["status"] = status
            if status in {TrainingJobStatus.completed, TrainingJobStatus.failed, TrainingJobStatus.cancelled}:
                updates["executor_status"] = status.value
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
        return self._get_preflight_service().resolve_context(request)

    def _load_project_asset(self, project_id: str) -> tuple[TrainingAssetRef, dict[str, Any]]:
        return self._get_preflight_service().load_project_asset(project_id)

    def _load_tokenizer_asset(self, tokenizer_job_id: str) -> tuple[TrainingAssetRef, Path]:
        return self._get_preflight_service().load_tokenizer_asset(tokenizer_job_id)

    def _require_tokenizer_artifact_path(self, tokenizer_job_id: str) -> Path:
        return self._get_preflight_service().require_tokenizer_artifact_path(tokenizer_job_id)

    def _get_preflight_service(self) -> TrainingPreflightService:
        service = getattr(self, "_preflight_service", None)
        if service is None:
            service = TrainingPreflightService(tokenizer_store=self._tokenizer_store)
            self._preflight_service = service
        return service

    def _executor_for_kind(self, kind: str) -> TrainingExecutor:
        executor = self._executors.get(kind)
        if executor is None:
            raise ValueError(f"Unsupported training execution target: {kind}")
        return executor

    def _executor_for_job(self, job: StoredTrainingJob) -> TrainingExecutor:
        return self._executor_for_kind(job.executor_kind or "local")

    def _to_response(self, job: StoredTrainingJob) -> TrainingJobResponse:
        runtime_state = load_optional_json(Path(job.artifact_dir) / "runtime_state.json")
        return job_to_response(job, runtime_state=runtime_state)


def validate_identifier(value: str, pattern: re.Pattern[str]) -> None:
    if not pattern.fullmatch(value):
        raise KeyError(value)
