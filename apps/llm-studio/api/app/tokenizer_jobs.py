from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .config import tokenizer_max_job_workers, tokenizer_output_dir
from .dataset_credentials import inject_dataset_hf_tokens, split_dataset_hf_tokens
from .logging_config import (
    clear_known_secrets,
    known_secrets_for_scope,
    redact_known_secrets,
    register_known_secrets,
)
from .storage_safety import ensure_free_space, require_managed_path
from .tokenizer_models import (
    EvaluationSource,
    JobState,
    JobStatus,
    TokenPreviewTokenResponse,
    TokenizerPreviewResponse,
    TokenizerStatsResponse,
    TrainTokenizerRequest,
    TrainingJobResponse,
)
from .runtime_paths import ensure_source_root_on_path
from .tokenizer_storage import StoredJob, StudioStore

IMPORT_ROOT = ensure_source_root_on_path()

_FILENAME_SANITIZER = re.compile(r"[^a-zA-Z0-9._-]+")
_TRAINING_DATASET_EVAL_SENTINEL = "__training_dataset__"
_UNSET = object()
_PROGRESS_MIN_DELTA = 0.02
_PROGRESS_MIN_INTERVAL_SECONDS = 0.65
logger = logging.getLogger("llm_studio.tokenizer_jobs")


class TrainingJobManager:
    def __init__(self, store: StudioStore, workers: int | None = None) -> None:
        self._store = store
        max_workers = workers if workers is not None else tokenizer_max_job_workers()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="tokenizer-train",
        )
        self._dataset_hf_tokens: dict[str, tuple[str, ...]] = {}
        self._dataset_hf_tokens_lock = threading.RLock()

    def create_job(self, request: TrainTokenizerRequest) -> TrainingJobResponse:
        from tokenizer.dataloader_config import DataloaderConfig
        from tokenizer.loader import TokenizerConfig

        ensure_free_space(
            tokenizer_output_dir(),
            minimum_free_bytes=256 * 1024 * 1024,
            operation="tokenizer training and artifact creation",
        )
        tokenizer_config = TokenizerConfig.model_validate(request.tokenizer_config)
        sanitized_dataloader_payload, dataset_hf_tokens = split_dataset_hf_tokens(
            request.dataloader_config,
            fallback_token=request.hf_token,
        )
        dataloader_config = DataloaderConfig.model_validate(sanitized_dataloader_payload)
        execution_dataloader_config = DataloaderConfig.model_validate(
            inject_dataset_hf_tokens(sanitized_dataloader_payload, dataset_hf_tokens)
        )

        now = _utc_now()
        job_id = uuid4().hex
        managed = StoredJob(
            id=job_id,
            status=JobStatus.pending,
            stage="Queued",
            progress=0.0,
            created_at=now,
            started_at=None,
            finished_at=None,
            tokenizer_config=tokenizer_config.model_dump(mode="json"),
            dataloader_config=dataloader_config.model_dump(mode="json"),
            evaluation_thresholds=list(request.evaluation_thresholds),
            evaluation_text_path=_TRAINING_DATASET_EVAL_SENTINEL,
            artifact_file=None,
            artifact_path=None,
            stats=None,
            error=None,
        )

        self._store.create_job(managed)
        self._remember_dataset_hf_tokens(job_id, dataset_hf_tokens)
        _log_job_event("tokenizer.job.queued", "Tokenizer job queued.", job_id, JobStatus.pending)

        try:
            self._executor.submit(
                self._run_training_job,
                job_id,
                tokenizer_config,
                execution_dataloader_config,
                request.evaluation_thresholds,
            )
        except Exception as exc:
            self._set_failed(
                job_id,
                f"Failed to queue training execution: {type(exc).__name__}: {exc}",
            )
        return self.get_job(job_id)

    def list_jobs(self) -> list[TrainingJobResponse]:
        return [self._to_response(job) for job in self._store.list_jobs()]

    def get_job(self, job_id: str) -> TrainingJobResponse:
        job = self._store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return self._to_response(job)

    def delete_job(self, job_id: str) -> None:
        job = self._store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)

        if job.status in {JobStatus.pending, JobStatus.running}:
            raise RuntimeError("Only completed or failed jobs can be deleted")

        if job.artifact_path:
            artifact_path = require_managed_path(
                Path(job.artifact_path),
                tokenizer_output_dir(),
                description="tokenizer artifact deletion",
            )
            if artifact_path.exists():
                try:
                    artifact_path.unlink()
                except OSError as exc:
                    raise ValueError(f"Failed to delete artifact file: {artifact_path}") from exc

        deleted = self._store.delete_job(job_id)
        if deleted is None:
            raise KeyError(job_id)
        self._forget_dataset_hf_tokens(job_id)
        _log_job_event("tokenizer.job.deleted", "Tokenizer job deleted.", job_id, job.status)

    def get_artifact_path(self, job_id: str) -> Path:
        job = self._store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)

        if job.artifact_path is None:
            raise FileNotFoundError("This job has not produced an artifact yet")

        path = require_managed_path(
            Path(job.artifact_path),
            tokenizer_output_dir(),
            description="tokenizer artifact",
        )
        if not path.exists():
            raise FileNotFoundError(f"Artifact file does not exist: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Artifact path is not a file: {path}")
        return path

    def preview_tokens(self, job_id: str, text: str) -> TokenizerPreviewResponse:
        from tokenizers import Tokenizer

        artifact_path = self.get_artifact_path(job_id)
        tokenizer = Tokenizer.from_file(str(artifact_path))

        if text == "":
            return TokenizerPreviewResponse(
                job_id=job_id,
                text=text,
                text_length=0,
                num_tokens=0,
                tokens=[],
            )

        encoding = tokenizer.encode(text)
        offsets = getattr(encoding, "offsets", [])

        tokens: list[TokenPreviewTokenResponse] = []
        for index, (token_id, token_text) in enumerate(zip(encoding.ids, encoding.tokens)):
            start = 0
            end = 0
            if index < len(offsets):
                start, end = offsets[index]
            tokens.append(
                TokenPreviewTokenResponse(
                    index=index,
                    id=int(token_id),
                    token=token_text,
                    start=int(start),
                    end=int(end),
                )
            )

        return TokenizerPreviewResponse(
            job_id=job_id,
            text=text,
            text_length=len(text),
            num_tokens=len(tokens),
            tokens=tokens,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)
        # Active workers can still emit during process teardown. Their scopes
        # remain until terminal handling clears them or process memory exits.

    def _run_training_job(
        self,
        job_id: str,
        tokenizer_config: TokenizerConfig,
        dataloader_config: DataloaderConfig,
        evaluation_thresholds: list[int],
    ) -> None:
        try:
            from tokenizer.dataloader import build_dataset
            from tokenizer.tokenizer import ConfigurableTokenizer

            self._set_running(job_id, "Initializing tokenizer", 0.03)
            configurable = ConfigurableTokenizer(tokenizer_config)

            self._update_job(job_id, stage="Preparing training dataset stream", progress=0.10)
            dataset = build_dataset(dataloader_config)

            training_stage_prefix = "Training tokenizer"
            self._update_job(
                job_id,
                stage=self._format_budget_stage(
                    stage_prefix=training_stage_prefix,
                    budget_unit=dataloader_config.budget.unit,
                    fraction=0.0,
                    records=0,
                ),
                progress=0.14,
            )
            training_stream = self._stream_with_budget_progress(
                text_iter=dataset,
                job_id=job_id,
                stage_prefix=training_stage_prefix,
                budget_limit=dataloader_config.budget.limit,
                budget_unit=dataloader_config.budget.unit,
                progress_start=0.14,
                progress_end=0.72,
            )
            configurable.tokenizer.train_from_iterator(training_stream, configurable.trainer)

            output = tokenizer_output_dir()
            ensure_free_space(
                output,
                minimum_free_bytes=64 * 1024 * 1024,
                operation="tokenizer artifact save",
            )
            artifact_file = self._artifact_filename(tokenizer_config.name, job_id)
            artifact_path = output / artifact_file

            self._update_job(
                job_id,
                stage="Saving tokenizer artifact",
                progress=0.80,
                artifact_file=artifact_file,
                artifact_path=str(artifact_path),
            )
            configurable.tokenizer.save(str(artifact_path))

            evaluation_stage_prefix = "Evaluating tokenizer on training dataset"
            self._update_job(
                job_id,
                stage=self._format_budget_stage(
                    stage_prefix=evaluation_stage_prefix,
                    budget_unit=dataloader_config.budget.unit,
                    fraction=0.0,
                    records=0,
                ),
                progress=0.86,
            )
            evaluation_stream = self._stream_with_budget_progress(
                text_iter=dataset,
                job_id=job_id,
                stage_prefix=evaluation_stage_prefix,
                budget_limit=dataloader_config.budget.limit,
                budget_unit=dataloader_config.budget.unit,
                progress_start=0.86,
                progress_end=0.98,
            )
            stats = ConfigurableTokenizer.eval_tokenizer_on_iterator(
                thresholds=evaluation_thresholds,
                tokenizer=configurable.tokenizer,
                text_iter=evaluation_stream,
            )
            stats_payload = TokenizerStatsResponse.model_validate(
                stats.__dict__
            ).model_dump(mode="json")

            self._set_completed(
                job_id,
                stage="Completed",
                progress=1.0,
                stats=stats_payload,
                artifact_file=artifact_file,
                artifact_path=str(artifact_path),
            )
        except Exception as exc:
            self._set_failed(job_id, f"{type(exc).__name__}: {exc}")

    def _set_running(self, job_id: str, stage: str, progress: float) -> None:
        previous = self._store.get_job(job_id)
        self._update_job(
            job_id,
            status=JobStatus.running,
            stage=stage,
            progress=progress,
            started_at=_utc_now(),
            error=None,
        )
        if previous is not None and previous.status is not JobStatus.running:
            _log_job_event(
                "tokenizer.job.started",
                "Tokenizer job started.",
                job_id,
                JobStatus.running,
            )

    def _set_completed(
        self,
        job_id: str,
        stage: str,
        progress: float,
        stats: dict[str, Any],
        artifact_file: str,
        artifact_path: str,
    ) -> None:
        self._update_job(
            job_id,
            status=JobStatus.completed,
            stage=stage,
            progress=progress,
            finished_at=_utc_now(),
            stats=stats,
            artifact_file=artifact_file,
            artifact_path=artifact_path,
            error=None,
        )
        _log_job_event(
            "tokenizer.job.completed",
            "Tokenizer job completed.",
            job_id,
            JobStatus.completed,
        )
        self._forget_dataset_hf_tokens(job_id)

    def _set_failed(self, job_id: str, error: str) -> None:
        self._update_job(
            job_id,
            status=JobStatus.failed,
            stage="Failed",
            progress=1.0,
            finished_at=_utc_now(),
            error=redact_known_secrets(error, self._known_dataset_hf_tokens(job_id)),
        )
        _log_job_event("tokenizer.job.failed", "Tokenizer job failed.", job_id, JobStatus.failed)
        self._forget_dataset_hf_tokens(job_id)

    def _remember_dataset_hf_tokens(
        self,
        job_id: str,
        tokens: Iterable[str | None],
    ) -> None:
        scope = _tokenizer_secret_scope(job_id)
        normalized = register_known_secrets(scope, tokens)
        with self._token_registry_lock():
            if normalized:
                self._token_registry()[job_id] = normalized
            else:
                self._token_registry().pop(job_id, None)

    def _known_dataset_hf_tokens(self, job_id: str) -> tuple[str, ...]:
        with self._token_registry_lock():
            stored = self._token_registry().get(job_id)
        return stored or known_secrets_for_scope(_tokenizer_secret_scope(job_id))

    def _forget_dataset_hf_tokens(self, job_id: str) -> None:
        with self._token_registry_lock():
            self._token_registry().pop(job_id, None)
        clear_known_secrets(_tokenizer_secret_scope(job_id))

    def _token_registry(self) -> dict[str, tuple[str, ...]]:
        registry = getattr(self, "_dataset_hf_tokens", None)
        if registry is None:
            registry = {}
            self._dataset_hf_tokens = registry
        return registry

    def _token_registry_lock(self) -> threading.RLock:
        lock = getattr(self, "_dataset_hf_tokens_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._dataset_hf_tokens_lock = lock
        return lock

    def _stream_with_budget_progress(
        self,
        *,
        text_iter: Iterable[str],
        job_id: str,
        stage_prefix: str,
        budget_limit: int,
        budget_unit: str,
        progress_start: float,
        progress_end: float,
    ) -> Iterable[str]:
        consumed = 0
        records = 0
        last_reported_fraction = -1.0
        last_report_time = 0.0

        for raw_text in text_iter:
            text = raw_text if isinstance(raw_text, str) else str(raw_text)
            records += 1
            consumed += _measure_text(text, budget_unit)
            fraction = (consumed / budget_limit) if budget_limit > 0 else 1.0
            clamped_fraction = max(0.0, min(fraction, 1.0))

            now = time.monotonic()
            should_report = (
                last_reported_fraction < 0.0
                or clamped_fraction >= 1.0
                or (clamped_fraction - last_reported_fraction) >= _PROGRESS_MIN_DELTA
                or (now - last_report_time) >= _PROGRESS_MIN_INTERVAL_SECONDS
            )
            if should_report:
                self._update_job(
                    job_id,
                    stage=self._format_budget_stage(
                        stage_prefix=stage_prefix,
                        budget_unit=budget_unit,
                        fraction=clamped_fraction,
                        records=records,
                    ),
                    progress=_phase_progress(
                        clamped_fraction,
                        progress_start=progress_start,
                        progress_end=progress_end,
                    ),
                )
                last_reported_fraction = clamped_fraction
                last_report_time = now

            yield text

        if records == 0:
            self._update_job(
                job_id,
                stage=f"{stage_prefix} (dataset stream is empty)",
                progress=progress_start,
            )

    def _format_budget_stage(
        self,
        *,
        stage_prefix: str,
        budget_unit: str,
        fraction: float,
        records: int,
    ) -> str:
        percent = int(round(max(0.0, min(fraction, 1.0)) * 100))
        if records <= 0:
            return f"{stage_prefix} ({percent}% of {budget_unit} budget)"
        return f"{stage_prefix} ({percent}% of {budget_unit} budget, {records:,} records)"

    def _update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | object = _UNSET,
        stage: str | object = _UNSET,
        progress: float | object = _UNSET,
        started_at: datetime | None | object = _UNSET,
        finished_at: datetime | None | object = _UNSET,
        artifact_file: str | None | object = _UNSET,
        artifact_path: str | None | object = _UNSET,
        stats: dict[str, Any] | None | object = _UNSET,
        error: str | None | object = _UNSET,
    ) -> None:
        updates: dict[str, Any] = {}

        if status is not _UNSET:
            updates["status"] = status
        if stage is not _UNSET:
            updates["stage"] = stage
        if progress is not _UNSET:
            updates["progress"] = progress
        if started_at is not _UNSET:
            updates["started_at"] = started_at
        if finished_at is not _UNSET:
            updates["finished_at"] = finished_at
        if artifact_file is not _UNSET:
            updates["artifact_file"] = artifact_file
        if artifact_path is not _UNSET:
            updates["artifact_path"] = artifact_path
        if stats is not _UNSET:
            updates["stats"] = stats
        if error is not _UNSET:
            updates["error"] = error

        if updates:
            self._store.update_job(job_id, **updates)

    def _artifact_filename(self, config_name: str, job_id: str) -> str:
        base = _FILENAME_SANITIZER.sub("-", config_name).strip("-")
        if base == "":
            base = "tokenizer"
        return f"{base}-{job_id[:8]}.json"

    def _to_response(self, job: StoredJob) -> TrainingJobResponse:
        stats = None
        if job.stats is not None:
            stats = TokenizerStatsResponse.model_validate(job.stats)
        return TrainingJobResponse(
            id=job.id,
            status=job.status,
            state=_derive_job_state(job.status, job.stage),
            stage=job.stage,
            progress=job.progress,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            tokenizer_config=job.tokenizer_config,
            dataloader_config=job.dataloader_config,
            evaluation_source=_derive_evaluation_source(job.evaluation_text_path),
            evaluation_thresholds=job.evaluation_thresholds,
            evaluation_text_path=job.evaluation_text_path,
            artifact_file=job.artifact_file,
            artifact_path=job.artifact_path,
            stats=stats,
            error=job.error,
        )


def _phase_progress(fraction: float, *, progress_start: float, progress_end: float) -> float:
    clamped = max(0.0, min(fraction, 1.0))
    start = min(progress_start, progress_end)
    end = max(progress_start, progress_end)
    return start + (end - start) * clamped


def _measure_text(text: str, unit: str) -> int:
    if unit == "chars":
        return len(text)
    if unit == "bytes":
        return len(text.encode("utf-8"))
    raise ValueError(f"Unknown text unit: {unit}")


def _derive_evaluation_source(value: str) -> EvaluationSource:
    if value == _TRAINING_DATASET_EVAL_SENTINEL:
        return EvaluationSource.training_dataset
    return EvaluationSource.legacy_file


def _derive_job_state(status: JobStatus, stage: str) -> JobState:
    if status is JobStatus.pending:
        return JobState.queued
    if status is JobStatus.completed:
        return JobState.completed
    if status is JobStatus.failed:
        return JobState.failed

    normalized_stage = stage.strip().lower()
    if normalized_stage.startswith("initializing tokenizer"):
        return JobState.initializing
    if normalized_stage.startswith("preparing training dataset"):
        return JobState.preparing_dataset
    if normalized_stage.startswith("training tokenizer"):
        return JobState.training
    if normalized_stage.startswith("saving tokenizer artifact"):
        return JobState.saving_artifact
    if normalized_stage.startswith("evaluating tokenizer"):
        return JobState.evaluating
    return JobState.running


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _tokenizer_secret_scope(job_id: str) -> str:
    return f"tokenizer:{job_id}"


def _log_job_event(event_id: str, message: str, job_id: str, status: JobStatus) -> None:
    logger.info(
        message,
        extra={
            "event_id": event_id,
            "event_fields": {"job_id": job_id, "status": status.value},
        },
    )
