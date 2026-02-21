from __future__ import annotations

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from tokenizers import Tokenizer

from .config import max_job_workers, output_dir
from .models import (
    EvaluationSource,
    JobState,
    JobStatus,
    TokenPreviewTokenResponse,
    TokenizerPreviewResponse,
    TokenizerStatsResponse,
    TrainTokenizerRequest,
    TrainingJobResponse,
)
from .storage import StoredJob, StudioStore

IMPORT_ROOT = Path(__file__).resolve().parents[4]
if str(IMPORT_ROOT) not in sys.path:
    sys.path.append(str(IMPORT_ROOT))

from tokenizer.dataloader import build_dataset, measure_text
from tokenizer.dataloader_config import DataloaderConfig
from tokenizer.loader import TokenizerConfig
from tokenizer.tokenizer import ConfigurableTokenizer


_FILENAME_SANITIZER = re.compile(r"[^a-zA-Z0-9._-]+")
_TRAINING_DATASET_EVAL_SENTINEL = "__training_dataset__"
_UNSET = object()
_PROGRESS_MIN_DELTA = 0.02
_PROGRESS_MIN_INTERVAL_SECONDS = 0.65


class TrainingJobManager:
    def __init__(self, store: StudioStore, workers: int | None = None) -> None:
        self._store = store
        max_workers = workers if workers is not None else max_job_workers()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="tokenizer-train",
        )

    def create_job(self, request: TrainTokenizerRequest) -> TrainingJobResponse:
        tokenizer_config = TokenizerConfig.model_validate(request.tokenizer_config)
        dataloader_config = DataloaderConfig.model_validate(request.dataloader_config)

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

        try:
            self._executor.submit(
                self._run_training_job,
                job_id,
                tokenizer_config,
                dataloader_config,
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

    def get_artifact_path(self, job_id: str) -> Path:
        job = self._store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)

        if job.artifact_path is None:
            raise FileNotFoundError("This job has not produced an artifact yet")

        path = Path(job.artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact file does not exist: {path}")
        return path

    def preview_tokens(self, job_id: str, text: str) -> TokenizerPreviewResponse:
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

    def _run_training_job(
        self,
        job_id: str,
        tokenizer_config: TokenizerConfig,
        dataloader_config: DataloaderConfig,
        evaluation_thresholds: list[int],
    ) -> None:
        try:
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

            output = output_dir()
            output.mkdir(parents=True, exist_ok=True)
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
        self._update_job(
            job_id,
            status=JobStatus.running,
            stage=stage,
            progress=progress,
            started_at=_utc_now(),
            error=None,
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

    def _set_failed(self, job_id: str, error: str) -> None:
        self._update_job(
            job_id,
            status=JobStatus.failed,
            stage="Failed",
            progress=1.0,
            finished_at=_utc_now(),
            error=error,
        )

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
            consumed += measure_text(text, budget_unit)
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
