from __future__ import annotations

import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from tokenizers import Tokenizer

from .config import API_ROOT, max_job_workers, output_dir
from .models import (
    JobStatus,
    TokenPreviewTokenResponse,
    TokenizerPreviewResponse,
    TokenizerStatsResponse,
    TrainTokenizerRequest,
    TrainingJobResponse,
)

IMPORT_ROOT = Path(__file__).resolve().parents[4]
if str(IMPORT_ROOT) not in sys.path:
    sys.path.append(str(IMPORT_ROOT))

from tokenizer.dataloader import build_dataset
from tokenizer.dataloader_config import DataloaderConfig
from tokenizer.loader import TokenizerConfig
from tokenizer.tokenizer import ConfigurableTokenizer


_FILENAME_SANITIZER = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass
class ManagedJob:
    id: str
    status: JobStatus
    stage: str
    progress: float
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    tokenizer_config: dict[str, Any]
    dataloader_config: dict[str, Any]
    evaluation_thresholds: list[int]
    evaluation_text_path: str
    artifact_file: str | None
    artifact_path: str | None
    stats: dict[str, Any] | None
    error: str | None


class TrainingJobManager:
    def __init__(self, workers: int | None = None) -> None:
        max_workers = workers if workers is not None else max_job_workers()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="tokenizer-train",
        )
        self._jobs: dict[str, ManagedJob] = {}
        self._lock = Lock()

    def create_job(self, request: TrainTokenizerRequest) -> TrainingJobResponse:
        tokenizer_config = TokenizerConfig.model_validate(request.tokenizer_config)
        dataloader_config = DataloaderConfig.model_validate(request.dataloader_config)
        resolved_eval_text_path = self._resolve_eval_text_path(request.evaluation_text_path)

        now = _utc_now()
        job_id = uuid4().hex
        managed = ManagedJob(
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
            evaluation_text_path=str(resolved_eval_text_path),
            artifact_file=None,
            artifact_path=None,
            stats=None,
            error=None,
        )

        with self._lock:
            self._jobs[job_id] = managed

        self._executor.submit(
            self._run_training_job,
            job_id,
            tokenizer_config,
            dataloader_config,
            request.evaluation_thresholds,
            resolved_eval_text_path,
        )
        return self.get_job(job_id)

    def list_jobs(self) -> list[TrainingJobResponse]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda x: x.created_at, reverse=True)
        return [self._to_response(job) for job in jobs]

    def get_job(self, job_id: str) -> TrainingJobResponse:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return self._to_response(job)

    def get_artifact_path(self, job_id: str) -> Path:
        with self._lock:
            job = self._jobs.get(job_id)
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
        eval_path: Path,
    ) -> None:
        try:
            self._set_running(job_id, "Initializing tokenizer", 0.1)
            configurable = ConfigurableTokenizer(tokenizer_config)

            self._update_job(job_id, stage="Building dataset stream", progress=0.25)
            dataset = build_dataset(dataloader_config)

            self._update_job(
                job_id,
                stage="Training tokenizer (streaming dataset)",
                progress=0.6,
            )
            configurable.tokenizer.train_from_iterator(dataset, configurable.trainer)

            output = output_dir()
            output.mkdir(parents=True, exist_ok=True)
            artifact_file = self._artifact_filename(tokenizer_config.name, job_id)
            artifact_path = output / artifact_file

            self._update_job(
                job_id,
                stage="Saving tokenizer artifact",
                progress=0.8,
                artifact_file=artifact_file,
                artifact_path=str(artifact_path),
            )
            configurable.tokenizer.save(str(artifact_path))

            self._update_job(
                job_id,
                stage="Evaluating tokenizer statistics",
                progress=0.92,
            )
            stats = _evaluate_tokenizer(configurable.tokenizer, eval_path, evaluation_thresholds)

            self._set_completed(
                job_id,
                stage="Completed",
                progress=1.0,
                stats=stats.model_dump(mode="json"),
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

    def _update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        stage: str | None = None,
        progress: float | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        artifact_file: str | None = None,
        artifact_path: str | None = None,
        stats: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if status is not None:
                job.status = status
            if stage is not None:
                job.stage = stage
            if progress is not None:
                job.progress = max(0.0, min(progress, 1.0))
            if started_at is not None:
                job.started_at = started_at
            if finished_at is not None:
                job.finished_at = finished_at
            if artifact_file is not None:
                job.artifact_file = artifact_file
            if artifact_path is not None:
                job.artifact_path = artifact_path
            if stats is not None:
                job.stats = stats
            if error is not None:
                job.error = error

    def _resolve_eval_text_path(self, raw: str) -> Path:
        if raw.strip() == "":
            raise ValueError("evaluation_text_path is required")

        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = API_ROOT / path
        if not path.exists() or not path.is_file():
            raise ValueError(f"evaluation_text_path does not point to a file: {path}")
        return path

    def _artifact_filename(self, config_name: str, job_id: str) -> str:
        base = _FILENAME_SANITIZER.sub("-", config_name).strip("-")
        if base == "":
            base = "tokenizer"
        return f"{base}-{job_id[:8]}.json"

    def _to_response(self, job: ManagedJob) -> TrainingJobResponse:
        stats = None
        if job.stats is not None:
            stats = TokenizerStatsResponse.model_validate(job.stats)
        return TrainingJobResponse(
            id=job.id,
            status=job.status,
            stage=job.stage,
            progress=job.progress,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            tokenizer_config=job.tokenizer_config,
            dataloader_config=job.dataloader_config,
            evaluation_thresholds=job.evaluation_thresholds,
            evaluation_text_path=job.evaluation_text_path,
            artifact_file=job.artifact_file,
            artifact_path=job.artifact_path,
            stats=stats,
            error=job.error,
        )


def _evaluate_tokenizer(
    tokenizer: Tokenizer,
    text_path: Path,
    thresholds: list[int],
) -> TokenizerStatsResponse:
    text = text_path.read_text(encoding="utf-8")
    if text == "":
        return TokenizerStatsResponse(
            num_chars=0,
            num_tokens=0,
            token_per_char=0.0,
            vocab_size=tokenizer.get_vocab_size(),
            num_used_tokens=0,
            num_unused_tokens=tokenizer.get_vocab_size(),
            rare_tokens={threshold: 0 for threshold in thresholds},
            rare_token_fraction={threshold: 0.0 for threshold in thresholds},
        )

    encoding = tokenizer.encode(text)
    ids = encoding.ids
    num_tokens = len(ids)
    num_chars = len(text)
    tokens_per_char = (num_tokens / num_chars) if num_chars > 0 else 0.0

    frequencies = Counter(ids)
    num_used_tokens = len(frequencies)
    vocab_size = tokenizer.get_vocab_size()

    rare_tokens: dict[int, int] = {}
    rare_token_fraction: dict[int, float] = {}

    for threshold in thresholds:
        rare_count = sum(1 for count in frequencies.values() if count < threshold)
        fraction = (rare_count / num_used_tokens) if num_used_tokens > 0 else 0.0
        rare_tokens[threshold] = rare_count
        rare_token_fraction[threshold] = fraction

    return TokenizerStatsResponse(
        num_chars=num_chars,
        num_tokens=num_tokens,
        token_per_char=tokens_per_char,
        vocab_size=vocab_size,
        num_used_tokens=num_used_tokens,
        num_unused_tokens=max(vocab_size - num_used_tokens, 0),
        rare_tokens=rare_tokens,
        rare_token_fraction=rare_token_fraction,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
