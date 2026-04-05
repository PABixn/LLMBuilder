from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text, create_engine, event, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import training_database_url
from .training_models import TrainingJobState, TrainingJobStatus


class Base(DeclarativeBase):
    pass


class TrainingRunRow(Base):
    __tablename__ = "llm_training_jobs"
    __table_args__ = (
        Index("ix_llm_training_jobs_created_at", "created_at"),
        Index("ix_llm_training_jobs_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    state: Mapped[str] = mapped_column(String(64), nullable=False)
    stage: Mapped[str] = mapped_column(String(255), nullable=False)
    progress: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    project_id: Mapped[str] = mapped_column(String(64), nullable=False)
    project_name: Mapped[str] = mapped_column(String(255), nullable=False)
    tokenizer_job_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tokenizer_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    training_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    dataloader_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    resolved_runtime: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    memory_estimate: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    artifact_dir: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_bundle_file: Mapped[str | None] = mapped_column(String(512), nullable=True)
    stats_path: Mapped[str] = mapped_column(Text, nullable=False)
    samples_path: Mapped[str] = mapped_column(Text, nullable=False)
    stdout_path: Mapped[str] = mapped_column(Text, nullable=False)
    stderr_path: Mapped[str] = mapped_column(Text, nullable=False)
    last_step: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_steps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latest_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    latest_grad_norm: Mapped[float | None] = mapped_column(Float, nullable=True)
    latest_lr: Mapped[float | None] = mapped_column(Float, nullable=True)
    latest_tokens_per_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    checkpoint_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    process_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


@dataclass
class StoredTrainingJob:
    id: str
    name: str
    status: TrainingJobStatus
    state: TrainingJobState
    stage: str
    progress: float
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    project_id: str
    project_name: str
    tokenizer_job_id: str
    tokenizer_name: str
    model_config: dict[str, Any]
    training_config: dict[str, Any]
    dataloader_config: dict[str, Any]
    resolved_runtime: dict[str, Any] | None
    memory_estimate: dict[str, Any] | None
    artifact_dir: str
    artifact_bundle_file: str | None
    stats_path: str
    samples_path: str
    stdout_path: str
    stderr_path: str
    last_step: int
    max_steps: int
    latest_loss: float | None
    latest_grad_norm: float | None
    latest_lr: float | None
    latest_tokens_per_sec: float | None
    checkpoint_count: int
    sample_count: int
    error: str | None
    process_id: int | None
    output_size_bytes: int


class TrainingStudioStore:
    def __init__(self, url: str | None = None) -> None:
        self._url = url or training_database_url()
        self._engine = self._build_engine(self._url)
        self._session_factory = sessionmaker(
            bind=self._engine,
            autoflush=False,
            expire_on_commit=False,
        )

    def initialize(self) -> None:
        Base.metadata.create_all(self._engine)

    def dispose(self) -> None:
        self._engine.dispose()

    def create_job(self, job: StoredTrainingJob) -> None:
        with self._session() as session:
            session.add(
                TrainingRunRow(
                    id=job.id,
                    name=job.name,
                    status=job.status.value,
                    state=job.state.value,
                    stage=job.stage,
                    progress=_clamp_progress(job.progress),
                    created_at=_ensure_utc(job.created_at),
                    started_at=_ensure_optional_utc(job.started_at),
                    finished_at=_ensure_optional_utc(job.finished_at),
                    project_id=job.project_id,
                    project_name=job.project_name,
                    tokenizer_job_id=job.tokenizer_job_id,
                    tokenizer_name=job.tokenizer_name,
                    model_config=dict(job.model_config),
                    training_config=dict(job.training_config),
                    dataloader_config=dict(job.dataloader_config),
                    resolved_runtime=None if job.resolved_runtime is None else dict(job.resolved_runtime),
                    memory_estimate=None if job.memory_estimate is None else dict(job.memory_estimate),
                    artifact_dir=job.artifact_dir,
                    artifact_bundle_file=job.artifact_bundle_file,
                    stats_path=job.stats_path,
                    samples_path=job.samples_path,
                    stdout_path=job.stdout_path,
                    stderr_path=job.stderr_path,
                    last_step=max(0, int(job.last_step)),
                    max_steps=max(0, int(job.max_steps)),
                    latest_loss=job.latest_loss,
                    latest_grad_norm=job.latest_grad_norm,
                    latest_lr=job.latest_lr,
                    latest_tokens_per_sec=job.latest_tokens_per_sec,
                    checkpoint_count=max(0, int(job.checkpoint_count)),
                    sample_count=max(0, int(job.sample_count)),
                    error=job.error,
                    process_id=job.process_id,
                    output_size_bytes=max(0, int(job.output_size_bytes)),
                )
            )

    def get_job(self, job_id: str) -> StoredTrainingJob | None:
        with self._session() as session:
            row = session.get(TrainingRunRow, job_id)
            if row is None:
                return None
            return _row_to_stored_job(row)

    def list_jobs(self) -> list[StoredTrainingJob]:
        with self._session() as session:
            rows = session.scalars(
                select(TrainingRunRow).order_by(TrainingRunRow.created_at.desc())
            ).all()
            return [_row_to_stored_job(row) for row in rows]

    def update_job(self, job_id: str, **updates: Any) -> StoredTrainingJob | None:
        with self._session() as session:
            row = session.get(TrainingRunRow, job_id)
            if row is None:
                return None

            for field_name, value in updates.items():
                if field_name == "status":
                    row.status = _coerce_status(value).value
                elif field_name == "state":
                    row.state = _coerce_state(value).value
                elif field_name == "stage":
                    row.stage = str(value)
                elif field_name == "progress":
                    row.progress = _clamp_progress(float(value))
                elif field_name == "created_at":
                    row.created_at = _ensure_utc(value)
                elif field_name == "started_at":
                    row.started_at = _ensure_optional_utc(value)
                elif field_name == "finished_at":
                    row.finished_at = _ensure_optional_utc(value)
                elif field_name == "name":
                    row.name = str(value)
                elif field_name in {"project_id", "project_name", "tokenizer_job_id", "tokenizer_name"}:
                    setattr(row, field_name, str(value))
                elif field_name in {"model_config", "training_config", "dataloader_config"}:
                    setattr(row, field_name, dict(value))
                elif field_name in {"resolved_runtime", "memory_estimate"}:
                    setattr(row, field_name, None if value is None else dict(value))
                elif field_name in {
                    "artifact_dir",
                    "artifact_bundle_file",
                    "stats_path",
                    "samples_path",
                    "stdout_path",
                    "stderr_path",
                    "error",
                }:
                    setattr(row, field_name, None if value is None else str(value))
                elif field_name in {"last_step", "max_steps", "checkpoint_count", "sample_count", "process_id", "output_size_bytes"}:
                    setattr(row, field_name, None if value is None else int(value))
                elif field_name in {"latest_loss", "latest_grad_norm", "latest_lr", "latest_tokens_per_sec"}:
                    setattr(row, field_name, None if value is None else float(value))
                else:
                    raise ValueError(f"Unknown training job update field: {field_name}")

            session.flush()
            return _row_to_stored_job(row)

    def delete_job(self, job_id: str) -> StoredTrainingJob | None:
        with self._session() as session:
            row = session.get(TrainingRunRow, job_id)
            if row is None:
                return None
            stored = _row_to_stored_job(row)
            session.delete(row)
            session.flush()
            return stored

    def mark_incomplete_jobs_failed(self, reason: str) -> int:
        now = _utc_now()
        with self._session() as session:
            result = session.execute(
                update(TrainingRunRow)
                .where(
                    TrainingRunRow.status.in_(
                        [TrainingJobStatus.pending.value, TrainingJobStatus.running.value]
                    )
                )
                .values(
                    status=TrainingJobStatus.failed.value,
                    state=TrainingJobState.failed.value,
                    stage="Interrupted by API restart",
                    progress=1.0,
                    finished_at=now,
                    error=reason,
                )
            )
            return int(result.rowcount or 0)

    @contextmanager
    def _session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _build_engine(self, url: str) -> Engine:
        connect_args: dict[str, Any] = {}

        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            db_file = _sqlite_path_from_url(url)
            if db_file is not None:
                db_file.parent.mkdir(parents=True, exist_ok=True)

        engine = create_engine(
            url,
            connect_args=connect_args,
            pool_pre_ping=True,
            future=True,
        )

        if url.startswith("sqlite"):

            @event.listens_for(engine, "connect")
            def _set_sqlite_pragmas(dbapi_connection: Any, _: Any) -> None:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        return engine


def _row_to_stored_job(row: TrainingRunRow) -> StoredTrainingJob:
    return StoredTrainingJob(
        id=row.id,
        name=row.name,
        status=TrainingJobStatus(row.status),
        state=TrainingJobState(row.state),
        stage=row.stage,
        progress=_clamp_progress(row.progress),
        created_at=_ensure_utc(row.created_at),
        started_at=_ensure_optional_utc(row.started_at),
        finished_at=_ensure_optional_utc(row.finished_at),
        project_id=row.project_id,
        project_name=row.project_name,
        tokenizer_job_id=row.tokenizer_job_id,
        tokenizer_name=row.tokenizer_name,
        model_config=dict(row.model_config),
        training_config=dict(row.training_config),
        dataloader_config=dict(row.dataloader_config),
        resolved_runtime=dict(row.resolved_runtime) if row.resolved_runtime is not None else None,
        memory_estimate=dict(row.memory_estimate) if row.memory_estimate is not None else None,
        artifact_dir=row.artifact_dir,
        artifact_bundle_file=row.artifact_bundle_file,
        stats_path=row.stats_path,
        samples_path=row.samples_path,
        stdout_path=row.stdout_path,
        stderr_path=row.stderr_path,
        last_step=max(0, int(row.last_step)),
        max_steps=max(0, int(row.max_steps)),
        latest_loss=row.latest_loss,
        latest_grad_norm=row.latest_grad_norm,
        latest_lr=row.latest_lr,
        latest_tokens_per_sec=row.latest_tokens_per_sec,
        checkpoint_count=max(0, int(row.checkpoint_count)),
        sample_count=max(0, int(row.sample_count)),
        error=row.error,
        process_id=row.process_id,
        output_size_bytes=max(0, int(row.output_size_bytes)),
    )


def _coerce_status(value: TrainingJobStatus | str) -> TrainingJobStatus:
    if isinstance(value, TrainingJobStatus):
        return value
    return TrainingJobStatus(str(value))


def _coerce_state(value: TrainingJobState | str) -> TrainingJobState:
    if isinstance(value, TrainingJobState):
        return value
    return TrainingJobState(str(value))


def _clamp_progress(progress: float) -> float:
    return max(0.0, min(float(progress), 1.0))


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _ensure_optional_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return _ensure_utc(value)


def _sqlite_path_from_url(url: str) -> Path | None:
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return None
    raw = url[len(prefix) :]
    if raw in ("", ":memory:"):
        return None
    return Path(raw)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
