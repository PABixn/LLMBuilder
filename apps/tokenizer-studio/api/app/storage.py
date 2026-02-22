from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text, create_engine, event, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import database_url
from .models import JobStatus

UploadKind = Literal["train", "validation"]


class Base(DeclarativeBase):
    pass


class TrainingJobRow(Base):
    __tablename__ = "training_jobs"
    __table_args__ = (
        Index("ix_training_jobs_created_at", "created_at"),
        Index("ix_training_jobs_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(255), nullable=False)
    progress: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    tokenizer_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    dataloader_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    evaluation_thresholds: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    evaluation_text_path: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_file: Mapped[str | None] = mapped_column(String(512), nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    stats: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class UploadedFileRow(Base):
    __tablename__ = "uploaded_files"
    __table_args__ = (
        Index("ix_uploaded_files_created_at", "created_at"),
        Index("ix_uploaded_files_kind", "kind"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


@dataclass
class StoredJob:
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


class StudioStore:
    def __init__(self, url: str | None = None) -> None:
        self._url = url or database_url()
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

    def create_job(self, job: StoredJob) -> None:
        with self._session() as session:
            session.add(
                TrainingJobRow(
                    id=job.id,
                    status=job.status.value,
                    stage=job.stage,
                    progress=_clamp_progress(job.progress),
                    created_at=_ensure_utc(job.created_at),
                    started_at=_ensure_optional_utc(job.started_at),
                    finished_at=_ensure_optional_utc(job.finished_at),
                    tokenizer_config=job.tokenizer_config,
                    dataloader_config=job.dataloader_config,
                    evaluation_thresholds=[int(value) for value in job.evaluation_thresholds],
                    evaluation_text_path=job.evaluation_text_path,
                    artifact_file=job.artifact_file,
                    artifact_path=job.artifact_path,
                    stats=job.stats,
                    error=job.error,
                )
            )

    def get_job(self, job_id: str) -> StoredJob | None:
        with self._session() as session:
            row = session.get(TrainingJobRow, job_id)
            if row is None:
                return None
            return _row_to_stored_job(row)

    def list_jobs(self) -> list[StoredJob]:
        with self._session() as session:
            rows = session.scalars(
                select(TrainingJobRow).order_by(TrainingJobRow.created_at.desc())
            ).all()
            return [_row_to_stored_job(row) for row in rows]

    def update_job(self, job_id: str, **updates: Any) -> StoredJob | None:
        with self._session() as session:
            row = session.get(TrainingJobRow, job_id)
            if row is None:
                return None

            if "status" in updates:
                status = updates["status"]
                if isinstance(status, JobStatus):
                    row.status = status.value
                elif isinstance(status, str):
                    row.status = JobStatus(status).value
                else:
                    raise ValueError("status update must be JobStatus or string")

            if "stage" in updates:
                row.stage = str(updates["stage"])

            if "progress" in updates:
                row.progress = _clamp_progress(float(updates["progress"]))

            if "started_at" in updates:
                started_at = updates["started_at"]
                row.started_at = _ensure_optional_utc(started_at)

            if "finished_at" in updates:
                finished_at = updates["finished_at"]
                row.finished_at = _ensure_optional_utc(finished_at)

            if "artifact_file" in updates:
                row.artifact_file = (
                    None if updates["artifact_file"] is None else str(updates["artifact_file"])
                )

            if "artifact_path" in updates:
                row.artifact_path = (
                    None if updates["artifact_path"] is None else str(updates["artifact_path"])
                )

            if "stats" in updates:
                stats = updates["stats"]
                row.stats = None if stats is None else dict(stats)

            if "error" in updates:
                error = updates["error"]
                row.error = None if error is None else str(error)

            session.flush()
            return _row_to_stored_job(row)

    def mark_incomplete_jobs_failed(self, reason: str) -> int:
        now = _utc_now()
        with self._session() as session:
            result = session.execute(
                update(TrainingJobRow)
                .where(
                    TrainingJobRow.status.in_(
                        [JobStatus.pending.value, JobStatus.running.value]
                    )
                )
                .values(
                    status=JobStatus.failed.value,
                    stage="Interrupted by API restart",
                    progress=1.0,
                    finished_at=now,
                    error=reason,
                )
            )
            return int(result.rowcount or 0)

    def record_uploaded_file(
        self,
        kind: UploadKind,
        file_name: str,
        file_path: str,
        size_bytes: int,
    ) -> None:
        with self._session() as session:
            session.add(
                UploadedFileRow(
                    kind=kind,
                    file_name=file_name,
                    file_path=file_path,
                    size_bytes=max(0, int(size_bytes)),
                    created_at=_utc_now(),
                )
            )

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


def _row_to_stored_job(row: TrainingJobRow) -> StoredJob:
    return StoredJob(
        id=row.id,
        status=JobStatus(row.status),
        stage=row.stage,
        progress=_clamp_progress(row.progress),
        created_at=_ensure_utc(row.created_at),
        started_at=_ensure_optional_utc(row.started_at),
        finished_at=_ensure_optional_utc(row.finished_at),
        tokenizer_config=dict(row.tokenizer_config),
        dataloader_config=dict(row.dataloader_config),
        evaluation_thresholds=[int(value) for value in row.evaluation_thresholds],
        evaluation_text_path=row.evaluation_text_path,
        artifact_file=row.artifact_file,
        artifact_path=row.artifact_path,
        stats=dict(row.stats) if row.stats is not None else None,
        error=row.error,
    )


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
