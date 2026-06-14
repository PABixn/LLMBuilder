from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import JSON, Boolean, DateTime, Float, Index, Integer, String, Text, create_engine, event, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from ..config import get_settings, training_database_url
from ..dataset_credentials import strip_hf_tokens
from ..logging_config import redact_secrets
from ..managed_locations import encode_managed_location, resolve_managed_location
from ..storage_safety import database_unavailable_error, ensure_directory
from .migrations import apply_sqlite_migrations
from .schemas import TrainingJobState, TrainingJobStatus


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
    executor_kind: Mapped[str] = mapped_column(String(32), nullable=False, default="local")
    executor_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    runpod_pod_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    runpod_pod_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    runpod_network_volume_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    runpod_data_center_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    runpod_gpu_type_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    runpod_gpu_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    runpod_cloud_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    runpod_interruptible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    runpod_cost_per_hr: Mapped[float | None] = mapped_column(Float, nullable=True)
    runpod_public_ip: Mapped[str | None] = mapped_column(String(255), nullable=True)
    runpod_port_mappings: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    runpod_agent_base_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    runpod_agent_token_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    runpod_last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    runpod_last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    runpod_cleanup_policy: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    remote_workspace_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    remote_error: Mapped[str | None] = mapped_column(Text, nullable=True)


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
    executor_kind: str = "local"
    executor_status: str | None = None
    runpod_pod_id: str | None = None
    runpod_pod_name: str | None = None
    runpod_network_volume_id: str | None = None
    runpod_data_center_id: str | None = None
    runpod_gpu_type_id: str | None = None
    runpod_gpu_count: int = 1
    runpod_cloud_type: str | None = None
    runpod_interruptible: bool = False
    runpod_cost_per_hr: float | None = None
    runpod_public_ip: str | None = None
    runpod_port_mappings: dict[str, Any] | None = None
    runpod_agent_base_url: str | None = None
    runpod_agent_token_hash: str | None = None
    runpod_last_heartbeat_at: datetime | None = None
    runpod_last_sync_at: datetime | None = None
    runpod_cleanup_policy: dict[str, Any] | None = None
    remote_workspace_path: str | None = None
    remote_error: str | None = None


class TrainingStudioStore:
    def __init__(
        self,
        url: str | None = None,
        *,
        sqlite_timeout_seconds: float = 5.0,
        managed_root: Path | None = None,
    ) -> None:
        self._managed_root = (
            managed_root.expanduser().resolve(strict=False)
            if managed_root is not None
            else get_settings().data_dir.expanduser().resolve(strict=False)
            if url is None
            else None
        )
        self._url = url or training_database_url()
        self._database_path = _sqlite_path_from_url(self._url)
        self._sqlite_timeout_seconds = max(0.0, float(sqlite_timeout_seconds))
        self._engine = self._build_engine(self._url)
        self._session_factory = sessionmaker(
            bind=self._engine,
            autoflush=False,
            expire_on_commit=False,
        )

    def initialize(self) -> None:
        try:
            Base.metadata.create_all(self._engine)
            self._migrate_schema()
            self._sanitize_stored_secrets()
        except SQLAlchemyError as exc:
            raise database_unavailable_error("training", self._database_path) from exc

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
                    dataloader_config=strip_hf_tokens(job.dataloader_config),
                    resolved_runtime=None if job.resolved_runtime is None else dict(job.resolved_runtime),
                    memory_estimate=None if job.memory_estimate is None else dict(job.memory_estimate),
                    artifact_dir=self._encode_path(job.artifact_dir),
                    artifact_bundle_file=job.artifact_bundle_file,
                    stats_path=self._encode_path(job.stats_path),
                    samples_path=self._encode_path(job.samples_path),
                    stdout_path=self._encode_path(job.stdout_path),
                    stderr_path=self._encode_path(job.stderr_path),
                    last_step=max(0, int(job.last_step)),
                    max_steps=max(0, int(job.max_steps)),
                    latest_loss=job.latest_loss,
                    latest_grad_norm=job.latest_grad_norm,
                    latest_lr=job.latest_lr,
                    latest_tokens_per_sec=job.latest_tokens_per_sec,
                    checkpoint_count=max(0, int(job.checkpoint_count)),
                    sample_count=max(0, int(job.sample_count)),
                    error=None if job.error is None else redact_secrets(job.error),
                    process_id=job.process_id,
                    output_size_bytes=max(0, int(job.output_size_bytes)),
                    executor_kind=job.executor_kind,
                    executor_status=job.executor_status,
                    runpod_pod_id=job.runpod_pod_id,
                    runpod_pod_name=job.runpod_pod_name,
                    runpod_network_volume_id=job.runpod_network_volume_id,
                    runpod_data_center_id=job.runpod_data_center_id,
                    runpod_gpu_type_id=job.runpod_gpu_type_id,
                    runpod_gpu_count=max(1, int(job.runpod_gpu_count)),
                    runpod_cloud_type=job.runpod_cloud_type,
                    runpod_interruptible=bool(job.runpod_interruptible),
                    runpod_cost_per_hr=job.runpod_cost_per_hr,
                    runpod_public_ip=job.runpod_public_ip,
                    runpod_port_mappings=job.runpod_port_mappings,
                    runpod_agent_base_url=job.runpod_agent_base_url,
                    runpod_agent_token_hash=job.runpod_agent_token_hash,
                    runpod_last_heartbeat_at=_ensure_optional_utc(job.runpod_last_heartbeat_at),
                    runpod_last_sync_at=_ensure_optional_utc(job.runpod_last_sync_at),
                    runpod_cleanup_policy=job.runpod_cleanup_policy,
                    remote_workspace_path=job.remote_workspace_path,
                    remote_error=None if job.remote_error is None else redact_secrets(job.remote_error),
                )
            )

    def get_job(self, job_id: str) -> StoredTrainingJob | None:
        with self._session() as session:
            row = session.get(TrainingRunRow, job_id)
            if row is None:
                return None
            return _row_to_stored_job(row, self._managed_root)

    def list_jobs(self) -> list[StoredTrainingJob]:
        with self._session() as session:
            rows = session.scalars(
                select(TrainingRunRow).order_by(TrainingRunRow.created_at.desc())
            ).all()
            return [_row_to_stored_job(row, self._managed_root) for row in rows]

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
                    payload = dict(value)
                    setattr(row, field_name, strip_hf_tokens(payload) if field_name == "dataloader_config" else payload)
                elif field_name in {"resolved_runtime", "memory_estimate"}:
                    setattr(row, field_name, None if value is None else dict(value))
                elif field_name in {
                    "artifact_dir",
                    "stats_path",
                    "samples_path",
                    "stdout_path",
                    "stderr_path",
                }:
                    setattr(row, field_name, self._encode_path(value))
                elif field_name in {
                    "artifact_bundle_file",
                    "error",
                    "executor_kind",
                    "executor_status",
                    "runpod_pod_id",
                    "runpod_pod_name",
                    "runpod_network_volume_id",
                    "runpod_data_center_id",
                    "runpod_gpu_type_id",
                    "runpod_cloud_type",
                    "runpod_public_ip",
                    "runpod_agent_base_url",
                    "runpod_agent_token_hash",
                    "remote_workspace_path",
                    "remote_error",
                }:
                    text = None if value is None else str(value)
                    if field_name in {"error", "remote_error"} and text is not None:
                        text = redact_secrets(text)
                    setattr(row, field_name, text)
                elif field_name in {"last_step", "max_steps", "checkpoint_count", "sample_count", "process_id", "output_size_bytes", "runpod_gpu_count"}:
                    setattr(row, field_name, None if value is None else int(value))
                elif field_name in {"latest_loss", "latest_grad_norm", "latest_lr", "latest_tokens_per_sec", "runpod_cost_per_hr"}:
                    setattr(row, field_name, None if value is None else float(value))
                elif field_name in {"runpod_port_mappings", "runpod_cleanup_policy"}:
                    setattr(row, field_name, None if value is None else dict(value))
                elif field_name in {"runpod_interruptible"}:
                    setattr(row, field_name, bool(value))
                elif field_name in {"runpod_last_heartbeat_at", "runpod_last_sync_at"}:
                    setattr(row, field_name, _ensure_optional_utc(value))
                else:
                    raise ValueError(f"Unknown training job update field: {field_name}")

            session.flush()
            return _row_to_stored_job(row, self._managed_root)

    def _sanitize_stored_secrets(self) -> None:
        with self._session() as session:
            for row in session.scalars(select(TrainingRunRow)).all():
                row.dataloader_config = strip_hf_tokens(dict(row.dataloader_config))
                if row.error is not None:
                    row.error = redact_secrets(row.error)
                if row.remote_error is not None:
                    row.remote_error = redact_secrets(row.remote_error)

    def delete_job(self, job_id: str) -> StoredTrainingJob | None:
        with self._session() as session:
            row = session.get(TrainingRunRow, job_id)
            if row is None:
                return None
            stored = _row_to_stored_job(row, self._managed_root)
            session.delete(row)
            session.flush()
            return stored

    def mark_incomplete_jobs_failed(self, reason: str) -> int:
        """Mark every non-terminal job failed after an API restart.

        Kept for compatibility with the original startup behavior. New startup
        code should prefer the local/RunPod-specific methods below so remote
        jobs are not marked failed solely because process-local credentials were
        lost.
        """
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

    def mark_incomplete_local_jobs_failed(self, reason: str) -> int:
        """Mark only local subprocess jobs failed after an API restart."""
        now = _utc_now()
        with self._session() as session:
            result = session.execute(
                update(TrainingRunRow)
                .where(
                    TrainingRunRow.executor_kind == "local",
                    TrainingRunRow.status.in_(
                        [TrainingJobStatus.pending.value, TrainingJobStatus.running.value]
                    ),
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

    def list_incomplete_runpod_jobs(self) -> list[StoredTrainingJob]:
        with self._session() as session:
            rows = session.scalars(
                select(TrainingRunRow)
                .where(
                    TrainingRunRow.executor_kind == "runpod_pod",
                    TrainingRunRow.status.in_(
                        [TrainingJobStatus.pending.value, TrainingJobStatus.running.value]
                    ),
                )
                .order_by(TrainingRunRow.created_at.desc())
            ).all()
            return [_row_to_stored_job(row, self._managed_root) for row in rows]

    def mark_incomplete_runpod_jobs_recovery_limited(self, reason: str) -> int:
        """Record that RunPod jobs may still be running but cannot auto-refresh.

        Raw RunPod API keys and pod-agent tokens are intentionally process-local.
        After a restart, persisted RunPod rows remain running and get a
        `remote_error` that tells the user why automatic refresh/cleanup is
        limited until a new run is launched or manual RunPod cleanup is done.
        """
        with self._session() as session:
            result = session.execute(
                update(TrainingRunRow)
                .where(
                    TrainingRunRow.executor_kind == "runpod_pod",
                    TrainingRunRow.status.in_(
                        [TrainingJobStatus.pending.value, TrainingJobStatus.running.value]
                    ),
                )
                .values(
                    remote_error=reason,
                    executor_status="running",
                )
            )
            return int(result.rowcount or 0)

    @contextmanager
    def _session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as exc:
            try:
                session.rollback()
            except SQLAlchemyError:
                pass
            raise database_unavailable_error("training", self._database_path) from exc
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _build_engine(self, url: str) -> Engine:
        connect_args: dict[str, Any] = {}

        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            connect_args["timeout"] = self._sqlite_timeout_seconds
            db_file = _sqlite_path_from_url(url)
            if db_file is not None:
                ensure_directory(db_file.parent, operation="training database initialization")

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

    def _encode_path(self, value: object) -> str:
        text = str(value)
        if self._managed_root is None:
            return text
        return encode_managed_location(text, self._managed_root)

    def _migrate_schema(self) -> None:
        if not self._url.startswith("sqlite"):
            return
        apply_sqlite_migrations(self._engine)


def _row_to_stored_job(
    row: TrainingRunRow,
    managed_root: Path | None = None,
) -> StoredTrainingJob:
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
        model_config=_json_dict(row.model_config),
        training_config=_json_dict(row.training_config),
        dataloader_config=strip_hf_tokens(_json_dict(row.dataloader_config)),
        resolved_runtime=_json_optional_dict(row.resolved_runtime),
        memory_estimate=_json_optional_dict(row.memory_estimate),
        artifact_dir=_resolve_path(row.artifact_dir, managed_root),
        artifact_bundle_file=row.artifact_bundle_file,
        stats_path=_resolve_path(row.stats_path, managed_root),
        samples_path=_resolve_path(row.samples_path, managed_root),
        stdout_path=_resolve_path(row.stdout_path, managed_root),
        stderr_path=_resolve_path(row.stderr_path, managed_root),
        last_step=max(0, int(row.last_step)),
        max_steps=max(0, int(row.max_steps)),
        latest_loss=row.latest_loss,
        latest_grad_norm=row.latest_grad_norm,
        latest_lr=row.latest_lr,
        latest_tokens_per_sec=row.latest_tokens_per_sec,
        checkpoint_count=max(0, int(row.checkpoint_count)),
        sample_count=max(0, int(row.sample_count)),
        error=None if row.error is None else redact_secrets(row.error),
        process_id=row.process_id,
        output_size_bytes=max(0, int(row.output_size_bytes)),
        executor_kind=row.executor_kind or "local",
        executor_status=row.executor_status,
        runpod_pod_id=row.runpod_pod_id,
        runpod_pod_name=row.runpod_pod_name,
        runpod_network_volume_id=row.runpod_network_volume_id,
        runpod_data_center_id=row.runpod_data_center_id,
        runpod_gpu_type_id=row.runpod_gpu_type_id,
        runpod_gpu_count=max(1, int(row.runpod_gpu_count or 1)),
        runpod_cloud_type=row.runpod_cloud_type,
        runpod_interruptible=bool(row.runpod_interruptible),
        runpod_cost_per_hr=row.runpod_cost_per_hr,
        runpod_public_ip=row.runpod_public_ip,
        runpod_port_mappings=_json_optional_dict(row.runpod_port_mappings),
        runpod_agent_base_url=row.runpod_agent_base_url,
        runpod_agent_token_hash=row.runpod_agent_token_hash,
        runpod_last_heartbeat_at=_ensure_optional_utc(row.runpod_last_heartbeat_at),
        runpod_last_sync_at=_ensure_optional_utc(row.runpod_last_sync_at),
        runpod_cleanup_policy=_json_optional_dict(row.runpod_cleanup_policy),
        remote_workspace_path=row.remote_workspace_path,
        remote_error=None if row.remote_error is None else redact_secrets(row.remote_error),
    )


def _resolve_path(value: str, managed_root: Path | None) -> str:
    if managed_root is None:
        return value
    return resolve_managed_location(value, managed_root)


def _coerce_status(value: TrainingJobStatus | str) -> TrainingJobStatus:
    if isinstance(value, TrainingJobStatus):
        return value
    return TrainingJobStatus(str(value))


def _coerce_state(value: TrainingJobState | str) -> TrainingJobState:
    if isinstance(value, TrainingJobState):
        return value
    return TrainingJobState(str(value))


def _json_dict(value: Any) -> dict[str, Any]:
    parsed = _json_optional_dict(value)
    return parsed or {}


def _json_optional_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = __import__("json").loads(value)
        except Exception:
            return None
        return dict(parsed) if isinstance(parsed, dict) else None
    return None


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
