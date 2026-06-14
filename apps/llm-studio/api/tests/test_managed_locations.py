from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pytest

from app.managed_locations import (
    MANAGED_LOCATION_PREFIX,
    encode_managed_location,
    resolve_managed_location,
)
from app.storage_safety import UnsafeManagedPathError
from app.tokenizer_models import JobStatus
from app.tokenizer_storage import StoredJob, StudioStore
from app.training_runs.schemas import TrainingJobState, TrainingJobStatus
from app.training_runs.store import StoredTrainingJob, TrainingStudioStore


def test_managed_location_codec_is_relocatable_and_preserves_external_values(
    tmp_path: Path,
) -> None:
    original_root = tmp_path / "original data"
    relocated_root = tmp_path / "relocated data"
    managed_path = original_root / "training" / "jobs" / "job-1" / "stats.jsonl"
    external_path = tmp_path / "external" / "stats.jsonl"

    encoded = encode_managed_location(str(managed_path), original_root)

    assert encoded == f"{MANAGED_LOCATION_PREFIX}training/jobs/job-1/stats.jsonl"
    assert resolve_managed_location(encoded, relocated_root) == str(
        (relocated_root / "training" / "jobs" / "job-1" / "stats.jsonl").resolve()
    )
    assert encode_managed_location(str(external_path), original_root) == str(external_path)
    assert encode_managed_location("__training_dataset__", original_root) == "__training_dataset__"


@pytest.mark.parametrize(
    "value",
    [
        f"{MANAGED_LOCATION_PREFIX}",
        f"{MANAGED_LOCATION_PREFIX}../outside",
        f"{MANAGED_LOCATION_PREFIX}/absolute",
        f"{MANAGED_LOCATION_PREFIX}nested\\windows-escape",
    ],
)
def test_managed_location_codec_rejects_unsafe_typed_values(
    value: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(UnsafeManagedPathError):
        resolve_managed_location(value, tmp_path / "data")


def test_tokenizer_store_persists_managed_paths_as_typed_locations(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    database = data_root / "db" / "tokenizer.db"
    artifact = data_root / "artifacts" / "tokenizers" / "tokenizer.json"
    upload = data_root / "uploads" / "train.txt"
    store = StudioStore(
        url=f"sqlite:///{database}",
        managed_root=data_root,
    )
    store.initialize()
    store.create_job(
        StoredJob(
            id="tokenizer-job",
            status=JobStatus.completed,
            stage="Completed",
            progress=1,
            created_at=datetime.now(timezone.utc),
            started_at=None,
            finished_at=None,
            tokenizer_config={},
            dataloader_config={},
            evaluation_thresholds=[],
            evaluation_text_path="__training_dataset__",
            artifact_file=artifact.name,
            artifact_path=str(artifact),
            stats=None,
            error=None,
        )
    )
    store.record_uploaded_file("train", upload.name, str(upload), 1)
    loaded = store.get_job("tokenizer-job")
    store.dispose()

    with sqlite3.connect(database) as connection:
        raw_artifact = connection.execute(
            "SELECT artifact_path FROM training_jobs WHERE id = ?",
            ("tokenizer-job",),
        ).fetchone()[0]
        raw_upload = connection.execute("SELECT file_path FROM uploaded_files").fetchone()[0]

    assert raw_artifact == f"{MANAGED_LOCATION_PREFIX}artifacts/tokenizers/tokenizer.json"
    assert raw_upload == f"{MANAGED_LOCATION_PREFIX}uploads/train.txt"
    assert loaded is not None
    assert loaded.artifact_path == str(artifact.resolve())
    assert loaded.evaluation_text_path == "__training_dataset__"


def test_training_store_encodes_only_local_managed_paths(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    database = data_root / "db" / "training.db"
    job_dir = data_root / "training" / "jobs" / "training-job"
    external_bundle = tmp_path / "external" / "bundle.zip"
    store = TrainingStudioStore(
        url=f"sqlite:///{database}",
        managed_root=data_root,
    )
    store.initialize()
    store.create_job(_training_job(job_dir, external_bundle))
    loaded = store.get_job("training-job")
    store.dispose()

    with sqlite3.connect(database) as connection:
        raw = connection.execute(
            """
            SELECT artifact_dir, stats_path, artifact_bundle_file, remote_workspace_path
            FROM llm_training_jobs WHERE id = ?
            """,
            ("training-job",),
        ).fetchone()

    assert raw == (
        f"{MANAGED_LOCATION_PREFIX}training/jobs/training-job",
        f"{MANAGED_LOCATION_PREFIX}training/jobs/training-job/stats.jsonl",
        str(external_bundle),
        "/workspace/llm-studio/jobs/training-job",
    )
    assert loaded is not None
    assert loaded.artifact_dir == str(job_dir.resolve())
    assert loaded.stats_path == str((job_dir / "stats.jsonl").resolve())
    assert loaded.remote_workspace_path == "/workspace/llm-studio/jobs/training-job"


def _training_job(job_dir: Path, external_bundle: Path) -> StoredTrainingJob:
    return StoredTrainingJob(
        id="training-job",
        name="Training job",
        status=TrainingJobStatus.completed,
        state=TrainingJobState.completed,
        stage="Completed",
        progress=1,
        created_at=datetime.now(timezone.utc),
        started_at=None,
        finished_at=None,
        project_id="project",
        project_name="Project",
        tokenizer_job_id="tokenizer",
        tokenizer_name="Tokenizer",
        model_config={},
        training_config={},
        dataloader_config={},
        resolved_runtime=None,
        memory_estimate=None,
        artifact_dir=str(job_dir),
        artifact_bundle_file=str(external_bundle),
        stats_path=str(job_dir / "stats.jsonl"),
        samples_path=str(job_dir / "samples.jsonl"),
        stdout_path=str(job_dir / "stdout.log"),
        stderr_path=str(job_dir / "stderr.log"),
        last_step=0,
        max_steps=0,
        latest_loss=None,
        latest_grad_norm=None,
        latest_lr=None,
        latest_tokens_per_sec=None,
        checkpoint_count=0,
        sample_count=0,
        error=None,
        process_id=None,
        output_size_bytes=0,
        remote_workspace_path="/workspace/llm-studio/jobs/training-job",
    )
