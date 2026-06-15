from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import io
from pathlib import Path
import tarfile

from app.training_runs.artifacts import prepare_training_job
from app.training_runs.executors.runpod.bundle import build_remote_bundle
from app.training_runs.preflight import ResolvedPreflightContext
from app.training_runs.responses import job_to_response
from app.training_runs.schemas import CreateTrainingJobRequest, TrainingAssetRef, TrainingExecutorKind
from app.training_runs.store import TrainingStudioStore


def make_context() -> ResolvedPreflightContext:
    return ResolvedPreflightContext(
        valid=True,
        model_project=TrainingAssetRef(id="project123", name="Project", artifact_file="model_config.json", status="READY"),
        tokenizer=TrainingAssetRef(id="tok123456", name="Tokenizer", artifact_file="tokenizer.json", status="completed"),
        model_config={"context_length": 8, "vocab_size": 16},
        normalized_training_config={"max_steps": 10},
        normalized_dataloader_config={"datasets": []},
        warnings=[],
        errors=[],
        recommended_fixes=[],
        compatibility=None,
        derived_runtime=None,
        memory_estimate=None,
        batch_and_lr_recommendation=None,
    )


def test_prepare_training_job_writes_inputs_and_builds_stored_row(tmp_path: Path) -> None:
    tokenizer_source = tmp_path / "tokenizer.json"
    tokenizer_source.write_text("{}", encoding="utf-8")
    request = CreateTrainingJobRequest.model_validate(
        {
            "project_id": "project123",
            "tokenizer_job_id": "tok123456",
            "training_config": {},
            "dataloader_config": {},
            "execution_target": {"kind": TrainingExecutorKind.local.value},
        }
    )

    prepared = prepare_training_job(
        request=request,
        context=make_context(),
        preflight_payload={"valid": True},
        tokenizer_source_path=tokenizer_source,
        training_jobs_root=tmp_path / "jobs",
        runpod_default_gpu_count=1,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        job_id="job123456",
    )

    assert prepared.stored.id == "job123456"
    assert prepared.stored.name == "Project x Tokenizer"
    assert prepared.stored.executor_kind == "local"
    assert prepared.bundle.manifest["format"] == "llm-studio-training-bundle-v1"
    assert prepared.bundle.manifest["job_id"] == "job123456"
    assert prepared.bundle.model_config_path.read_text(encoding="utf-8")
    assert prepared.bundle.tokenizer_path.read_text(encoding="utf-8") == "{}"
    assert prepared.bundle.resolved_preflight_path.read_text(encoding="utf-8")


def test_runpod_ui_key_remains_memory_only_and_is_excluded_from_remote_bundle(tmp_path: Path) -> None:
    runpod_key = "rpa_0123456789abcdef0123456789abcdef"
    tokenizer_source = tmp_path / "tokenizer.json"
    tokenizer_source.write_text("{}", encoding="utf-8")
    request = CreateTrainingJobRequest.model_validate(
        {
            "project_id": "project123",
            "tokenizer_job_id": "tok123456",
            "training_config": {},
            "dataloader_config": {},
            "execution_target": {
                "kind": TrainingExecutorKind.runpod_pod.value,
                "api_key": runpod_key,
            },
        }
    )

    prepared = prepare_training_job(
        request=request,
        context=make_context(),
        preflight_payload={"valid": True},
        tokenizer_source_path=tokenizer_source,
        training_jobs_root=tmp_path / "jobs",
        runpod_default_gpu_count=1,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        job_id="job123456",
    )

    assert prepared.bundle.manifest["execution_target"]["api_key"] == runpod_key
    assert runpod_key not in repr(asdict(prepared.stored))
    for path in prepared.job_dir.rglob("*"):
        if path.is_file():
            assert runpod_key.encode() not in path.read_bytes()

    database_path = tmp_path / "training.db"
    store = TrainingStudioStore(url=f"sqlite:///{database_path}")
    store.initialize()
    store.create_job(prepared.stored)
    store.dispose()
    assert runpod_key.encode() not in database_path.read_bytes()

    result = build_remote_bundle(prepared.bundle)
    if result.content_type == "application/zstd":
        import zstandard as zstd

        archive_bytes = zstd.ZstdDecompressor().decompress(result.path.read_bytes())
        archive = tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:")
    else:
        archive = tarfile.open(result.path, mode="r:gz")
    with archive:
        for member in archive.getmembers():
            if member.isfile():
                extracted = archive.extractfile(member)
                assert extracted is not None
                assert runpod_key.encode() not in extracted.read()


def test_hf_dataset_credentials_are_execution_only_and_excluded_from_durable_artifacts(
    tmp_path: Path,
) -> None:
    hf_token = "hf_0123456789abcdef0123456789abcdef"
    tokenizer_source = tmp_path / "tokenizer.json"
    tokenizer_source.write_text("{}", encoding="utf-8")
    context = make_context()
    context.normalized_dataloader_config = {
        "datasets": [{"name": "private-dataset", "hf_token": hf_token}]
    }
    request = CreateTrainingJobRequest.model_validate(
        {
            "project_id": "project123",
            "tokenizer_job_id": "tok123456",
            "training_config": {},
            "dataloader_config": {
                "datasets": [{"name": "private-dataset", "hf_token": hf_token}]
            },
            "hf_token": "hf_fallback_should_not_override_embedded",
            "execution_target": {"kind": TrainingExecutorKind.local.value},
        }
    )

    prepared = prepare_training_job(
        request=request,
        context=context,
        preflight_payload={
            "valid": True,
            "normalized_dataloader_config": context.normalized_dataloader_config,
        },
        tokenizer_source_path=tokenizer_source,
        training_jobs_root=tmp_path / "jobs",
        runpod_default_gpu_count=1,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        job_id="job123456",
    )

    assert prepared.bundle.manifest["dataset_hf_tokens"] == [hf_token]
    assert hf_token not in repr(asdict(prepared.stored))
    for path in prepared.job_dir.rglob("*"):
        if path.is_file():
            assert hf_token.encode() not in path.read_bytes()

    database_path = tmp_path / "training.db"
    store = TrainingStudioStore(url=f"sqlite:///{database_path}")
    store.initialize()
    store.create_job(prepared.stored)
    store.dispose()
    assert hf_token.encode() not in database_path.read_bytes()

    result = build_remote_bundle(prepared.bundle)
    if result.content_type == "application/zstd":
        import zstandard as zstd

        archive_bytes = zstd.ZstdDecompressor().decompress(result.path.read_bytes())
        archive = tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:")
    else:
        archive = tarfile.open(result.path, mode="r:gz")
    with archive:
        for member in archive.getmembers():
            if member.isfile():
                extracted = archive.extractfile(member)
                assert extracted is not None
                assert hf_token.encode() not in extracted.read()


def test_job_to_response_maps_runtime_elapsed_fields(tmp_path: Path) -> None:
    tokenizer_source = tmp_path / "tokenizer.json"
    tokenizer_source.write_text("{}", encoding="utf-8")
    request = CreateTrainingJobRequest.model_validate(
        {
            "project_id": "project123",
            "tokenizer_job_id": "tok123456",
            "training_config": {},
            "dataloader_config": {},
            "execution_target": {"kind": TrainingExecutorKind.local.value},
        }
    )
    prepared = prepare_training_job(
        request=request,
        context=make_context(),
        preflight_payload={"valid": True},
        tokenizer_source_path=tokenizer_source,
        training_jobs_root=tmp_path / "jobs",
        runpod_default_gpu_count=1,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        job_id="job123456",
    )

    response = job_to_response(
        prepared.stored,
        runtime_state={"elapsed_seconds": "12.5", "eta_seconds": "-1"},
    )

    assert response.id == "job123456"
    assert response.elapsed_seconds == 12.5
    assert response.eta_seconds is None
