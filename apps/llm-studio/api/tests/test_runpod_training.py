from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.training_executors.remote_sync import build_remote_bundle, rewrite_local_dataset_files
from app.training_executors.runpod_client import CreatePodRequest
from app.training_storage import TrainingStudioStore


def test_training_store_migrates_old_sqlite_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "training.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE llm_training_jobs (
            id VARCHAR(64) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            status VARCHAR(32) NOT NULL,
            state VARCHAR(64) NOT NULL,
            stage VARCHAR(255) NOT NULL,
            progress FLOAT NOT NULL,
            created_at DATETIME NOT NULL,
            started_at DATETIME,
            finished_at DATETIME,
            project_id VARCHAR(64) NOT NULL,
            project_name VARCHAR(255) NOT NULL,
            tokenizer_job_id VARCHAR(64) NOT NULL,
            tokenizer_name VARCHAR(255) NOT NULL,
            model_config JSON NOT NULL,
            training_config JSON NOT NULL,
            dataloader_config JSON NOT NULL,
            resolved_runtime JSON,
            memory_estimate JSON,
            artifact_dir TEXT NOT NULL,
            artifact_bundle_file VARCHAR(512),
            stats_path TEXT NOT NULL,
            samples_path TEXT NOT NULL,
            stdout_path TEXT NOT NULL,
            stderr_path TEXT NOT NULL,
            last_step INTEGER NOT NULL DEFAULT 0,
            max_steps INTEGER NOT NULL DEFAULT 0,
            latest_loss FLOAT,
            latest_grad_norm FLOAT,
            latest_lr FLOAT,
            latest_tokens_per_sec FLOAT,
            checkpoint_count INTEGER NOT NULL DEFAULT 0,
            sample_count INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            process_id INTEGER,
            output_size_bytes INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.execute(
        """
        INSERT INTO llm_training_jobs (
            id, name, status, state, stage, progress, created_at, project_id, project_name,
            tokenizer_job_id, tokenizer_name, model_config, training_config, dataloader_config,
            artifact_dir, stats_path, samples_path, stdout_path, stderr_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "job123456",
            "Old job",
            "completed",
            "completed",
            "Completed",
            1.0,
            datetime.now(timezone.utc).isoformat(),
            "project123",
            "Project",
            "tok123456",
            "Tokenizer",
            "{}",
            "{}",
            "{}",
            str(tmp_path),
            str(tmp_path / "stats.jsonl"),
            str(tmp_path / "samples.jsonl"),
            str(tmp_path / "stdout.log"),
            str(tmp_path / "stderr.log"),
        ),
    )
    connection.commit()
    connection.close()

    store = TrainingStudioStore(url=f"sqlite:///{db_path}")
    store.initialize()
    job = store.get_job("job123456")

    assert job is not None
    assert job.executor_kind == "local"
    assert job.runpod_gpu_count == 1
    assert job.runpod_pod_id is None


def test_runpod_create_pod_payload_uses_current_image_name_field() -> None:
    payload = CreatePodRequest(
        name="llm-studio-test",
        image_name="ghcr.io/example/llm-builder-training:latest",
        gpu_type_id="NVIDIA GeForce RTX 4090",
        gpu_count=1,
        cloud_type="SECURE",
        data_center_id="US-KS-2",
        container_disk_gb=50,
        volume_gb=100,
        volume_mount_path="/workspace",
        ports=["8021/http"],
        env={"LLM_STUDIO_REMOTE_JOB_ID": "job123"},
    ).to_payload()

    assert payload["imageName"] == "ghcr.io/example/llm-builder-training:latest"
    assert payload["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
    assert payload["ports"] == ["8021/http"]
    assert "image" not in payload


def test_rewrite_local_dataset_files_copies_and_rewrites_paths(tmp_path: Path) -> None:
    source = tmp_path / "data.txt"
    source.write_text("hello", encoding="utf-8")
    inputs_dir = tmp_path / "bundle" / "inputs"
    payload = {
        "datasets": [
            {
                "name": "local-text",
                "data_files": {"train": str(source)},
            }
        ]
    }

    rewritten, metadata = rewrite_local_dataset_files(
        payload,
        source_base=tmp_path,
        target_inputs_dir=inputs_dir,
    )

    train_path = rewritten["datasets"][0]["data_files"]["train"]
    assert train_path.startswith("inputs/local_files/000-local-text/")
    assert (tmp_path / "bundle" / train_path).exists()
    assert metadata[0]["original_path"] == str(source)
    assert metadata[0]["remote_path"] == train_path
