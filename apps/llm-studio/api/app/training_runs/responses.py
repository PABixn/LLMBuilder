from __future__ import annotations

import math
from typing import Any

from .schemas import TrainingJobResponse
from .store import StoredTrainingJob


def job_to_response(
    job: StoredTrainingJob,
    *,
    runtime_state: dict[str, Any] | None = None,
) -> TrainingJobResponse:
    return TrainingJobResponse(
        id=job.id,
        name=job.name,
        status=job.status,
        state=job.state,
        stage=job.stage,
        progress=job.progress,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        project_id=job.project_id,
        project_name=job.project_name,
        tokenizer_job_id=job.tokenizer_job_id,
        tokenizer_name=job.tokenizer_name,
        model_config=job.model_config,
        training_config=job.training_config,
        dataloader_config=job.dataloader_config,
        resolved_runtime=job.resolved_runtime,
        memory_estimate=job.memory_estimate,
        artifact_dir=job.artifact_dir,
        artifact_bundle_file=job.artifact_bundle_file,
        stats_path=job.stats_path,
        samples_path=job.samples_path,
        stdout_path=job.stdout_path,
        stderr_path=job.stderr_path,
        last_step=job.last_step,
        max_steps=job.max_steps,
        elapsed_seconds=optional_float_from_payload(runtime_state, "elapsed_seconds"),
        eta_seconds=optional_float_from_payload(runtime_state, "eta_seconds"),
        latest_loss=job.latest_loss,
        latest_grad_norm=job.latest_grad_norm,
        latest_lr=job.latest_lr,
        latest_tokens_per_sec=job.latest_tokens_per_sec,
        checkpoint_count=job.checkpoint_count,
        sample_count=job.sample_count,
        error=job.error,
        process_id=job.process_id,
        output_size_bytes=job.output_size_bytes,
        executor_kind=job.executor_kind,
        executor_status=job.executor_status,
        runpod_pod_id=job.runpod_pod_id,
        runpod_pod_name=job.runpod_pod_name,
        runpod_network_volume_id=job.runpod_network_volume_id,
        runpod_data_center_id=job.runpod_data_center_id,
        runpod_gpu_type_id=job.runpod_gpu_type_id,
        runpod_gpu_count=job.runpod_gpu_count,
        runpod_cloud_type=job.runpod_cloud_type,
        runpod_interruptible=job.runpod_interruptible,
        runpod_cost_per_hr=job.runpod_cost_per_hr,
        runpod_public_ip=job.runpod_public_ip,
        runpod_port_mappings=job.runpod_port_mappings,
        runpod_agent_base_url=job.runpod_agent_base_url,
        runpod_last_heartbeat_at=job.runpod_last_heartbeat_at,
        runpod_last_sync_at=job.runpod_last_sync_at,
        runpod_cleanup_policy=job.runpod_cleanup_policy,
        remote_workspace_path=job.remote_workspace_path,
        remote_error=job.remote_error,
    )


def optional_float_from_payload(payload: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed
