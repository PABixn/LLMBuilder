from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ....config import get_settings
from ....dataset_credentials import HF_DATASET_TOKENS_ENV, encode_dataset_hf_tokens
from ....logging_config import redact_secrets
from ...schemas import TrainingJobState, TrainingJobStatus
from ...store import StoredTrainingJob
from ..base import CleanupPolicy, ExecutionHandle, ExecutionSnapshot, TrainingJobBundle
from .agent_client import RemoteAgentClient
from .bundle import build_remote_bundle
from .cleanup import policy_payload, terminal_cleanup_policy
from .client import CreatePodRequest, RunPodClient, RunPodClientError
from .config import optional_str as _optional_str
from .config import resolve_runpod_target
from .lifecycle_log import (
    compact_log_detail,
    log_lifecycle,
    sanitize_log_fields,
    sanitize_log_value,
)
from .errors import RemoteAgentError
from .ports import build_agent_base_url, extract_port_mappings
from .state import coerce_state, coerce_status, remote_executor_status
from .sync import sync_final_outputs, sync_incremental_outputs, sync_small_outputs
from .tokens import RunPodTokenRegistry, hash_token

CURRENT_AGENT_PROTOCOL_VERSION = 1
REMOTE_BUNDLE_FORMAT_VERSION = "llm-studio-training-bundle-v1"


class RunPodPodExecutor:
    kind = "runpod_pod"

    def __init__(self) -> None:
        self._token_registry = RunPodTokenRegistry()
        self._agent_tokens = self._token_registry.agent_tokens
        self._api_keys = self._token_registry.api_keys

    def submit(self, job: StoredTrainingJob, bundle: TrainingJobBundle) -> ExecutionHandle:
        target = bundle.manifest.get("execution_target") if isinstance(bundle.manifest, dict) else {}
        if not isinstance(target, dict):
            target = {}
        settings = get_settings()
        resolved_target = resolve_runpod_target(target, settings)
        agent_token = self._token_registry.create_agent_token(job.id)
        self._token_registry.set_api_key(job.id, resolved_target.api_key)
        cleanup_policy = resolved_target.cleanup_policy
        log_lifecycle(
            job,
            "submit_start",
            "Starting RunPod training submission.",
            image=resolved_target.image_name,
            gpu_type_id=resolved_target.gpu_type_id,
            gpu_count=resolved_target.gpu_count,
            cloud_type=resolved_target.cloud_type,
            data_center_id=resolved_target.data_center_id,
            container_disk_gb=resolved_target.container_disk_gb,
            volume_size_gb=resolved_target.volume_size_gb,
            volume_mount_path=resolved_target.volume_mount_path,
            agent_port=resolved_target.agent_port,
            agent_port_protocol=resolved_target.agent_port_protocol,
            cleanup_policy=cleanup_policy,
            interruptible=resolved_target.interruptible,
            api_key_source=resolved_target.api_key_source,
        )

        client = RunPodClient(resolved_target.api_key)
        encoded_hf_tokens = encode_dataset_hf_tokens(
            bundle.manifest.get("dataset_hf_tokens", [])
            if isinstance(bundle.manifest.get("dataset_hf_tokens"), list)
            else []
        )
        pod_id = ""
        agent_base_url = ""
        started_monotonic = time.monotonic()
        try:
            log_lifecycle(job, "create_pod_start", "Creating RunPod pod.")
            pod_request = CreatePodRequest(
                name=f"llm-studio-{job.id[:12]}",
                image_name=resolved_target.image_name,
                gpu_type_id=resolved_target.gpu_type_id,
                gpu_count=resolved_target.gpu_count,
                cloud_type=resolved_target.cloud_type,
                data_center_id=resolved_target.data_center_id,
                container_disk_gb=resolved_target.container_disk_gb,
                volume_gb=resolved_target.volume_size_gb,
                volume_mount_path=resolved_target.volume_mount_path,
                ports=[f"{resolved_target.agent_port}/{resolved_target.agent_port_protocol}"],
                env={
                    "LLM_STUDIO_REMOTE_AGENT_TOKEN": agent_token,
                    "LLM_STUDIO_REMOTE_JOB_ID": job.id,
                    "LLM_STUDIO_REMOTE_WORKSPACE": f"{resolved_target.volume_mount_path.rstrip('/')}/llm-studio",
                    "HF_HOME": f"{resolved_target.volume_mount_path.rstrip('/')}/llm-studio/cache/huggingface",
                    "HF_DATASETS_CACHE": f"{resolved_target.volume_mount_path.rstrip('/')}/llm-studio/cache/huggingface/datasets",
                    "LLM_STUDIO_RUNPOD_AGENT_PORT": str(resolved_target.agent_port),
                    "PYTHONUNBUFFERED": "1",
                    **({HF_DATASET_TOKENS_ENV: encoded_hf_tokens} if encoded_hf_tokens is not None else {}),
                },
                interruptible=resolved_target.interruptible,
            )
            pod = self._create_pod_with_retries(client, pod_request, job=job)
            pod_id = str(pod.get("id") or pod.get("podId") or "")
            if not pod_id:
                raise RunPodClientError("RunPod did not return a pod id after creation.", payload=pod)
            log_lifecycle(
                job,
                "create_pod_done",
                "RunPod pod created.",
                pod_id=pod_id,
                pod_status=_optional_str(pod.get("status")) or _optional_str(pod.get("desiredStatus")),
                elapsed_seconds=round(time.monotonic() - started_monotonic, 3),
            )

            pod = self._wait_for_pod_ready(client, pod_id, job=job)
            agent_base_url = build_agent_base_url(pod, resolved_target.agent_port)
            log_lifecycle(
                job,
                "agent_url_resolved",
                "Resolved RunPod pod-agent URL.",
                pod_id=pod_id,
                agent_base_url=agent_base_url,
                port_mappings=extract_port_mappings(pod),
            )
            agent = RemoteAgentClient(agent_base_url, agent_token, job.id)
            self._wait_for_agent(agent, job=job)
            self._verify_agent_compatibility(agent, job=job)
            log_lifecycle(job, "bundle_build_start", "Building remote training bundle.")
            bundle_result = build_remote_bundle(bundle)
            bundle_size = bundle_result.path.stat().st_size if bundle_result.path.exists() else None
            log_lifecycle(
                job,
                "bundle_build_done",
                "Remote training bundle built.",
                bundle_path=str(bundle_result.path),
                bundle_size_bytes=bundle_size,
                content_type=bundle_result.content_type,
                file_count=len(bundle_result.manifest.get("files", []))
                if isinstance(bundle_result.manifest.get("files"), list)
                else None,
            )
            log_lifecycle(job, "bundle_upload_start", "Uploading remote training bundle.", bundle_size_bytes=bundle_size)
            try:
                agent.upload_bundle(bundle_result.path, content_type=bundle_result.content_type)
            except RemoteAgentError as exc:
                raise _bundle_upload_error(exc, agent_base_url=agent_base_url) from exc
            log_lifecycle(job, "bundle_upload_done", "Remote training bundle uploaded.")
            log_lifecycle(job, "agent_start_start", "Requesting remote training process start.")
            start_payload = agent.start()
            log_lifecycle(
                job,
                "agent_start_done",
                "Remote training process started.",
                process_id=start_payload.get("process_id"),
                elapsed_seconds=round(time.monotonic() - started_monotonic, 3),
            )
        except Exception as exc:
            log_lifecycle(
                job,
                "submit_failed",
                "RunPod training submission failed.",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=12),
                pod_id=pod_id or None,
                agent_base_url=agent_base_url or None,
                elapsed_seconds=round(time.monotonic() - started_monotonic, 3),
            )
            if pod_id:
                self._cleanup_after_submit_failure(client, job, pod_id, cleanup_policy=cleanup_policy)
            self._token_registry.clear(job.id)
            raise

        now = _utc_now()
        return ExecutionHandle(
            executor_kind=self.kind,
            started_at=now,
            agent_base_url=agent_base_url,
            remote_ids={"runpod_pod_id": pod_id},
            updates={
                "executor_kind": self.kind,
                "executor_status": "running",
                "runpod_pod_id": pod_id,
                "runpod_pod_name": _optional_str(pod.get("name")) or f"llm-studio-{job.id[:12]}",
                "runpod_data_center_id": resolved_target.data_center_id or _optional_str(pod.get("dataCenterId")),
                "runpod_gpu_type_id": resolved_target.gpu_type_id,
                "runpod_gpu_count": resolved_target.gpu_count,
                "runpod_cloud_type": resolved_target.cloud_type,
                "runpod_interruptible": resolved_target.interruptible,
                "runpod_public_ip": _optional_str(pod.get("publicIp")),
                "runpod_port_mappings": extract_port_mappings(pod),
                "runpod_agent_base_url": agent_base_url,
                "runpod_agent_token_hash": hash_token(agent_token),
                "runpod_last_heartbeat_at": now,
                "runpod_last_sync_at": now,
                "runpod_cleanup_policy": cleanup_policy,
                "remote_workspace_path": f"{resolved_target.volume_mount_path.rstrip('/')}/llm-studio/jobs/{job.id}",
            },
        )

    def refresh(self, job: StoredTrainingJob) -> ExecutionSnapshot:
        if job.status in {TrainingJobStatus.completed, TrainingJobStatus.failed, TrainingJobStatus.cancelled}:
            return self._refresh_terminal_job(job)
        if not job.runpod_agent_base_url:
            log_lifecycle(
                job,
                "refresh_waiting_for_agent_url",
                "RunPod refresh is waiting for the agent URL.",
                throttle_seconds=30,
            )
            return ExecutionSnapshot(
                updates={"executor_status": "provisioning"},
            )
        token = self._unavailable_token(job)
        if token is None:
            log_lifecycle(
                job,
                "refresh_missing_agent_token",
                "RunPod refresh cannot authenticate because the in-memory pod-agent token is unavailable.",
                throttle_seconds=30,
            )
            return ExecutionSnapshot(
                updates={
                    "executor_status": job.executor_status or "running",
                    "remote_error": self._token_registry.missing_token_error(),
                }
            )
        agent = RemoteAgentClient(job.runpod_agent_base_url, token, job.id)
        try:
            state = agent.runtime_state()
            sync_result = sync_incremental_outputs(agent, job)
        except RemoteAgentError as exc:
            log_lifecycle(
                job,
                "refresh_agent_error",
                "RunPod pod-agent refresh failed.",
                error=str(exc),
                throttle_seconds=30,
            )
            return ExecutionSnapshot(updates={"remote_error": redact_secrets(str(exc))})
        updates = {
            "runpod_last_heartbeat_at": _utc_now(),
            "runpod_last_sync_at": _utc_now(),
            "executor_status": remote_executor_status(state),
        }
        if sync_result.checkpoint_count is not None:
            updates["checkpoint_count"] = sync_result.checkpoint_count
        status = coerce_status(state.get("status"))
        snapshot = ExecutionSnapshot(
            status=status,
            state=coerce_state(state.get("state")),
            stage=state.get("stage") if isinstance(state.get("stage"), str) else None,
            progress=float(state["progress"]) if isinstance(state.get("progress"), (int, float)) else None,
            error=redact_secrets(state["error"]) if isinstance(state.get("error"), str) else None,
            updates=updates,
        )
        if status in {TrainingJobStatus.completed, TrainingJobStatus.failed, TrainingJobStatus.cancelled}:
            snapshot.finished_at = _parse_datetime(state.get("finished_at")) or _utc_now()
            policy = terminal_cleanup_policy(job, status)
            if policy.pod == "delete_after_sync":
                try:
                    final_sync_result = sync_final_outputs(agent, job)
                    if final_sync_result.checkpoint_count is not None:
                        snapshot.updates["checkpoint_count"] = final_sync_result.checkpoint_count
                    snapshot.updates["runpod_last_sync_at"] = _utc_now()
                except RemoteAgentError as exc:
                    log_lifecycle(
                        job,
                        "cleanup_waiting_for_final_sync",
                        "RunPod cleanup is waiting for final artifact sync to complete.",
                        error=str(exc),
                    )
                    snapshot.updates["executor_status"] = "syncing"
                    snapshot.updates["remote_error"] = redact_secrets(
                        f"Training finished, but final artifact sync is incomplete: {exc}"
                    )
                    return snapshot
            try:
                log_lifecycle(job, "cleanup_start", "Applying RunPod cleanup policy.", policy=policy_payload(policy))
                self.cleanup(job, policy)
                log_lifecycle(job, "cleanup_done", "RunPod cleanup policy applied.", policy=policy_payload(policy))
                snapshot.updates["executor_status"] = status.value
                self._token_registry.clear(job.id)
            except Exception as exc:
                log_lifecycle(
                    job,
                    "cleanup_failed",
                    "Training finished, but RunPod cleanup failed.",
                    error=f"{type(exc).__name__}: {exc}",
                )
                snapshot.updates["remote_error"] = redact_secrets(
                    f"Training finished, but cleanup failed: {exc}"
                )
        return snapshot

    def _refresh_terminal_job(self, job: StoredTrainingJob) -> ExecutionSnapshot:
        terminal_status = job.status.value
        if job.executor_status in {terminal_status, "cleaned_up"}:
            return ExecutionSnapshot()

        updates = {"executor_status": terminal_status}
        if not job.runpod_pod_id:
            self._token_registry.clear(job.id)
            return ExecutionSnapshot(updates=updates)

        try:
            policy = terminal_cleanup_policy(job, job.status)
            if policy.pod == "delete_after_sync" and not (Path(job.artifact_dir) / "artifact_manifest.json").exists():
                token = self._unavailable_token(job)
                if token is None or not job.runpod_agent_base_url:
                    updates["executor_status"] = "syncing"
                    updates["remote_error"] = (
                        "Training finished, but final artifact sync is incomplete and the pod-agent token is unavailable."
                    )
                    return ExecutionSnapshot(updates=updates)
                try:
                    final_sync_result = sync_final_outputs(
                        RemoteAgentClient(job.runpod_agent_base_url, token, job.id),
                        job,
                    )
                    if final_sync_result.checkpoint_count is not None:
                        updates["checkpoint_count"] = final_sync_result.checkpoint_count
                    updates["runpod_last_sync_at"] = _utc_now()
                except RemoteAgentError as exc:
                    updates["executor_status"] = "syncing"
                    updates["remote_error"] = redact_secrets(
                        f"Training finished, but final artifact sync is incomplete: {exc}"
                    )
                    return ExecutionSnapshot(updates=updates)
            log_lifecycle(
                job,
                "cleanup_start",
                "Applying RunPod cleanup policy for terminal job.",
                policy=policy_payload(policy),
            )
            self.cleanup(job, policy)
            log_lifecycle(
                job,
                "cleanup_done",
                "RunPod cleanup policy applied for terminal job.",
                policy=policy_payload(policy),
            )
            self._token_registry.clear(job.id)
        except Exception as exc:
            log_lifecycle(
                job,
                "cleanup_failed",
                "Terminal RunPod job cleanup failed.",
                error=f"{type(exc).__name__}: {exc}",
            )
            updates["remote_error"] = redact_secrets(f"Training finished, but cleanup failed: {exc}")
        return ExecutionSnapshot(updates=updates)

    def stop(self, job: StoredTrainingJob) -> ExecutionSnapshot:
        token = self._unavailable_token(job)
        if token is not None and job.runpod_agent_base_url:
            try:
                log_lifecycle(job, "stop_agent_cancel_start", "Sending cancel request to RunPod pod agent.")
                RemoteAgentClient(job.runpod_agent_base_url, token, job.id).cancel()
                log_lifecycle(job, "stop_agent_cancel_done", "RunPod pod-agent cancel request returned.")
            except RemoteAgentError as exc:
                log_lifecycle(job, "stop_agent_cancel_failed", "RunPod pod-agent cancel request failed.", error=str(exc))
        if job.runpod_pod_id:
            try:
                log_lifecycle(job, "stop_pod_start", "Stopping RunPod pod.", pod_id=job.runpod_pod_id)
                self._client_for_job(job).stop_pod(job.runpod_pod_id)
                log_lifecycle(job, "stop_pod_done", "RunPod pod stop request returned.", pod_id=job.runpod_pod_id)
            except Exception as exc:
                log_lifecycle(
                    job,
                    "stop_pod_failed",
                    "RunPod pod stop request failed.",
                    pod_id=job.runpod_pod_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
        return ExecutionSnapshot(
            status=TrainingJobStatus.cancelled,
            state=TrainingJobState.cancelled,
            stage="Cancelled",
            progress=1.0,
            error="Training was cancelled by the user.",
            finished_at=_utc_now(),
            updates={"executor_status": "cancelled"},
        )

    def cleanup(self, job: StoredTrainingJob, policy: CleanupPolicy) -> None:
        if not job.runpod_pod_id:
            return
        client = self._client_for_job(job)
        if policy.pod == "delete_after_sync":
            log_lifecycle(job, "cleanup_delete_pod_start", "Deleting RunPod pod.", pod_id=job.runpod_pod_id)
            client.delete_pod(job.runpod_pod_id)
            log_lifecycle(job, "cleanup_delete_pod_done", "RunPod pod delete request returned.", pod_id=job.runpod_pod_id)
        elif policy.pod == "stop_after_sync":
            log_lifecycle(job, "cleanup_stop_pod_start", "Stopping RunPod pod.", pod_id=job.runpod_pod_id)
            client.stop_pod(job.runpod_pod_id)
            log_lifecycle(job, "cleanup_stop_pod_done", "RunPod pod stop request returned.", pod_id=job.runpod_pod_id)

    def _cleanup_after_submit_failure(
        self,
        client: RunPodClient,
        job: StoredTrainingJob,
        pod_id: str,
        *,
        cleanup_policy: dict[str, Any],
    ) -> None:
        pod_policy = str(cleanup_policy.get("pod") or "delete_after_sync")
        if pod_policy == "keep":
            log_lifecycle(
                job,
                "cleanup_after_failure_keep",
                "Keeping failed RunPod pod for inspection because cleanup policy is keep.",
                pod_id=pod_id,
            )
            return
        try:
            log_lifecycle(
                job,
                "cleanup_after_failure_stop_start",
                "Stopping failed RunPod pod for inspection.",
                pod_id=pod_id,
                cleanup_policy=cleanup_policy,
            )
            client.stop_pod(pod_id)
            log_lifecycle(job, "cleanup_after_failure_stop_done", "Stopped failed RunPod pod.", pod_id=pod_id)
        except Exception as stop_exc:
            log_lifecycle(
                job,
                "cleanup_after_failure_stop_failed",
                "Failed to stop RunPod pod after launch failure; pod was not deleted.",
                pod_id=pod_id,
                error=f"{type(stop_exc).__name__}: {stop_exc}",
            )

    def _client_for_job(self, job: StoredTrainingJob) -> RunPodClient:
        api_key = self._token_registry.api_key(job.id) or get_settings().runpod_api_key
        if not api_key:
            raise ValueError(
                "RunPod API key is required for cleanup. Keep the API process running after launch, "
                "or set LLM_STUDIO_RUNPOD_API_KEY so cleanup can recover after restart."
            )
        return RunPodClient(api_key)

    def _create_pod_with_retries(
        self,
        client: RunPodClient,
        request: CreatePodRequest,
        *,
        job: StoredTrainingJob,
    ) -> dict[str, Any]:
        max_attempts = 3
        last_error: RunPodClientError | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return client.create_pod(request)
            except RunPodClientError as exc:
                last_error = exc
                if not _is_retryable_create_pod_error(exc) or attempt >= max_attempts:
                    log_lifecycle(
                        job,
                        "create_pod_failed",
                        "RunPod pod creation failed.",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        status_code=exc.status_code,
                        error=str(exc),
                        payload=exc.payload,
                    )
                    raise
                delay_seconds = 3 * attempt
                log_lifecycle(
                    job,
                    "create_pod_retry",
                    "RunPod pod creation failed with a transient capacity error; retrying.",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay_seconds=delay_seconds,
                    error=str(exc),
                )
                time.sleep(delay_seconds)
        assert last_error is not None
        raise last_error

    def _wait_for_pod_ready(self, client: RunPodClient, pod_id: str, *, job: StoredTrainingJob) -> dict[str, Any]:
        deadline = time.monotonic() + 600
        last_pod: dict[str, Any] = {}
        last_log_at = 0.0
        started = time.monotonic()
        while time.monotonic() < deadline:
            last_pod = client.get_pod(pod_id)
            agent_url = build_agent_base_url(last_pod, get_settings().runpod_agent_port)
            now = time.monotonic()
            if now - last_log_at >= 15 or agent_url:
                last_log_at = now
                log_lifecycle(
                    job,
                    "pod_ready_poll",
                    "Polling RunPod pod for exposed agent port.",
                    pod_id=pod_id,
                    elapsed_seconds=round(now - started, 3),
                    pod_status=_optional_str(last_pod.get("status")) or _optional_str(last_pod.get("desiredStatus")),
                    public_ip=_optional_str(last_pod.get("publicIp")),
                    agent_url=agent_url or None,
                    port_mappings=extract_port_mappings(last_pod),
                )
            if agent_url:
                return last_pod
            time.sleep(5)
        log_lifecycle(
            job,
            "pod_ready_timeout",
            "RunPod pod did not expose the training agent port before timeout.",
            pod_id=pod_id,
            elapsed_seconds=round(time.monotonic() - started, 3),
            pod_status=_optional_str(last_pod.get("status")) or _optional_str(last_pod.get("desiredStatus")),
            public_ip=_optional_str(last_pod.get("publicIp")),
            port_mappings=extract_port_mappings(last_pod),
        )
        raise TimeoutError(f"RunPod pod {pod_id} did not expose the training agent port before timeout.")

    def _wait_for_agent(self, agent: RemoteAgentClient, *, job: StoredTrainingJob) -> None:
        deadline = time.monotonic() + 300
        last_error: Exception | None = None
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                agent.health()
                log_lifecycle(job, "agent_health_ok", "RunPod pod agent is reachable.", attempt=attempt)
                return
            except RemoteAgentError as exc:
                last_error = exc
                log_lifecycle(
                    job,
                    "agent_health_retry",
                    "RunPod pod agent is not reachable yet.",
                    attempt=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                    retryable=exc.retryable,
                )
                if not exc.retryable:
                    raise RuntimeError(f"RunPod pod agent health check failed permanently: {exc}") from exc
                time.sleep(3)
            except Exception as exc:
                last_error = exc
                log_lifecycle(
                    job,
                    "agent_health_retry",
                    "RunPod pod agent is not reachable yet.",
                    attempt=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                )
                time.sleep(3)
        raise TimeoutError(f"RunPod pod booted, but the training agent did not become healthy: {last_error}")

    def _verify_agent_compatibility(self, agent: RemoteAgentClient, *, job: StoredTrainingJob | None = None) -> None:
        try:
            system = agent.system()
        except RemoteAgentError as exc:
            if _system_check_can_be_skipped(exc):
                if job is not None:
                    log_lifecycle(
                        job,
                        "agent_system_unavailable",
                        "RunPod pod-agent system check is unavailable; continuing with legacy compatibility path.",
                        error=str(exc),
                        status_code=exc.status_code,
                    )
                return
            raise RuntimeError(f"RunPod pod agent is healthy, but the authenticated system check failed: {exc}") from exc

        runner = system.get("runner") if isinstance(system.get("runner"), dict) else None
        _verify_agent_protocol(system)
        if job is not None:
            log_lifecycle(
                job,
                "agent_system",
                "RunPod pod-agent system check returned.",
                agent_protocol_version=system.get("agent_protocol_version"),
                bundle_format_versions=system.get("bundle_format_versions"),
                supports_optional_files=system.get("supports_optional_files"),
                supports_checkpoint_manifest=system.get("supports_checkpoint_manifest"),
                workspace=system.get("workspace"),
                cuda_visible_devices=system.get("cuda_visible_devices"),
                runner=runner,
            )
        if runner is None:
            raise RuntimeError(
                "RunPod training image is too old for this app: the pod agent did not report trainer "
                "compatibility. Set LLM_STUDIO_RUNPOD_TRAINING_IMAGE to "
                "ghcr.io/pabixn/llm-builder-training:latest or another image built from the current Dockerfile."
            )
        if runner.get("import_ok") is not True:
            detail = runner.get("error") if isinstance(runner.get("error"), str) else "unknown import error"
            raise RuntimeError(f"RunPod training image cannot import the training runner: {detail}")

    def _unavailable_token(self, job: StoredTrainingJob) -> str | None:
        return self._token_registry.agent_token(job.id)


def _system_check_can_be_skipped(exc: RemoteAgentError) -> bool:
    if exc.status_code == 404:
        return True
    if exc.status_code == 422 and _is_missing_system_query_job_id_error(exc.payload):
        return True
    if exc.status_code is not None and exc.status_code >= 500:
        return True
    return False


def _verify_agent_protocol(system: dict[str, Any]) -> None:
    version = system.get("agent_protocol_version")
    if version is None:
        return
    if not isinstance(version, int) or version < CURRENT_AGENT_PROTOCOL_VERSION:
        raise RuntimeError(
            "RunPod training image is too old for this app: the pod agent reported unsupported "
            f"protocol version {version!r}."
        )
    bundle_versions = system.get("bundle_format_versions")
    if not isinstance(bundle_versions, list) or REMOTE_BUNDLE_FORMAT_VERSION not in bundle_versions:
        raise RuntimeError(
            "RunPod training image is too old for this app: the pod agent does not support "
            f"{REMOTE_BUNDLE_FORMAT_VERSION} bundles."
        )
    if system.get("supports_optional_files") is not True:
        raise RuntimeError(
            "RunPod training image is too old for this app: the pod agent does not support optional file sync."
        )
    if system.get("supports_checkpoint_manifest") is not True:
        raise RuntimeError(
            "RunPod training image is too old for this app: the pod agent does not support checkpoint manifests."
        )


def _is_retryable_create_pod_error(exc: RunPodClientError) -> bool:
    message = str(exc).lower()
    if "no instances currently available" in message:
        return True
    if "insufficient capacity" in message:
        return True
    return exc.status_code is not None and exc.status_code in {500, 502, 503, 504}


def _is_missing_system_query_job_id_error(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    details = payload.get("detail")
    if not isinstance(details, list):
        return False
    for item in details:
        if not isinstance(item, dict):
            continue
        if item.get("loc") == ["query", "job_id"] and item.get("type") == "missing":
            return True
    return False


def _bundle_upload_error(exc: RemoteAgentError, *, agent_base_url: str) -> RuntimeError | RemoteAgentError:
    if exc.status_code != 404:
        return exc
    if "proxy.runpod.net" in agent_base_url:
        return RuntimeError(
            "RunPod pod agent is healthy, but the HTTP proxy returned 404 for bundle upload. "
            "Use the default LLM_STUDIO_RUNPOD_AGENT_PORT_PROTOCOL=tcp so bundle uploads bypass "
            "RunPod's Cloudflare-backed proxy."
        )
    return RuntimeError(
        "RunPod pod agent is healthy, but the bundle upload endpoint returned 404. "
        "The pod is likely running a stale or wrong training image. Set "
        "LLM_STUDIO_RUNPOD_TRAINING_IMAGE to an image built from the current docker/training/Dockerfile."
    )


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
