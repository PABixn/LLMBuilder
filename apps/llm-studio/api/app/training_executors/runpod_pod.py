from __future__ import annotations

import hashlib
import json
import secrets
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..training_models import TrainingJobState, TrainingJobStatus
from ..training_storage import StoredTrainingJob
from .base import CleanupPolicy, ExecutionHandle, ExecutionSnapshot, TrainingJobBundle
from .remote_sync import RemoteAgentClient, RemoteAgentError, build_remote_bundle
from .runpod_client import CreatePodRequest, RunPodClient, RunPodClientError

_LAST_LIFECYCLE_LOG_AT: dict[tuple[str, str, str], float] = {}


class RunPodPodExecutor:
    kind = "runpod_pod"

    def __init__(self) -> None:
        self._agent_tokens: dict[str, str] = {}
        self._api_keys: dict[str, str] = {}

    def submit(self, job: StoredTrainingJob, bundle: TrainingJobBundle) -> ExecutionHandle:
        target = bundle.manifest.get("execution_target") if isinstance(bundle.manifest, dict) else {}
        if not isinstance(target, dict):
            target = {}
        settings = get_settings()
        api_key = str(target.get("api_key") or settings.runpod_api_key or "").strip()
        if not api_key:
            raise ValueError("RunPod API key is required. Paste one in the UI or set LLM_STUDIO_RUNPOD_API_KEY.")

        agent_token = secrets.token_urlsafe(32)
        self._agent_tokens[job.id] = agent_token
        self._api_keys[job.id] = api_key
        gpu_type_id = str(target.get("gpu_type_id") or settings.runpod_default_gpu_type)
        gpu_count = int(target.get("gpu_count") or settings.runpod_default_gpu_count)
        cloud_type = str(target.get("cloud_type") or settings.runpod_default_cloud_type).upper()
        data_center_id = _optional_str(target.get("data_center_id") or settings.runpod_default_data_center_id)
        volume_size_gb = int(target.get("network_volume_size_gb") or settings.runpod_default_volume_size_gb)
        cleanup_policy = target.get("cleanup_policy") if isinstance(target.get("cleanup_policy"), dict) else {}
        log_lifecycle(
            job,
            "submit_start",
            "Starting RunPod training submission.",
            image=settings.runpod_training_image,
            gpu_type_id=gpu_type_id,
            gpu_count=gpu_count,
            cloud_type=cloud_type,
            data_center_id=data_center_id,
            container_disk_gb=settings.runpod_container_disk_gb,
            volume_size_gb=volume_size_gb,
            volume_mount_path=settings.runpod_volume_mount_path,
            agent_port=settings.runpod_agent_port,
            agent_port_protocol=settings.runpod_agent_port_protocol,
            cleanup_policy=cleanup_policy,
            interruptible=bool(target.get("interruptible", False)),
        )

        client = RunPodClient(api_key)
        pod_id = ""
        agent_base_url = ""
        started_monotonic = time.monotonic()
        try:
            log_lifecycle(job, "create_pod_start", "Creating RunPod pod.")
            pod = client.create_pod(
                CreatePodRequest(
                    name=f"llm-studio-{job.id[:12]}",
                    image_name=settings.runpod_training_image,
                    gpu_type_id=gpu_type_id,
                    gpu_count=gpu_count,
                    cloud_type=cloud_type,
                    data_center_id=data_center_id,
                    container_disk_gb=settings.runpod_container_disk_gb,
                    volume_gb=volume_size_gb,
                    volume_mount_path=settings.runpod_volume_mount_path,
                    ports=[f"{settings.runpod_agent_port}/{settings.runpod_agent_port_protocol}"],
                    env={
                        "LLM_STUDIO_REMOTE_AGENT_TOKEN": agent_token,
                        "LLM_STUDIO_REMOTE_JOB_ID": job.id,
                        "LLM_STUDIO_REMOTE_WORKSPACE": f"{settings.runpod_volume_mount_path.rstrip('/')}/llm-studio",
                        "HF_HOME": f"{settings.runpod_volume_mount_path.rstrip('/')}/llm-studio/cache/huggingface",
                        "HF_DATASETS_CACHE": f"{settings.runpod_volume_mount_path.rstrip('/')}/llm-studio/cache/huggingface/datasets",
                        "LLM_STUDIO_RUNPOD_AGENT_PORT": str(settings.runpod_agent_port),
                        "PYTHONUNBUFFERED": "1",
                    },
                    interruptible=bool(target.get("interruptible", False)),
                )
            )
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
            agent_base_url = build_agent_base_url(pod, settings.runpod_agent_port)
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
            self._agent_tokens.pop(job.id, None)
            self._api_keys.pop(job.id, None)
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
                "runpod_data_center_id": data_center_id or _optional_str(pod.get("dataCenterId")),
                "runpod_gpu_type_id": gpu_type_id,
                "runpod_gpu_count": gpu_count,
                "runpod_cloud_type": cloud_type,
                "runpod_interruptible": bool(target.get("interruptible", False)),
                "runpod_public_ip": _optional_str(pod.get("publicIp")),
                "runpod_port_mappings": extract_port_mappings(pod),
                "runpod_agent_base_url": agent_base_url,
                "runpod_agent_token_hash": hash_token(agent_token),
                "runpod_last_heartbeat_at": now,
                "runpod_last_sync_at": now,
                "runpod_cleanup_policy": cleanup_policy,
                "remote_workspace_path": f"{settings.runpod_volume_mount_path.rstrip('/')}/llm-studio/jobs/{job.id}",
            },
        )

    def refresh(self, job: StoredTrainingJob) -> ExecutionSnapshot:
        if job.status in {TrainingJobStatus.completed, TrainingJobStatus.failed, TrainingJobStatus.cancelled}:
            return ExecutionSnapshot()
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
                    "remote_error": "Pod-agent token is not recoverable after API restart; use remote reattach with a fresh token or stop the pod from RunPod.",
                }
            )
        agent = RemoteAgentClient(job.runpod_agent_base_url, token, job.id)
        try:
            state = agent.runtime_state()
            sync_small_outputs(agent, job)
        except RemoteAgentError as exc:
            log_lifecycle(
                job,
                "refresh_agent_error",
                "RunPod pod-agent refresh failed.",
                error=str(exc),
                throttle_seconds=30,
            )
            return ExecutionSnapshot(updates={"remote_error": str(exc)})
        updates = {
            "runpod_last_heartbeat_at": _utc_now(),
            "runpod_last_sync_at": _utc_now(),
            "executor_status": remote_executor_status(state),
        }
        status = _coerce_status(state.get("status"))
        snapshot = ExecutionSnapshot(
            status=status,
            state=_coerce_state(state.get("state")),
            stage=state.get("stage") if isinstance(state.get("stage"), str) else None,
            progress=float(state["progress"]) if isinstance(state.get("progress"), (int, float)) else None,
            error=state.get("error") if isinstance(state.get("error"), str) else None,
            updates=updates,
        )
        if status in {TrainingJobStatus.completed, TrainingJobStatus.failed, TrainingJobStatus.cancelled}:
            snapshot.finished_at = _parse_datetime(state.get("finished_at")) or _utc_now()
            policy = policy_from_job(job)
            try:
                log_lifecycle(job, "cleanup_start", "Applying RunPod cleanup policy.", policy=policy_payload(policy))
                self.cleanup(job, policy)
                log_lifecycle(job, "cleanup_done", "RunPod cleanup policy applied.", policy=policy_payload(policy))
                snapshot.updates["executor_status"] = status.value
                self._agent_tokens.pop(job.id, None)
                self._api_keys.pop(job.id, None)
            except Exception as exc:
                log_lifecycle(
                    job,
                    "cleanup_failed",
                    "Training finished, but RunPod cleanup failed.",
                    error=f"{type(exc).__name__}: {exc}",
                )
                snapshot.updates["remote_error"] = f"Training finished, but cleanup failed: {exc}"
        return snapshot

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
        api_key = self._api_keys.get(job.id) or get_settings().runpod_api_key
        if not api_key:
            raise ValueError(
                "RunPod API key is required for cleanup. Keep the API process running after launch, "
                "or set LLM_STUDIO_RUNPOD_API_KEY so cleanup can recover after restart."
            )
        return RunPodClient(api_key)

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
        if job is not None:
            log_lifecycle(
                job,
                "agent_system",
                "RunPod pod-agent system check returned.",
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
        return self._agent_tokens.get(job.id)


def _system_check_can_be_skipped(exc: RemoteAgentError) -> bool:
    if exc.status_code == 404:
        return True
    if exc.status_code == 422 and _is_missing_system_query_job_id_error(exc.payload):
        return True
    if exc.status_code is not None and exc.status_code >= 500:
        return True
    return False


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


def sync_small_outputs(agent: RemoteAgentClient, job: StoredTrainingJob) -> None:
    root = __import__("pathlib").Path(job.artifact_dir)
    for remote_kind, local_path in (
        ("metrics", root / "stats.jsonl"),
        ("samples", root / "samples.jsonl"),
        ("logs/stdout", root / "stdout.log"),
        ("logs/stderr", root / "stderr.log"),
        ("logs/startup", root / "runpod_startup.log"),
        ("logs/agent", root / "runpod_agent.log"),
        ("logs/runner", root / "runpod_runner.log"),
    ):
        before = local_path.stat().st_size if local_path.exists() else 0
        try:
            agent.download_append_file(remote_kind, local_path)
        except RemoteAgentError as exc:
            log_lifecycle(
                job,
                "sync_output_unavailable",
                "Remote append-only output is not available yet.",
                remote_kind=remote_kind,
                error=str(exc),
                throttle_seconds=60,
                throttle_key=remote_kind,
            )
            continue
        after = local_path.stat().st_size if local_path.exists() else 0
        if after > before:
            log_lifecycle(
                job,
                "sync_append",
                "Synced appended remote output.",
                remote_kind=remote_kind,
                local_path=str(local_path),
                bytes_added=after - before,
                size_bytes=after,
            )
    for remote_name in ("runtime_state.json", "metadata.json", "artifact_manifest.json", "training_data_preview.json"):
        try:
            agent.download_file(remote_name, root / remote_name, optional=True)
        except RemoteAgentError as exc:
            log_lifecycle(
                job,
                "sync_file_unavailable",
                "Remote output file is not available yet.",
                remote_name=remote_name,
                error=str(exc),
                throttle_seconds=60,
                throttle_key=remote_name,
            )
            continue


def build_agent_base_url(pod: dict[str, Any], port: int) -> str:
    pod_id = _optional_str(pod.get("id")) or _optional_str(pod.get("podId"))
    mappings = extract_port_mappings(pod)
    public_ip = _optional_str(pod.get("publicIp"))
    for value in mappings.values():
        if isinstance(value, dict):
            url = value.get("url") or value.get("uri")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return url.rstrip("/")
    for key, value in mappings.items():
        if isinstance(value, dict):
            if pod_id and is_http_port_mapping(value, port):
                return f"https://{pod_id}-{port}.proxy.runpod.net"
            if not is_tcp_port_mapping(value):
                continue
            host = tcp_mapping_host(value, public_ip)
            mapped_port = value.get("publicPort") or value.get("externalPort") or value.get("port")
            if host and mapped_port:
                return f"http://{host}:{mapped_port}"
        elif public_ip and str(key) == str(port) and isinstance(value, (int, str)):
            try:
                mapped_port = int(value)
            except (TypeError, ValueError):
                continue
            return f"http://{public_ip}:{mapped_port}"
        elif isinstance(value, str) and value.startswith(("http://", "https://")):
            return value.rstrip("/")
    if pod_id and pod_exposes_http_port(pod, port):
        return f"https://{pod_id}-{port}.proxy.runpod.net"
    return ""


def extract_port_mappings(pod: dict[str, Any]) -> dict[str, Any]:
    value = pod.get("portMappings")
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {str(index): item for index, item in enumerate(value)}
    runtime = pod.get("runtime")
    if isinstance(runtime, dict):
        value = runtime.get("ports")
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {str(index): item for index, item in enumerate(value)}
    value = pod.get("ports")
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {str(index): item for index, item in enumerate(value)}
    return {}


def pod_exposes_http_port(pod: dict[str, Any], port: int) -> bool:
    expected = f"{port}/http"
    ports = pod.get("ports")
    if isinstance(ports, list) and any(item == expected for item in ports):
        return True
    mappings = extract_port_mappings(pod)
    return any(is_http_port_mapping(value, port) for value in mappings.values())


def is_http_port_mapping(value: Any, port: int) -> bool:
    if not isinstance(value, dict):
        return False
    mapping_type = str(value.get("type") or value.get("protocol") or "").lower()
    private_port = value.get("privatePort") or value.get("containerPort") or value.get("port")
    try:
        private_port_number = int(private_port)
    except (TypeError, ValueError):
        private_port_number = None
    return mapping_type == "http" and private_port_number == port


def is_tcp_port_mapping(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    mapping_type = str(value.get("type") or value.get("protocol") or "").lower()
    return mapping_type in {"", "tcp"}


def tcp_mapping_host(value: dict[str, Any], public_ip: str | None) -> Any:
    host = value.get("host") or value.get("ip")
    if value.get("isIpPublic") is False and public_ip:
        return public_ip
    return host or public_ip


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def policy_from_job(job: StoredTrainingJob) -> CleanupPolicy:
    payload = job.runpod_cleanup_policy or {}
    return CleanupPolicy(
        pod=str(payload.get("pod") or "delete_after_sync"),
        network_volume=str(payload.get("network_volume") or "keep"),
    )


def remote_executor_status(state: dict[str, Any]) -> str:
    status = state.get("status")
    if status in {"completed", "failed", "cancelled"}:
        return str(status)
    stage = str(state.get("stage") or "").lower()
    if "sync" in stage:
        return "syncing"
    if "upload" in stage:
        return "uploading"
    return "running"


def _coerce_status(value: Any) -> TrainingJobStatus | None:
    try:
        return TrainingJobStatus(str(value))
    except Exception:
        return None


def _coerce_state(value: Any) -> TrainingJobState | None:
    try:
        return TrainingJobState(str(value))
    except Exception:
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def policy_payload(policy: CleanupPolicy) -> dict[str, str]:
    return {"pod": policy.pod, "network_volume": policy.network_volume}


def log_lifecycle(
    job: StoredTrainingJob,
    event: str,
    message: str,
    *,
    throttle_seconds: float | None = None,
    throttle_key: str | None = None,
    **fields: Any,
) -> None:
    if throttle_seconds is not None:
        key = (job.id, event, throttle_key or "")
        now_monotonic = time.monotonic()
        previous = _LAST_LIFECYCLE_LOG_AT.get(key)
        if previous is not None and now_monotonic - previous < throttle_seconds:
            return
        _LAST_LIFECYCLE_LOG_AT[key] = now_monotonic

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": job.id,
        "executor": RunPodPodExecutor.kind,
        "event": event,
        "message": message,
        **sanitize_log_fields(fields),
    }
    line = json.dumps(payload, ensure_ascii=True, default=str, sort_keys=True)

    artifact_dir = Path(job.artifact_dir)
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with (artifact_dir / "runpod_lifecycle.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
        with (artifact_dir / "runpod_lifecycle.log").open("a", encoding="utf-8") as handle:
            handle.write(f"[runpod:{event}] {message}")
            detail = compact_log_detail(payload)
            if detail:
                handle.write(f" | {detail}")
            handle.write("\n")
    except Exception:
        pass

    print(f"[llm-studio-runpod] {line}", flush=True)


def sanitize_log_fields(fields: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in fields.items():
        key_lower = key.lower()
        if "token" in key_lower or "api_key" in key_lower or "authorization" in key_lower:
            sanitized[key] = "[redacted]"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_fields(value)
        elif isinstance(value, list):
            sanitized[key] = [sanitize_log_value(item) for item in value]
        else:
            sanitized[key] = sanitize_log_value(value)
    return sanitized


def sanitize_log_value(value: Any) -> Any:
    if isinstance(value, dict):
        return sanitize_log_fields(value)
    if isinstance(value, list):
        return [sanitize_log_value(item) for item in value]
    return value


def compact_log_detail(payload: dict[str, Any]) -> str:
    details = {
        key: value
        for key, value in payload.items()
        if key not in {"timestamp", "job_id", "executor", "event", "message"}
        and value is not None
        and value != {}
        and value != []
    }
    if not details:
        return ""
    return json.dumps(details, ensure_ascii=True, default=str, sort_keys=True)
