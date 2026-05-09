from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...store import StoredTrainingJob
from .agent_client import RemoteAgentClient
from .errors import RemoteAgentError
from .lifecycle_log import log_lifecycle


@dataclass(frozen=True)
class RemoteSyncResult:
    checkpoint_count: int | None = None
    final_manifest_verified: bool = False


def sync_incremental_outputs(agent: RemoteAgentClient, job: StoredTrainingJob) -> RemoteSyncResult:
    root = Path(job.artifact_dir)
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
    for remote_name in ("runtime_state.json", "metadata.json", "artifact_manifest.json"):
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
    checkpoint_count = sync_remote_checkpoints(agent, job)
    return RemoteSyncResult(checkpoint_count=checkpoint_count)


def sync_final_outputs(agent: RemoteAgentClient, job: StoredTrainingJob) -> RemoteSyncResult:
    result = sync_incremental_outputs(agent, job)
    manifest_path = Path(job.artifact_dir) / "artifact_manifest.json"
    if not manifest_path.exists():
        agent.download_file("artifact_manifest.json", manifest_path, optional=False)
    manifest = _read_manifest(manifest_path)
    if manifest.get("job_id") not in {None, job.id}:
        raise RemoteAgentError("Remote artifact manifest belongs to a different training job.")
    log_lifecycle(
        job,
        "sync_final_manifest_verified",
        "Verified final remote artifact manifest before cleanup.",
        manifest_path=str(manifest_path),
        file_count=len(manifest.get("files", [])) if isinstance(manifest.get("files"), list) else None,
        checkpoint_count=result.checkpoint_count,
    )
    return RemoteSyncResult(
        checkpoint_count=result.checkpoint_count,
        final_manifest_verified=True,
    )


def sync_remote_checkpoints(agent: RemoteAgentClient, job: StoredTrainingJob) -> int:
    root = Path(job.artifact_dir)
    try:
        checkpoints = agent.checkpoints()
    except RemoteAgentError as exc:
        log_lifecycle(
            job,
            "sync_checkpoints_unavailable",
            "Remote checkpoint manifest is not available yet.",
            error=str(exc),
            throttle_seconds=60,
        )
        return _local_checkpoint_count(root)

    synced_steps: set[int] = set()
    for checkpoint in checkpoints:
        step = _checkpoint_step(checkpoint)
        files = checkpoint.get("files")
        if not isinstance(files, list):
            continue
        for file_entry in files:
            if not isinstance(file_entry, dict):
                continue
            remote_path = file_entry.get("path")
            if not isinstance(remote_path, str) or not _safe_relative_path(remote_path):
                log_lifecycle(
                    job,
                    "sync_checkpoint_file_skipped",
                    "Skipped unsafe remote checkpoint path.",
                    remote_path=remote_path,
                )
                continue
            local_path = root / remote_path
            expected_size = file_entry.get("size_bytes")
            expected_sha256 = file_entry.get("sha256")
            if _local_file_matches(
                local_path,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            ):
                synced_steps.add(step)
                continue
            agent.download_file(remote_path, local_path, optional=False)
            _verify_downloaded_file(
                local_path,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
            synced_steps.add(step)
            log_lifecycle(
                job,
                "sync_checkpoint_file",
                "Synced remote checkpoint file.",
                remote_path=remote_path,
                local_path=str(local_path),
                size_bytes=local_path.stat().st_size if local_path.exists() else None,
                step=step,
            )
    return max(len(synced_steps), _local_checkpoint_count(root))


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RemoteAgentError(f"Remote artifact manifest could not be verified: {exc}") from exc
    if not isinstance(payload, dict):
        raise RemoteAgentError("Remote artifact manifest is not an object.")
    return payload


def _checkpoint_step(checkpoint: dict[str, Any]) -> int:
    value = checkpoint.get("step")
    return value if isinstance(value, int) else 0


def _safe_relative_path(path: str) -> bool:
    candidate = Path(path)
    return not candidate.is_absolute() and ".." not in candidate.parts


def _local_file_matches(
    path: Path,
    *,
    expected_size: Any,
    expected_sha256: Any,
) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if isinstance(expected_size, int) and path.stat().st_size != expected_size:
        return False
    if isinstance(expected_sha256, str) and _sha256_file(path) != expected_sha256:
        return False
    return True


def _verify_downloaded_file(
    path: Path,
    *,
    expected_size: Any,
    expected_sha256: Any,
) -> None:
    if not path.exists() or not path.is_file():
        raise RemoteAgentError(f"Remote checkpoint file was not downloaded: {path}")
    if isinstance(expected_size, int) and path.stat().st_size != expected_size:
        raise RemoteAgentError(
            f"Remote checkpoint file size mismatch for {path}: expected {expected_size}, got {path.stat().st_size}."
        )
    if isinstance(expected_sha256, str):
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RemoteAgentError(
                f"Remote checkpoint checksum mismatch for {path}: expected {expected_sha256}, got {actual_sha256}."
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_checkpoint_count(root: Path) -> int:
    checkpoints_root = root / "checkpoints"
    if not checkpoints_root.exists():
        return 0
    return sum(1 for path in checkpoints_root.iterdir() if path.is_dir())


sync_small_outputs = sync_incremental_outputs
