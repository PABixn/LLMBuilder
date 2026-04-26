from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import shutil
import ssl
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import certifi

from ..schemas import load_json, write_json
from .base import TrainingJobBundle

IMPORT_ROOT = Path(__file__).resolve().parents[5]


DEFAULT_POD_AGENT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class RemoteAgentError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        payload: Any = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload
        self.retryable = retryable


@dataclass(slots=True)
class BundleBuildResult:
    path: Path
    manifest: dict[str, Any]
    content_type: str


class RemoteAgentClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        job_id: str,
        *,
        timeout: float = 30.0,
        user_agent: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._job_id = job_id
        self._timeout = timeout
        configured_user_agent = (
            user_agent
            or os.getenv("LLM_STUDIO_RUNPOD_AGENT_USER_AGENT")
            or DEFAULT_POD_AGENT_USER_AGENT
        )
        self._user_agent = configured_user_agent.strip() or DEFAULT_POD_AGENT_USER_AGENT
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    def health(self) -> dict[str, Any]:
        return self._json("GET", "/health", include_job_header=False, include_auth=False)

    def system(self) -> dict[str, Any]:
        query = urlencode({"job_id": self._job_id})
        return self._json("GET", f"/v1/system?{query}")

    def upload_bundle(self, bundle_path: Path, *, content_type: str) -> dict[str, Any]:
        data = bundle_path.read_bytes()
        return self._json(
            "POST",
            f"/v1/jobs/{quote(self._job_id)}/bundle",
            body=data,
            content_type=content_type,
            timeout=max(self._timeout, 120.0),
        )

    def start(self) -> dict[str, Any]:
        return self._json("POST", f"/v1/jobs/{quote(self._job_id)}/start")

    def status(self) -> dict[str, Any]:
        return self._json("GET", f"/v1/jobs/{quote(self._job_id)}")

    def runtime_state(self) -> dict[str, Any]:
        return self._json("GET", f"/v1/jobs/{quote(self._job_id)}/runtime-state")

    def checkpoints(self) -> list[dict[str, Any]]:
        payload = self._json("GET", f"/v1/jobs/{quote(self._job_id)}/checkpoints")
        items = payload.get("checkpoints") if isinstance(payload, dict) else None
        return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []

    def cancel(self) -> dict[str, Any]:
        return self._json("POST", f"/v1/jobs/{quote(self._job_id)}/cancel")

    def shutdown(self) -> dict[str, Any]:
        return self._json("POST", f"/v1/jobs/{quote(self._job_id)}/shutdown")

    def download_append_file(self, remote_kind: str, local_path: Path) -> None:
        offset = local_path.stat().st_size if local_path.exists() else 0
        query = urlencode({"offset": offset})
        raw = self._bytes("GET", f"/v1/jobs/{quote(self._job_id)}/{remote_kind}?{query}")
        if raw:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with local_path.open("ab") as handle:
                handle.write(raw)

    def download_file(self, relative_path: str, local_path: Path, *, offset: int = 0, optional: bool = False) -> bytes:
        query_payload: dict[str, str | int] = {"path": relative_path, "offset": offset}
        if optional:
            query_payload["optional"] = "1"
        query = urlencode(query_payload)
        try:
            raw = self._bytes("GET", f"/v1/jobs/{quote(self._job_id)}/files?{query}", timeout=120.0)
        except RemoteAgentError as exc:
            if optional and exc.status_code == 404:
                return b""
            raise
        if raw:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "ab" if offset else "wb"
            with local_path.open(mode) as handle:
                handle.write(raw)
        return raw

    def _json(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
        include_job_header: bool = True,
        include_auth: bool = True,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        raw = self._request(
            method,
            path,
            body=body,
            content_type=content_type,
            include_job_header=include_job_header,
            include_auth=include_auth,
            timeout=timeout,
        )
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RemoteAgentError("Pod agent returned a non-JSON response.") from exc
        return payload if isinstance(payload, dict) else {"value": payload}

    def _bytes(self, method: str, path: str, *, timeout: float | None = None) -> bytes:
        return self._request(method, path, timeout=timeout)

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
        include_job_header: bool = True,
        include_auth: bool = True,
        timeout: float | None = None,
    ) -> bytes:
        headers = {
            "Accept": "application/json",
            "User-Agent": self._user_agent,
        }
        if include_auth:
            headers["Authorization"] = f"Bearer {self._token}"
        if include_job_header:
            headers["X-LLM-Studio-Job-Id"] = self._job_id
        if content_type:
            headers["Content-Type"] = content_type
        request = Request(f"{self._base_url}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout or self._timeout, context=self._ssl_context) as response:
                return response.read()
        except HTTPError as exc:
            detail, payload = _decode_http_error(exc)
            retryable = _is_retryable_agent_http_error(exc.code, payload)
            message = _format_agent_http_error(exc.code, detail, payload)
            raise RemoteAgentError(message, status_code=exc.code, payload=payload, retryable=retryable) from exc
        except URLError as exc:
            raise RemoteAgentError(f"Pod agent is unreachable: {exc.reason}") from exc


def _decode_http_error(exc: HTTPError) -> tuple[str, Any]:
    raw = exc.read()
    detail = raw.decode("utf-8", errors="replace") if raw else ""
    try:
        return detail, json.loads(detail)
    except Exception:
        return detail, None


def _format_agent_http_error(status_code: int, detail: str, payload: Any) -> str:
    if isinstance(payload, dict):
        error_code = payload.get("error_code")
        error_name = payload.get("error_name")
        title = payload.get("title")
        if payload.get("cloudflare_error") is True and error_code == 1010:
            return (
                "RunPod proxy rejected the pod-agent request with Cloudflare 1010 "
                f"({error_name or title or 'browser signature blocked'})."
            )
    return f"Pod agent request failed with HTTP {status_code}: {detail}"


def _is_retryable_agent_http_error(status_code: int, payload: Any) -> bool:
    if isinstance(payload, dict) and payload.get("cloudflare_error") is True:
        retryable = payload.get("retryable")
        if retryable is False:
            return False
    if status_code in {401, 403}:
        return False
    return status_code == 404 or status_code >= 500


def build_remote_bundle(bundle: TrainingJobBundle) -> BundleBuildResult:
    staging_dir = bundle.job_dir / ".remote_bundle"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    inputs_dir = staging_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(bundle.model_config_path, inputs_dir / "model_config.json")
    shutil.copy2(bundle.tokenizer_path, inputs_dir / "tokenizer_artifact.json")
    shutil.copy2(bundle.training_config_path, inputs_dir / "training_config.json")
    dataloader_payload = load_json(bundle.dataloader_config_path)
    rewritten_dataloader, local_file_metadata = rewrite_local_dataset_files(
        dataloader_payload,
        source_base=bundle.job_dir,
        target_inputs_dir=inputs_dir,
    )
    write_json(inputs_dir / "dataloader_config.json", rewritten_dataloader)

    preflight_payload = load_json(bundle.resolved_preflight_path)
    if isinstance(preflight_payload, dict):
        preflight_payload["remote_local_files"] = local_file_metadata
    write_json(inputs_dir / "resolved_preflight.json", preflight_payload)

    manifest = {
        "format": "llm-studio-training-bundle-v1",
        "job_id": bundle.job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "runner": {
            "module": "training.runner",
            "args": {
                "model_config_path": "inputs/model_config.json",
                "tokenizer_path": "inputs/tokenizer_artifact.json",
                "training_config_path": "inputs/training_config.json",
                "dataloader_config_path": "inputs/dataloader_config.json",
                "output_dir": "outputs",
            },
        },
    }
    for path in sorted(item for item in inputs_dir.rglob("*") if item.is_file()):
        rel = path.relative_to(staging_dir).as_posix()
        manifest["files"].append(
            {
                "path": rel,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    write_json(staging_dir / "manifest.json", manifest)

    zst_path = bundle.job_dir / "bundle.tar.zst"
    try:
        import zstandard as zstd  # type: ignore[import-not-found]
    except Exception:
        gz_path = bundle.job_dir / "bundle.tar.gz"
        with tarfile.open(gz_path, "w:gz") as archive:
            archive.add(staging_dir, arcname=".")
        return BundleBuildResult(path=gz_path, manifest=manifest, content_type="application/gzip")

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        archive.add(staging_dir, arcname=".")
    compressed = zstd.ZstdCompressor(level=10).compress(tar_buffer.getvalue())
    zst_path.write_bytes(compressed)
    return BundleBuildResult(path=zst_path, manifest=manifest, content_type="application/zstd")


def rewrite_local_dataset_files(
    dataloader_payload: dict[str, Any],
    *,
    source_base: Path,
    target_inputs_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rewritten = json.loads(json.dumps(dataloader_payload))
    metadata: list[dict[str, Any]] = []
    datasets = rewritten.get("datasets")
    if not isinstance(datasets, list):
        return rewritten, metadata
    local_root = target_inputs_dir / "local_files"
    for dataset_index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict) or "data_files" not in dataset:
            continue
        dataset_id = sanitize_path_part(str(dataset.get("name") or f"dataset-{dataset_index}"))
        dataset_root = local_root / f"{dataset_index:03d}-{dataset_id}"
        dataset["data_files"] = _rewrite_data_files_value(
            dataset["data_files"],
            source_base=source_base,
            dataset_root=dataset_root,
            remote_prefix=f"inputs/local_files/{dataset_index:03d}-{dataset_id}",
            metadata=metadata,
        )
    return rewritten, metadata


def _rewrite_data_files_value(
    value: Any,
    *,
    source_base: Path,
    dataset_root: Path,
    remote_prefix: str,
    metadata: list[dict[str, Any]],
) -> Any:
    if isinstance(value, str):
        if "://" in value:
            return value
        paths = [Path(item) for item in glob.glob(str(resolve_data_path(value, source_base)), recursive=True)] if has_glob_magic(value) else [resolve_data_path(value, source_base)]
        remote_paths: list[str] = []
        for path in paths:
            if not path.exists() or not path.is_file():
                continue
            target = dataset_root / f"{len(metadata):05d}-{sanitize_path_part(path.name)}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            remote_path = f"{remote_prefix}/{target.name}"
            remote_paths.append(remote_path)
            metadata.append(
                {
                    "original_path": str(path),
                    "remote_path": remote_path,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        if has_glob_magic(value) or len(remote_paths) != 1:
            return remote_paths
        return remote_paths[0]
    if isinstance(value, list):
        return [
            _rewrite_data_files_value(item, source_base=source_base, dataset_root=dataset_root, remote_prefix=remote_prefix, metadata=metadata)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _rewrite_data_files_value(item, source_base=source_base, dataset_root=dataset_root, remote_prefix=remote_prefix, metadata=metadata)
            for key, item in value.items()
        }
    return value


def resolve_data_path(raw_path: str, source_base: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (IMPORT_ROOT / candidate).resolve()


def has_glob_magic(value: str) -> bool:
    return any(ch in value for ch in "*?[")


def sanitize_path_part(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-")
    return sanitized or "file"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
