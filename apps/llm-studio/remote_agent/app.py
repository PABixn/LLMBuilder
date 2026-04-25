from __future__ import annotations

import json
import os
import importlib
import traceback
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse

from .auth import configured_job_id, require_agent_auth
from .bundle import extract_bundle, safe_join
from .diagnostics import log_event, log_file_path
from .runner import RemoteTrainingRunner, repo_root as training_repo_root
from .sync_manifest import checkpoint_entries

app = FastAPI(title="LLM Studio Remote Training Agent", version="0.1.0")
runner = RemoteTrainingRunner()
manifests: dict[str, dict[str, Any]] = {}


@app.middleware("http")
async def log_failed_requests(request: Request, call_next):
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception as exc:
        agent_log(
            "request_exception",
            job_id=request.headers.get("x-llm-studio-job-id") or configured_job_id() or None,
            method=request.method,
            path=request.url.path,
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=8),
        )
        raise
    if response.status_code >= 400 and request.url.path.startswith("/v1/"):
        agent_log(
            "request_error_response",
            job_id=request.headers.get("x-llm-studio-job-id") or configured_job_id() or None,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
    return response


def workspace_root() -> Path:
    return Path(os.getenv("LLM_STUDIO_REMOTE_WORKSPACE", "/workspace/llm-studio")).resolve()


def job_root(job_id: str) -> Path:
    return workspace_root() / "jobs" / job_id


def outputs_dir(job_id: str) -> Path:
    return job_root(job_id) / "outputs"


AuthDependency = Annotated[None, Depends(require_agent_auth)]


@app.get("/")
def root() -> dict[str, Any]:
    return {"ok": True, "service": "llm-studio-remote-training-agent", "health": "/health"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "job_id": configured_job_id()}


@app.get("/v1/system")
def system(_: AuthDependency) -> dict[str, Any]:
    payload = {
        "workspace": str(workspace_root()),
        "job_id": configured_job_id(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "image_revision": os.getenv("LLM_STUDIO_IMAGE_REVISION"),
        "image_built_at": os.getenv("LLM_STUDIO_IMAGE_BUILT_AT"),
        "logs": {
            "startup": str(log_file_path("startup.log")),
            "agent": str(log_file_path("agent.log")),
            "runner": str(log_file_path("runner.log")),
        },
        "runner": runner_probe(),
    }
    agent_log("system", job_id=configured_job_id(), **payload)
    return payload


@app.post("/v1/jobs/{job_id}/bundle")
async def upload_bundle(
    job_id: str,
    _: AuthDependency,
    request: Request,
    content_type: str | None = Header(default=None),
) -> dict[str, Any]:
    raw = await request.body()
    if not raw:
        agent_log("bundle_empty", job_id=job_id)
        raise HTTPException(status_code=400, detail="Bundle upload is empty.")
    incoming = workspace_root() / "incoming" / f"{job_id}.bundle"
    agent_log(
        "bundle_received",
        job_id=job_id,
        content_type=content_type,
        size_bytes=len(raw),
        incoming_path=str(incoming),
        job_root=str(job_root(job_id)),
    )
    try:
        manifest = extract_bundle(raw, content_type=content_type, incoming_path=incoming, job_root=job_root(job_id))
    except Exception as exc:
        agent_log(
            "bundle_failed",
            job_id=job_id,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=8),
        )
        raise
    manifests[job_id] = manifest
    agent_log(
        "bundle_extracted",
        job_id=job_id,
        file_count=len(manifest.get("files", [])) if isinstance(manifest.get("files"), list) else None,
        runner=manifest.get("runner"),
    )
    return {"ok": True, "job_id": job_id, "files": len(manifest.get("files", []))}


@app.post("/v1/jobs/{job_id}/start")
def start_job(job_id: str, _: AuthDependency) -> dict[str, Any]:
    agent_log("start_requested", job_id=job_id)
    manifest = manifests.get(job_id)
    manifest_path = job_root(job_id) / "manifest.json"
    if manifest is None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[job_id] = manifest
    if manifest is None:
        raise HTTPException(status_code=409, detail="Upload a bundle before starting the job.")
    try:
        pid = runner.start(job_id=job_id, job_root=job_root(job_id), manifest=manifest)
    except RuntimeError as exc:
        agent_log("start_failed", job_id=job_id, error=str(exc), traceback=traceback.format_exc(limit=8))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    agent_log("start_done", job_id=job_id, process_id=pid)
    return {"ok": True, "job_id": job_id, "process_id": pid}


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str, _: AuthDependency) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "process": runner.status(job_id),
        "runtime_state": read_json(outputs_dir(job_id) / "runtime_state.json"),
    }


@app.get("/v1/jobs/{job_id}/runtime-state")
def runtime_state(job_id: str, _: AuthDependency) -> dict[str, Any]:
    payload = read_json(outputs_dir(job_id) / "runtime_state.json")
    process = runner.status(job_id)
    exit_code = process.get("exit_code")
    if isinstance(exit_code, int):
        terminal_statuses = {"completed", "failed", "cancelled"}
        if payload is None or payload.get("status") not in terminal_statuses:
            return {
                "job_id": job_id,
                "status": "failed",
                "state": "failed",
                "stage": "Training runner exited before reporting progress",
                "progress": 1.0,
                "error": f"Training subprocess exited with code {exit_code}. Check stderr.log for details.",
                "process": process,
            }
    if payload is None:
        if process.get("running") is True:
            return {
                "job_id": job_id,
                "status": "running",
                "state": "preflight",
                "stage": "Launching training runner",
                "progress": 0.0,
                "process": process,
            }
        return {"job_id": job_id, "status": "pending", "state": "queued", "stage": "Waiting for runner", "progress": 0.0}
    payload.setdefault("process", process)
    return payload


@app.get("/v1/jobs/{job_id}/metrics")
def metrics(job_id: str, _: AuthDependency, offset: int = Query(default=0, ge=0)) -> Response:
    return ranged_file(outputs_dir(job_id) / "stats.jsonl", offset)


@app.get("/v1/jobs/{job_id}/samples")
def samples(job_id: str, _: AuthDependency, offset: int = Query(default=0, ge=0)) -> Response:
    return ranged_file(outputs_dir(job_id) / "samples.jsonl", offset)


@app.get("/v1/jobs/{job_id}/logs/stdout")
def stdout_log(job_id: str, _: AuthDependency, offset: int = Query(default=0, ge=0)) -> Response:
    return ranged_file(outputs_dir(job_id) / "stdout.log", offset)


@app.get("/v1/jobs/{job_id}/logs/stderr")
def stderr_log(job_id: str, _: AuthDependency, offset: int = Query(default=0, ge=0)) -> Response:
    return ranged_file(outputs_dir(job_id) / "stderr.log", offset)


@app.get("/v1/jobs/{job_id}/logs/startup")
def startup_log(job_id: str, _: AuthDependency, offset: int = Query(default=0, ge=0)) -> Response:
    return ranged_file(log_file_path("startup.log"), offset)


@app.get("/v1/jobs/{job_id}/logs/agent")
def agent_diagnostic_log(job_id: str, _: AuthDependency, offset: int = Query(default=0, ge=0)) -> Response:
    return ranged_file(log_file_path("agent.log"), offset)


@app.get("/v1/jobs/{job_id}/logs/runner")
def runner_diagnostic_log(job_id: str, _: AuthDependency, offset: int = Query(default=0, ge=0)) -> Response:
    return ranged_file(log_file_path("runner.log"), offset)


@app.get("/v1/jobs/{job_id}/checkpoints")
def checkpoints(job_id: str, _: AuthDependency) -> dict[str, Any]:
    return {"job_id": job_id, "checkpoints": checkpoint_entries(outputs_dir(job_id))}


@app.get("/v1/jobs/{job_id}/files")
def files(
    job_id: str,
    _: AuthDependency,
    path: str = Query(min_length=1),
    offset: int = Query(default=0, ge=0),
) -> Response:
    target = safe_join(outputs_dir(job_id), path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File is not available.")
    return ranged_file(target, offset)


@app.post("/v1/jobs/{job_id}/cancel")
def cancel(job_id: str, _: AuthDependency) -> dict[str, Any]:
    agent_log("cancel_requested", job_id=job_id)
    runner.cancel(job_id)
    agent_log("cancel_done", job_id=job_id)
    return {"ok": True, "job_id": job_id}


@app.post("/v1/jobs/{job_id}/shutdown")
def shutdown(job_id: str, _: AuthDependency) -> dict[str, Any]:
    agent_log("shutdown_requested", job_id=job_id)
    runner.cancel(job_id)
    agent_log("shutdown_done", job_id=job_id)
    return {"ok": True, "job_id": job_id}


def ranged_file(path: Path, offset: int) -> Response:
    if not path.exists():
        return PlainTextResponse("", media_type="text/plain")
    size = path.stat().st_size
    if offset >= size:
        return PlainTextResponse("", media_type="text/plain", headers={"X-File-Size": str(size)})
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    return Response(
        data,
        media_type="application/octet-stream",
        headers={"X-File-Size": str(size), "X-Start-Offset": str(offset)},
    )


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


@lru_cache(maxsize=1)
def runner_probe() -> dict[str, Any]:
    modules = ("torch", "datasets", "tokenizers", "llm_builder.local_text_data", "training.runner")
    imported: list[str] = []
    try:
        for module in modules:
            importlib.import_module(module)
            imported.append(module)
    except Exception as exc:
        return {
            "import_ok": False,
            "imported": imported,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
    return {
        "import_ok": True,
        "imported": imported,
        "repo_root": str(training_repo_root()),
    }


def agent_log(event: str, *, job_id: str | None = None, **fields: Any) -> None:
    log_event(
        service="llm-studio-remote-training-agent",
        event=event,
        job_id=job_id,
        file_name="agent.log",
        prefix="llm-studio-agent",
        **fields,
    )
