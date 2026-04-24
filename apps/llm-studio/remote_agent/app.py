from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse

from .auth import configured_job_id, require_agent_auth
from .bundle import extract_bundle, safe_join
from .runner import RemoteTrainingRunner
from .sync_manifest import checkpoint_entries

app = FastAPI(title="LLM Studio Remote Training Agent", version="0.1.0")
runner = RemoteTrainingRunner()
manifests: dict[str, dict[str, Any]] = {}


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
    return {
        "workspace": str(workspace_root()),
        "job_id": configured_job_id(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }


@app.post("/v1/jobs/{job_id}/bundle")
async def upload_bundle(
    job_id: str,
    _: AuthDependency,
    request: Request,
    content_type: str | None = Header(default=None),
) -> dict[str, Any]:
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Bundle upload is empty.")
    incoming = workspace_root() / "incoming" / f"{job_id}.bundle"
    manifest = extract_bundle(raw, content_type=content_type, incoming_path=incoming, job_root=job_root(job_id))
    manifests[job_id] = manifest
    return {"ok": True, "job_id": job_id, "files": len(manifest.get("files", []))}


@app.post("/v1/jobs/{job_id}/start")
def start_job(job_id: str, _: AuthDependency) -> dict[str, Any]:
    manifest = manifests.get(job_id)
    manifest_path = job_root(job_id) / "manifest.json"
    if manifest is None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[job_id] = manifest
    if manifest is None:
        raise HTTPException(status_code=409, detail="Upload a bundle before starting the job.")
    pid = runner.start(job_id=job_id, job_root=job_root(job_id), manifest=manifest)
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
    if payload is None:
        return {"job_id": job_id, "status": "pending", "state": "queued", "stage": "Waiting for runner", "progress": 0.0}
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
    runner.cancel(job_id)
    return {"ok": True, "job_id": job_id}


@app.post("/v1/jobs/{job_id}/shutdown")
def shutdown(job_id: str, _: AuthDependency) -> dict[str, Any]:
    runner.cancel(job_id)
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
