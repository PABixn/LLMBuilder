from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from fastapi import HTTPException


def extract_bundle(raw: bytes, *, content_type: str | None, incoming_path: Path, job_root: Path) -> dict[str, Any]:
    incoming_path.parent.mkdir(parents=True, exist_ok=True)
    incoming_path.write_bytes(raw)
    if job_root.exists():
        shutil.rmtree(job_root)
    job_root.mkdir(parents=True, exist_ok=True)

    if content_type == "application/zstd" or incoming_path.suffix == ".zst":
        try:
            import zstandard as zstd  # type: ignore[import-not-found]
        except Exception as exc:
            raise HTTPException(status_code=415, detail="zstandard is not installed in the training image.") from exc
        payload = zstd.ZstdDecompressor().decompress(raw)
        fileobj = io.BytesIO(payload)
        mode = "r:"
    else:
        fileobj = io.BytesIO(raw)
        mode = "r:gz" if content_type == "application/gzip" or incoming_path.suffix == ".gz" else "r:"

    try:
        with tarfile.open(fileobj=fileobj, mode=mode) as archive:
            _safe_extract(archive, job_root)
    except tarfile.TarError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid training bundle: {exc}") from exc

    manifest_path = job_root / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=400, detail="Training bundle is missing manifest.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest(job_root, manifest)
    (job_root / "outputs").mkdir(parents=True, exist_ok=True)
    return manifest


def verify_manifest(job_root: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise HTTPException(status_code=400, detail="Bundle manifest files must be a list.")
    for item in files:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Bundle manifest file entries must be objects.")
        rel = item.get("path")
        expected_hash = item.get("sha256")
        expected_size = item.get("size_bytes")
        if not isinstance(rel, str) or not isinstance(expected_hash, str):
            raise HTTPException(status_code=400, detail="Bundle manifest file entry is incomplete.")
        path = safe_join(job_root, rel)
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=400, detail=f"Bundle file is missing: {rel}")
        if isinstance(expected_size, int) and path.stat().st_size != expected_size:
            raise HTTPException(status_code=400, detail=f"Bundle file size mismatch: {rel}")
        if sha256_file(path) != expected_hash:
            raise HTTPException(status_code=400, detail=f"Bundle file checksum mismatch: {rel}")


def safe_join(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise HTTPException(status_code=403, detail="Path traversal is not allowed.")
    return candidate


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        destination_resolved = destination.resolve()
        if target != destination_resolved and destination_resolved not in target.parents:
            raise HTTPException(status_code=403, detail="Bundle contains an unsafe path.")
    archive.extractall(destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
