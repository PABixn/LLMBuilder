from __future__ import annotations

import io
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from ....schemas import load_json, write_json
from ..base import TrainingJobBundle
from .dataset_files import rewrite_local_dataset_files, sha256_file


class RemoteBundleManifest(TypedDict):
    format: str
    job_id: str
    created_at: str
    files: list[dict[str, Any]]
    runner: dict[str, Any]


@dataclass(slots=True)
class BundleBuildResult:
    path: Path
    manifest: RemoteBundleManifest
    content_type: str


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

    manifest: RemoteBundleManifest = {
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
        shutil.rmtree(staging_dir, ignore_errors=True)
        return BundleBuildResult(path=gz_path, manifest=manifest, content_type="application/gzip")

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        archive.add(staging_dir, arcname=".")
    compressed = zstd.ZstdCompressor(level=10).compress(tar_buffer.getvalue())
    zst_path.write_bytes(compressed)
    shutil.rmtree(staging_dir, ignore_errors=True)
    return BundleBuildResult(path=zst_path, manifest=manifest, content_type="application/zstd")
