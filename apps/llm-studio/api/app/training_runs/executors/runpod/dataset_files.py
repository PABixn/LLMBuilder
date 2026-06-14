from __future__ import annotations

import glob
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ....runtime_paths import source_root

IMPORT_ROOT = source_root()


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
        paths = (
            [Path(item) for item in glob.glob(str(resolve_data_path(value, source_base)), recursive=True)]
            if has_glob_magic(value)
            else [resolve_data_path(value, source_base)]
        )
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
            _rewrite_data_files_value(
                item,
                source_base=source_base,
                dataset_root=dataset_root,
                remote_prefix=remote_prefix,
                metadata=metadata,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _rewrite_data_files_value(
                item,
                source_base=source_base,
                dataset_root=dataset_root,
                remote_prefix=remote_prefix,
                metadata=metadata,
            )
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
