from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

from ..schemas import TrainingIssue
from .assets import IMPORT_ROOT
from .config_validation import issue

from training.dataloader_config import TrainingDataloaderConfig


def validate_local_data_files(config: TrainingDataloaderConfig) -> list[TrainingIssue]:
    errors: list[TrainingIssue] = []
    for dataset_index, dataset in enumerate(config.datasets):
        if dataset.data_files is None:
            continue
        paths = flatten_data_files(dataset.data_files)
        for raw_path in paths:
            if "://" in raw_path:
                continue
            resolved = resolve_data_path(raw_path)
            if has_glob_magic(raw_path):
                if not glob.glob(str(resolved), recursive=True):
                    errors.append(
                        issue(
                            "local_dataset_file_missing",
                            f"Local dataset glob did not match any files: {raw_path}",
                            f"$.dataloader_config.datasets[{dataset_index}].data_files",
                        )
                    )
            elif not resolved.exists():
                errors.append(
                    issue(
                        "local_dataset_file_missing",
                        f"Local dataset file does not exist: {raw_path}",
                        f"$.dataloader_config.datasets[{dataset_index}].data_files",
                    )
                )
    return errors


def flatten_data_files(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        results: list[str] = []
        for item in value:
            results.extend(flatten_data_files(item))
        return results
    if isinstance(value, dict):
        results: list[str] = []
        for item in value.values():
            results.extend(flatten_data_files(item))
        return results
    return []


def resolve_data_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return IMPORT_ROOT / candidate


def has_glob_magic(value: str) -> bool:
    return any(ch in value for ch in "*?[")
