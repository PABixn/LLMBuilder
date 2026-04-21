from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent


def flatten_selected_data_files(value: Any, *, split: str | None = None) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        results: list[str] = []
        for item in value:
            results.extend(flatten_selected_data_files(item, split=None))
        return results
    if isinstance(value, dict):
        if split is not None and split in value:
            return flatten_selected_data_files(value[split], split=None)
        results: list[str] = []
        for item in value.values():
            results.extend(flatten_selected_data_files(item, split=None))
        return results
    return []


def is_local_text_dataset(name: str, data_files: Any, *, split: str | None = None) -> bool:
    if name != "text" or data_files is None:
        return False
    flattened = flatten_selected_data_files(data_files, split=split)
    return bool(flattened) and all("://" not in raw_path for raw_path in flattened)


def resolve_local_data_files(
    data_files: Any,
    *,
    split: str | None = None,
    relative_base: Path | None = None,
) -> list[Path]:
    selected = flatten_selected_data_files(data_files, split=split)
    base_dir = relative_base or REPO_ROOT
    resolved_paths: list[Path] = []
    seen_paths: set[Path] = set()

    for raw_path in selected:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate

        matches = (
            [Path(match) for match in glob.glob(str(candidate), recursive=True)]
            if has_glob_magic(raw_path)
            else [candidate]
        )
        for match in matches:
            normalized = match.resolve()
            if normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            resolved_paths.append(normalized)

    return resolved_paths


def has_glob_magic(value: str) -> bool:
    return any(ch in value for ch in "*?[")
