from __future__ import annotations

from pathlib import Path
from typing import Any

from .bundle import sha256_file


def checkpoint_entries(outputs_dir: Path) -> list[dict[str, Any]]:
    checkpoints_root = outputs_dir / "checkpoints"
    if not checkpoints_root.exists():
        return []
    entries: list[dict[str, Any]] = []
    for checkpoint_dir in sorted((p for p in checkpoints_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        files = []
        for path in sorted(p for p in checkpoint_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(outputs_dir).as_posix()
            files.append(
                {
                    "path": rel,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "mtime": path.stat().st_mtime,
                }
            )
        entries.append(
            {
                "step": int(checkpoint_dir.name) if checkpoint_dir.name.isdigit() else 0,
                "directory": checkpoint_dir.relative_to(outputs_dir).as_posix(),
                "files": files,
            }
        )
    return entries
