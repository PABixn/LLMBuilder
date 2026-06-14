from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile


class ManagedStorageError(RuntimeError):
    """Raised when a managed writable location cannot be used safely."""


class InsufficientStorageError(ManagedStorageError):
    """Raised before a managed write when the target filesystem is too full."""


class ManagedDatabaseError(RuntimeError):
    """Raised when a managed database is locked, read-only, or corrupt."""


class UnsafeManagedPathError(ValueError):
    """Raised when persisted managed metadata points outside its allowed root."""


def require_managed_path(
    candidate: Path,
    root: Path,
    *,
    description: str,
    must_exist: bool = False,
    allow_root: bool = False,
) -> Path:
    """Resolve a managed path and prove it remains under its configured root."""
    root_resolved = root.expanduser().resolve(strict=False)
    try:
        candidate_resolved = candidate.expanduser().resolve(strict=must_exist)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise UnsafeManagedPathError(f"Invalid {description}: {candidate}") from exc

    try:
        relative = candidate_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise UnsafeManagedPathError(
            f"Refusing {description} outside its managed root."
        ) from exc
    if not allow_root and relative == Path("."):
        raise UnsafeManagedPathError(f"Refusing to use the managed root as {description}.")
    return candidate_resolved


def ensure_directory(path: Path, *, operation: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ManagedStorageError(
            f"Could not prepare managed storage for {operation}: {path}. "
            "Check folder permissions and available disk space."
        ) from exc


def ensure_writable_directory(path: Path, *, operation: str) -> None:
    """Prove a managed directory can durably create and remove a unique file."""
    ensure_directory(path, operation=operation)
    probe_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".llm-studio-write-probe-",
            dir=path,
            delete=False,
        ) as probe:
            probe_path = Path(probe.name)
            probe.write(b"ok")
            probe.flush()
            os.fsync(probe.fileno())
        probe_path.unlink()
    except OSError as exc:
        if probe_path is not None:
            try:
                probe_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ManagedStorageError(
            f"Managed storage is not writable for {operation}: {path}. "
            "Check folder permissions and available disk space."
        ) from exc


def ensure_free_space(
    path: Path,
    *,
    minimum_free_bytes: int,
    operation: str,
) -> int:
    """Ensure a managed write root exists and has a conservative free-space floor."""
    ensure_directory(path, operation=operation)
    try:
        free_bytes = shutil.disk_usage(path).free
    except OSError as exc:
        raise InsufficientStorageError(
            f"Could not inspect free space before {operation}: {path}"
        ) from exc
    if free_bytes < minimum_free_bytes:
        raise InsufficientStorageError(
            f"Insufficient free space for {operation}: {free_bytes} bytes available, "
            f"{minimum_free_bytes} required."
        )
    return free_bytes


def database_unavailable_error(database_name: str, path: Path | None) -> ManagedDatabaseError:
    location = f" Database: {path}." if path is not None else ""
    return ManagedDatabaseError(
        f"The {database_name} database is unavailable.{location} It may be locked by another "
        "process, read-only, or corrupt. Close other LLM Studio instances or restore a backup, "
        "then retry."
    )
