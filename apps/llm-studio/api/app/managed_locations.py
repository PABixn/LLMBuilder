from __future__ import annotations

from pathlib import Path, PurePosixPath

from .storage_safety import UnsafeManagedPathError, require_managed_path

MANAGED_LOCATION_PREFIX = "llm-studio-data:v1/"


def encode_managed_location(value: str, data_root: Path) -> str:
    """Encode absolute paths under data_root; preserve external paths and sentinels."""
    if value.startswith(MANAGED_LOCATION_PREFIX):
        return _canonical_managed_location(value, data_root)

    path = Path(value).expanduser()
    if not path.is_absolute():
        return value
    try:
        managed_path = require_managed_path(
            path,
            data_root,
            description="persisted managed location",
        )
    except UnsafeManagedPathError:
        return value
    relative = managed_path.relative_to(data_root.expanduser().resolve(strict=False))
    return MANAGED_LOCATION_PREFIX + relative.as_posix()


def resolve_managed_location(value: str, data_root: Path) -> str:
    """Resolve a typed managed location against the current data root."""
    if not value.startswith(MANAGED_LOCATION_PREFIX):
        return value
    relative = _relative_managed_path(value)
    candidate = data_root.joinpath(*relative.parts)
    return str(
        require_managed_path(
            candidate,
            data_root,
            description="persisted managed location",
        )
    )


def _canonical_managed_location(value: str, data_root: Path) -> str:
    resolved = Path(resolve_managed_location(value, data_root))
    relative = resolved.relative_to(data_root.expanduser().resolve(strict=False))
    return MANAGED_LOCATION_PREFIX + relative.as_posix()


def _relative_managed_path(value: str) -> PurePosixPath:
    raw = value.removeprefix(MANAGED_LOCATION_PREFIX)
    relative = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise UnsafeManagedPathError("Invalid persisted managed location.")
    return relative
