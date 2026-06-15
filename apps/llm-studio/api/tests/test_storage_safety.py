from __future__ import annotations

from pathlib import Path

import pytest

from app import storage_safety


def test_require_managed_path_accepts_nested_paths(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    nested = root / "jobs" / "job-1"

    assert storage_safety.require_managed_path(
        nested,
        root,
        description="training job directory",
    ) == nested.resolve()


def test_require_managed_path_rejects_escape_and_root(tmp_path: Path) -> None:
    root = tmp_path / "managed"

    with pytest.raises(storage_safety.UnsafeManagedPathError, match="outside"):
        storage_safety.require_managed_path(
            tmp_path / "other",
            root,
            description="artifact",
        )
    with pytest.raises(storage_safety.UnsafeManagedPathError, match="managed root"):
        storage_safety.require_managed_path(
            root,
            root,
            description="artifact",
        )


def test_require_managed_path_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform.")

    with pytest.raises(storage_safety.UnsafeManagedPathError, match="outside"):
        storage_safety.require_managed_path(
            link / "artifact.json",
            root,
            description="artifact",
        )


def test_ensure_free_space_reports_operation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        storage_safety.shutil,
        "disk_usage",
        lambda _path: storage_safety.shutil._ntuple_diskusage(total=100, used=99, free=1),
    )

    with pytest.raises(storage_safety.InsufficientStorageError, match="tokenizer artifact"):
        storage_safety.ensure_free_space(
            tmp_path / "managed",
            minimum_free_bytes=2,
            operation="tokenizer artifact",
        )


def test_writable_probe_is_unique_and_preserves_existing_files(tmp_path: Path) -> None:
    managed = tmp_path / "managed"
    managed.mkdir()
    previous_fixed_probe = managed / ".llm-studio-write-probe"
    previous_fixed_probe.write_text("user data", encoding="utf-8")

    storage_safety.ensure_writable_directory(managed, operation="startup validation")

    assert previous_fixed_probe.read_text(encoding="utf-8") == "user data"
    assert sorted(path.name for path in managed.iterdir()) == [previous_fixed_probe.name]
