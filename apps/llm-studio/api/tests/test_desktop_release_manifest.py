from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "desktop" / "release_manifest.py"
SPEC = importlib.util.spec_from_file_location("desktop_release_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
release_manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_manifest)


def test_release_manifest_has_redacted_paths_and_matching_checksums(tmp_path: Path) -> None:
    first = tmp_path / "LLM-Studio.dmg"
    second = tmp_path / "LLM-Studio.AppImage"
    first.write_bytes(b"macos")
    second.write_bytes(b"linux")

    artifacts = release_manifest.validate_artifacts([second, first])
    manifest = release_manifest.build_manifest(artifacts)

    assert [artifact["name"] for artifact in manifest["artifacts"]] == [
        "LLM-Studio.AppImage",
        "LLM-Studio.dmg",
    ]
    assert all(str(tmp_path) not in json.dumps(artifact) for artifact in manifest["artifacts"])
    assert manifest["artifacts"][0]["sha256"] == release_manifest.sha256(second)
    assert manifest["release_gates"]["signatures"] == "external-required"


def test_release_manifest_writes_auditable_manifest_and_checksum_file(tmp_path: Path) -> None:
    artifact = tmp_path / "LLM-Studio.dmg"
    artifact.write_bytes(b"installer")
    output = tmp_path / "release"

    manifest_path, checksums_path = release_manifest.generate_release_audit_files(
        [artifact],
        output,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums = checksums_path.read_text(encoding="utf-8")
    assert manifest["artifacts"][0]["name"] == artifact.name
    assert f"{release_manifest.sha256(artifact)}  {artifact.name}" in checksums
    assert f"{release_manifest.sha256(manifest_path)}  {manifest_path.name}" in checksums
    assert not list(output.glob(".*.tmp"))


def test_release_manifest_rejects_symlinks_duplicate_names_and_unsafe_output(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    symlink = tmp_path / "artifact-link.bin"
    symlink.symlink_to(artifact)
    with pytest.raises(SystemExit, match="symlink"):
        release_manifest.validate_artifacts([symlink])

    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    duplicate = duplicate_dir / artifact.name
    duplicate.write_bytes(b"duplicate")
    with pytest.raises(SystemExit, match="unique"):
        release_manifest.validate_artifacts([artifact, duplicate])

    with pytest.raises(SystemExit, match="must remain under"):
        release_manifest.validate_output_directory(tmp_path / "outside")
