from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import platform

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "desktop" / "stage_runtime.py"
SPEC = importlib.util.spec_from_file_location("desktop_stage_runtime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
stage_runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_runtime)


def make_runtime(tmp_path: Path, *, build_mode: str = "portable") -> Path:
    runtime = tmp_path / "runtime"
    source = runtime / "source" / "required.txt"
    source.parent.mkdir(parents=True)
    source.write_text("required", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "schema_version": stage_runtime.SUPPORTED_RUNTIME_MANIFEST_SCHEMA,
        "api_contract_version": stage_runtime.SUPPORTED_API_CONTRACT,
        "data_schema_version": stage_runtime.SUPPORTED_DATA_SCHEMA,
        "platform": stage_runtime.normalize_platform(platform.system()),
        "architecture": stage_runtime.normalize_architecture(platform.machine()),
        "required_files": ["source/required.txt"],
        "file_hashes": {"source/required.txt": digest},
        "provenance": {"build_mode": build_mode},
        "size": {
            "threshold_kind": "release_threshold",
            "target": (
                f"{stage_runtime.normalize_platform(platform.system())}-"
                f"{stage_runtime.normalize_architecture(platform.machine())}"
            ),
            "max_payload_bytes": 1024,
            "max_payload_files": 10,
            "payload_file_count": 1,
            "payload_total_bytes": source.stat().st_size,
            "policy_file": "scripts/desktop/runtime-size-policy.json",
        },
    }
    (runtime / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return runtime


def test_stage_validation_accepts_verified_portable_runtime(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    stage_runtime.validate_runtime_for_staging(runtime, allow_development_runtime=False)


def test_stage_validation_rejects_development_runtime_and_hash_mismatch(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path, build_mode="linked-development")
    with pytest.raises(SystemExit, match="non-portable"):
        stage_runtime.validate_runtime_for_staging(runtime, allow_development_runtime=False)

    runtime = make_runtime(tmp_path / "hash")
    (runtime / "source" / "required.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(SystemExit, match="checksum mismatch"):
        stage_runtime.validate_runtime_for_staging(runtime, allow_development_runtime=False)


def test_stage_validation_rejects_unsafe_manifest_paths_and_release_symlinks(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path / "unsafe")
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["required_files"] = ["../outside"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="unsafe path"):
        stage_runtime.validate_runtime_for_staging(runtime, allow_development_runtime=False)

    runtime = make_runtime(tmp_path / "symlink")
    (runtime / "source" / "linked.txt").symlink_to(runtime / "source" / "required.txt")
    with pytest.raises(SystemExit, match="contains symlinks"):
        stage_runtime.validate_runtime_for_staging(runtime, allow_development_runtime=False)


def test_stage_validation_recomputes_release_size_and_requires_approved_threshold(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path / "mismatch")
    (runtime / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(SystemExit, match="payload file count mismatch"):
        stage_runtime.validate_runtime_for_staging(runtime, allow_development_runtime=False)

    runtime = make_runtime(tmp_path / "unapproved")
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["size"]["threshold_kind"] = "development_guardrail"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="approved release size threshold"):
        stage_runtime.validate_runtime_for_staging(runtime, allow_development_runtime=False)


def test_explicit_development_staging_accepts_development_size_guardrail(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path, build_mode="linked-development")
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["size"]["threshold_kind"] = "development_guardrail"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    stage_runtime.validate_runtime_for_staging(runtime, allow_development_runtime=True)


def test_stage_validation_rejects_size_threshold_for_different_target(
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path)
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["size"]["target"] = "other-platform-other-architecture"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit, match="size threshold target mismatch"):
        stage_runtime.validate_runtime_for_staging(runtime, allow_development_runtime=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("api_contract_version", "2"),
        ("data_schema_version", "4"),
    ],
)
def test_stage_validation_rejects_incompatible_runtime_contract(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    runtime = make_runtime(tmp_path)
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit, match=field):
        stage_runtime.validate_runtime_for_staging(runtime, allow_development_runtime=False)
