#!/usr/bin/env python3
"""Stage an already-built and smoke-tested runtime into Tauri resources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "apps" / "llm-studio" / "desktop" / "src-tauri" / "resources" / "runtime"
SUPPORTED_RUNTIME_MANIFEST_SCHEMA = 1
SUPPORTED_API_CONTRACT = "1"
SUPPORTED_DATA_SCHEMA = "3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    parser.add_argument(
        "--allow-development-runtime",
        action="store_true",
        help="Allow explicit local staging of a linked-development runtime. Never use for release.",
    )
    args = parser.parse_args()
    runtime = args.runtime.expanduser().resolve()
    validate_runtime_for_staging(
        runtime,
        allow_development_runtime=args.allow_development_runtime,
    )
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for child in DESTINATION.iterdir():
        if child.name != ".gitkeep":
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    shutil.copytree(runtime, DESTINATION, dirs_exist_ok=True, symlinks=True)
    print(f"Staged runtime for Tauri packaging: {runtime} -> {DESTINATION}")


def validate_runtime_for_staging(runtime: Path, *, allow_development_runtime: bool) -> None:
    manifest_path = runtime / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Runtime has no manifest: {runtime}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    build_mode = manifest.get("provenance", {}).get("build_mode")
    if build_mode != "portable" and not allow_development_runtime:
        raise SystemExit(
            f"Refusing to stage non-portable runtime ({build_mode!r}) for packaging. "
            "Build a target-native portable runtime first."
        )
    validate_runtime_contract(manifest)
    if normalize_platform(str(manifest.get("platform", ""))) != normalize_platform(
        platform.system()
    ):
        raise SystemExit("Runtime platform does not match the staging host.")
    if normalize_architecture(str(manifest.get("architecture", ""))) != normalize_architecture(
        platform.machine()
    ):
        raise SystemExit("Runtime architecture does not match the staging host.")

    failures: list[str] = []
    validate_runtime_size(
        runtime,
        manifest,
        failures,
        require_release_threshold=build_mode == "portable" and not allow_development_runtime,
    )
    for field in ("python_executable", "source_root"):
        value = manifest.get(field)
        if value is not None:
            safe_runtime_path(runtime, value)
    for relative in manifest.get("required_files", []):
        path = safe_runtime_path(runtime, relative)
        if not path.is_file():
            failures.append(f"missing required file: {relative}")
    for relative, expected in manifest.get("file_hashes", {}).items():
        path = safe_runtime_path(runtime, relative)
        if not path.is_file():
            failures.append(f"missing hashed file: {relative}")
            continue
        actual = sha256(path)
        if actual.lower() != str(expected).lower():
            failures.append(f"checksum mismatch: {relative}")
    symlinks = [
        path.relative_to(runtime).as_posix()
        for path in runtime.rglob("*")
        if path.is_symlink()
    ]
    if symlinks and not allow_development_runtime:
        failures.append(f"release runtime contains symlinks: {', '.join(symlinks[:5])}")
    if failures:
        raise SystemExit("Runtime staging validation failed:\n- " + "\n- ".join(failures))


def validate_runtime_contract(manifest: dict[str, object]) -> None:
    expected = {
        "schema_version": SUPPORTED_RUNTIME_MANIFEST_SCHEMA,
        "api_contract_version": SUPPORTED_API_CONTRACT,
        "data_schema_version": SUPPORTED_DATA_SCHEMA,
    }
    mismatches = [
        f"{name}: expected {value!r}, got {manifest.get(name)!r}"
        for name, value in expected.items()
        if manifest.get(name) != value
    ]
    if mismatches:
        raise SystemExit(
            "Runtime contract is incompatible with the desktop shell:\n- "
            + "\n- ".join(mismatches)
        )


def validate_runtime_size(
    runtime: Path,
    manifest: dict[str, object],
    failures: list[str],
    *,
    require_release_threshold: bool,
) -> None:
    size = manifest.get("size")
    if not isinstance(size, dict):
        if require_release_threshold:
            failures.append("release runtime manifest has no reviewed size threshold")
        return
    if require_release_threshold and size.get("threshold_kind") != "release_threshold":
        failures.append("release runtime was not built against an approved release size threshold")
    expected_target = (
        f"{normalize_platform(str(manifest.get('platform', '')))}-"
        f"{normalize_architecture(str(manifest.get('architecture', '')))}"
    )
    if size.get("target") != expected_target:
        failures.append(
            f"runtime size threshold target mismatch: expected {expected_target}, "
            f"got {size.get('target')}"
        )
    files = [
        path
        for path in runtime.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ]
    actual_files = len(files)
    actual_bytes = sum(path.stat().st_size for path in files)
    expected_files = size.get("payload_file_count")
    expected_bytes = size.get("payload_total_bytes")
    max_files = size.get("max_payload_files")
    max_bytes = size.get("max_payload_bytes")
    if (
        not isinstance(expected_files, int)
        or isinstance(expected_files, bool)
        or expected_files < 0
    ):
        failures.append("runtime payload_file_count is invalid")
    elif actual_files != expected_files:
        failures.append(
            f"runtime payload file count mismatch: expected {expected_files}, got {actual_files}"
        )
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        failures.append("runtime payload_total_bytes is invalid")
    elif actual_bytes != expected_bytes:
        failures.append(
            f"runtime payload byte count mismatch: expected {expected_bytes}, got {actual_bytes}"
        )
    if not isinstance(max_files, int) or isinstance(max_files, bool) or max_files <= 0:
        failures.append("release runtime max_payload_files is invalid")
    elif actual_files > max_files:
        failures.append(f"runtime payload file count exceeds release threshold: {actual_files} > {max_files}")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        failures.append("release runtime max_payload_bytes is invalid")
    elif actual_bytes > max_bytes:
        failures.append(f"runtime payload bytes exceed release threshold: {actual_bytes} > {max_bytes}")


def safe_runtime_path(runtime: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not is_portable_manifest_path(relative):
        raise SystemExit(f"Runtime manifest contains unsafe path: {relative}")
    return runtime.joinpath(*relative.split("/"))


def is_portable_manifest_path(relative: str) -> bool:
    segments = relative.split("/")
    return bool(relative) and all(
        segment not in {"", ".", ".."} and "\\" not in segment and ":" not in segment
        for segment in segments
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_platform(value: str) -> str:
    return {"darwin": "macos", "win32": "windows"}.get(value.lower(), value.lower())


def normalize_architecture(value: str) -> str:
    return {"arm64": "aarch64", "amd64": "x86_64"}.get(value.lower(), value.lower())


if __name__ == "__main__":
    main()
