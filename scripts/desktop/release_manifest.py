#!/usr/bin/env python3
"""Generate a path-redacted release manifest and SHA-256 checksum file."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BUILD_ROOT = ROOT / "build" / "desktop"
DEFAULT_OUTPUT = BUILD_ROOT / "release"
GENERATED_NAMES = {"release-manifest.json", "SHA256SUMS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = validate_output_directory(args.output)
    artifacts = validate_artifacts(args.artifacts)
    generate_release_audit_files(artifacts, output)
    print(f"Generated release audit files for {len(artifacts)} artifact(s): {output}")


def generate_release_audit_files(artifacts: list[Path], output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(artifacts)
    manifest_path = output / "release-manifest.json"
    write_atomic(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )

    checksum_entries = [
        (artifact["sha256"], artifact["name"])
        for artifact in manifest["artifacts"]
    ]
    checksum_entries.append((sha256(manifest_path), manifest_path.name))
    checksum_text = "".join(
        f"{digest}  {name}\n" for digest, name in sorted(checksum_entries, key=lambda item: item[1])
    )
    checksums_path = output / "SHA256SUMS"
    write_atomic(checksums_path, checksum_text.encode("utf-8"))
    return manifest_path, checksums_path


def validate_output_directory(value: Path) -> Path:
    raw_output = value.expanduser().absolute()
    if raw_output.is_symlink() or any(parent.is_symlink() for parent in raw_output.parents):
        raise SystemExit(f"Release output path must not traverse symlinks: {raw_output}")
    output = raw_output.resolve()
    build_root = BUILD_ROOT.resolve()
    if output != build_root and build_root not in output.parents:
        raise SystemExit(f"Release output must remain under {build_root}")
    return output


def validate_artifacts(values: list[Path]) -> list[Path]:
    artifacts: list[Path] = []
    names: set[str] = set()
    for value in values:
        path = value.expanduser()
        if path.is_symlink():
            raise SystemExit(f"Release artifact must not be a symlink: {path}")
        path = path.resolve()
        if not path.is_file():
            raise SystemExit(f"Release artifact is not a regular file: {path}")
        name = path.name
        if name in GENERATED_NAMES:
            raise SystemExit(f"Release artifact uses a reserved generated name: {name}")
        if any(character in name for character in "\r\n"):
            raise SystemExit("Release artifact names must not contain newlines.")
        if name in names:
            raise SystemExit(f"Release artifact names must be unique: {name}")
        names.add(name)
        artifacts.append(path)
    return sorted(artifacts, key=lambda path: path.name)


def build_manifest(artifacts: list[Path]) -> dict[str, Any]:
    desktop_package = read_json(ROOT / "apps" / "llm-studio" / "desktop" / "package.json")
    web_package = read_json(ROOT / "apps" / "llm-studio" / "web" / "package.json")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "product": "LLM Studio",
        "versions": {
            "shell": desktop_package["version"],
            "web": web_package["version"],
        },
        "target": {
            "platform": normalize_platform(platform.system()),
            "architecture": normalize_architecture(platform.machine()),
        },
        "provenance": {
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_tree_state": "dirty" if git_value("status", "--porcelain") else "clean",
        },
        "artifacts": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        ],
        "release_gates": {
            "signatures": "external-required",
            "notarization": "external-required-where-applicable",
            "clean_machine_smoke": "external-required",
        },
    }


def write_atomic(path: Path, content: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def normalize_platform(value: str) -> str:
    return {"darwin": "macos", "win32": "windows"}.get(value.lower(), value.lower())


def normalize_architecture(value: str) -> str:
    return {"arm64": "aarch64", "amd64": "x86_64"}.get(value.lower(), value.lower())


if __name__ == "__main__":
    main()
