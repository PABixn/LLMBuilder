#!/usr/bin/env python3
"""Build the desktop shell with deterministic, path-redacted Rust compiler flags."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
DESKTOP_DIR = ROOT / "apps" / "llm-studio" / "desktop"
ENCODED_SEPARATOR = "\x1f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-bundle",
        action="store_true",
        help="Compile the release shell without producing platform installers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="llm-studio-cargo-runner-") as temporary:
        runner = write_auditable_cargo_runner(Path(temporary))
        command = ["npm", "run", "build", "--", "--runner", str(runner)]
        if args.no_bundle:
            command.append("--no-bundle")
        subprocess.run(
            command,
            cwd=DESKTOP_DIR,
            env=release_build_environment(os.environ),
            check=True,
        )


def write_auditable_cargo_runner(directory: Path, *, windows: bool | None = None) -> Path:
    windows = os.name == "nt" if windows is None else windows
    directory.mkdir(parents=True, exist_ok=True)
    if windows:
        runner = directory / "cargo-auditable-runner.cmd"
        runner.write_bytes(b"@echo off\r\ncargo auditable %*\r\n")
        return runner
    runner = directory / "cargo-auditable-runner"
    runner.write_text('#!/bin/sh\nexec cargo auditable "$@"\n', encoding="ascii")
    runner.chmod(0o755)
    return runner


def release_build_environment(source: Mapping[str, str]) -> dict[str, str]:
    environment = dict(source)
    flags = without_conflicting_remaps(existing_rustflags(environment))
    remaps = [
        f"--remap-path-prefix={ROOT}=/build/source",
        f"--remap-path-prefix={Path.home()}=/redacted-builder",
    ]
    for flag in remaps:
        if flag not in flags:
            flags.append(flag)
    environment.pop("RUSTFLAGS", None)
    environment["CARGO_ENCODED_RUSTFLAGS"] = ENCODED_SEPARATOR.join(flags)
    return environment


def without_conflicting_remaps(flags: list[str]) -> list[str]:
    protected = (f"{ROOT}=", f"{Path.home()}=")
    filtered: list[str] = []
    index = 0
    while index < len(flags):
        flag = flags[index]
        if flag == "--remap-path-prefix" and index + 1 < len(flags):
            if flags[index + 1].startswith(protected):
                index += 2
                continue
        if flag.startswith("--remap-path-prefix="):
            value = flag.removeprefix("--remap-path-prefix=")
            if value.startswith(protected):
                index += 1
                continue
        filtered.append(flag)
        index += 1
    return filtered


def existing_rustflags(environment: dict[str, str]) -> list[str]:
    encoded = environment.get("CARGO_ENCODED_RUSTFLAGS")
    if encoded:
        return [flag for flag in encoded.split(ENCODED_SEPARATOR) if flag]
    value = environment.get("RUSTFLAGS", "")
    return shlex.split(value) if value else []


if __name__ == "__main__":
    main()
