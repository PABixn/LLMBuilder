#!/usr/bin/env python3
"""Validate desktop contributor toolchains and report relevant versions."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MINIMUM_RUST_VERSION = (1, 88, 0)
REQUIRED_CARGO_AUDITABLE_VERSION = "0.7.4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report: dict[str, object] = {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "tools": {},
        "python_packages": {},
        "locks": {},
    }
    failures: list[str] = []
    for tool, version_args in {
        "node": ["--version"],
        "npm": ["--version"],
        "rustc": ["--version"],
        "cargo": ["--version"],
        "cargo-audit": ["--version"],
    }.items():
        executable = shutil.which(tool)
        if executable is None:
            failures.append(f"Missing required tool: {tool}")
            continue
        version = run_version([executable, *version_args])
        report["tools"][tool] = version
        if tool == "rustc" and parsed_rust_version(version) < MINIMUM_RUST_VERSION:
            failures.append(
                "Rust 1.88.0 or newer is required; install the pinned rust-toolchain.toml toolchain."
            )

    cargo_auditable_version = installed_cargo_package_version("cargo-auditable")
    report["tools"]["cargo-auditable"] = cargo_auditable_version or "missing"
    if cargo_auditable_version is None:
        failures.append(
            "Missing required tool: cargo-auditable; install pinned version "
            f"{REQUIRED_CARGO_AUDITABLE_VERSION}."
        )
    elif cargo_auditable_version != REQUIRED_CARGO_AUDITABLE_VERSION:
        failures.append(
            "cargo-auditable "
            f"{REQUIRED_CARGO_AUDITABLE_VERSION} is required; found {cargo_auditable_version}."
        )

    for package in ("fastapi", "sqlalchemy", "datasets", "tokenizers", "torch", "pip-audit"):
        try:
            report["python_packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            failures.append(
                f"Missing Python package {package}; install apps/llm-studio/api/requirements-dev.txt"
            )

    for name, path in {
        "web_npm": ROOT / "apps/llm-studio/web/package-lock.json",
        "desktop_npm": ROOT / "apps/llm-studio/desktop/package-lock.json",
        "desktop_cargo": ROOT / "apps/llm-studio/desktop/src-tauri/Cargo.lock",
        "desktop_api_contract": ROOT / "docs/llm-studio-desktop-api-contract-fixtures.json",
        "desktop_runtime_size_policy": ROOT / "scripts/desktop/runtime-size-policy.json",
    }.items():
        report["locks"][name] = path.is_file()
        if not path.is_file():
            failures.append(f"Missing dependency lockfile: {path.relative_to(ROOT)}")

    report["status"] = "failed" if failures else "ready"
    report["failures"] = failures
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Desktop toolchain status: {report['status']}")
        print(f"Host: {report['platform']} ({report['architecture']})")
        print(f"Python: {report['python']}")
        for name, version in report["tools"].items():
            print(f"{name}: {version}")
        for name, version in report["python_packages"].items():
            print(f"{name}: {version}")
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
    if failures:
        raise SystemExit(1)


def run_version(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def parsed_rust_version(value: str) -> tuple[int, int, int]:
    match = re.search(r"\brustc\s+(\d+)\.(\d+)\.(\d+)\b", value)
    return tuple(int(part) for part in match.groups()) if match else (0, 0, 0)


def installed_cargo_package_version(package: str) -> str | None:
    if shutil.which(package) is None:
        return None
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    try:
        installed = (cargo_home / ".crates.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(
        rf'^"{re.escape(package)} ([^ ]+) \([^"]+\)"\s*=',
        installed,
        re.MULTILINE,
    )
    return match.group(1) if match else None


if __name__ == "__main__":
    main()
