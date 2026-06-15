from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "desktop" / "build_shell.py"


@pytest.fixture(scope="module")
def build_shell_module():
    spec = importlib.util.spec_from_file_location("desktop_build_shell", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_environment_remaps_workspace_and_home(build_shell_module) -> None:
    environment = build_shell_module.release_build_environment({})
    flags = environment["CARGO_ENCODED_RUSTFLAGS"].split(
        build_shell_module.ENCODED_SEPARATOR
    )

    assert f"--remap-path-prefix={build_shell_module.ROOT}=/build/source" in flags
    assert f"--remap-path-prefix={Path.home()}=/redacted-builder" in flags
    assert "RUSTFLAGS" not in environment


def test_release_environment_preserves_existing_encoded_flags(build_shell_module) -> None:
    separator = build_shell_module.ENCODED_SEPARATOR
    environment = build_shell_module.release_build_environment(
        {"CARGO_ENCODED_RUSTFLAGS": separator.join(["-C", "strip=symbols"])}
    )
    flags = environment["CARGO_ENCODED_RUSTFLAGS"].split(separator)

    assert flags[:2] == ["-C", "strip=symbols"]


def test_release_environment_replaces_conflicting_path_remaps(build_shell_module) -> None:
    separator = build_shell_module.ENCODED_SEPARATOR
    environment = build_shell_module.release_build_environment(
        {
            "CARGO_ENCODED_RUSTFLAGS": separator.join(
                [
                    f"--remap-path-prefix={Path.home()}=/home/developer",
                    "--remap-path-prefix",
                    f"{build_shell_module.ROOT}=/tmp/source",
                    "-C",
                    "strip=symbols",
                ]
            )
        }
    )
    flags = environment["CARGO_ENCODED_RUSTFLAGS"].split(separator)

    assert f"--remap-path-prefix={Path.home()}=/home/developer" not in flags
    assert f"{build_shell_module.ROOT}=/tmp/source" not in flags
    assert f"--remap-path-prefix={build_shell_module.ROOT}=/build/source" in flags
    assert f"--remap-path-prefix={Path.home()}=/redacted-builder" in flags
    assert flags[:2] == ["-C", "strip=symbols"]


def test_release_environment_parses_legacy_rustflags(build_shell_module) -> None:
    environment = build_shell_module.release_build_environment(
        {"RUSTFLAGS": "-C opt-level=2"}
    )
    flags = environment["CARGO_ENCODED_RUSTFLAGS"].split(
        build_shell_module.ENCODED_SEPARATOR
    )

    assert flags[:2] == ["-C", "opt-level=2"]
    assert "RUSTFLAGS" not in environment


def test_shell_build_resolves_platform_native_npm_command(
    build_shell_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    def fake_which(command: str) -> str:
        requested.append(command)
        return f"/resolved/{command}"

    monkeypatch.setattr(build_shell_module.shutil, "which", fake_which)

    assert build_shell_module.resolve_npm_command(windows=False) == "/resolved/npm"
    assert build_shell_module.resolve_npm_command(windows=True) == "/resolved/npm.cmd"
    assert requested == ["npm", "npm.cmd"]


def test_shell_build_fails_closed_when_npm_is_unavailable(
    build_shell_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build_shell_module.shutil, "which", lambda _command: None)

    with pytest.raises(SystemExit, match="npm.cmd"):
        build_shell_module.resolve_npm_command(windows=True)


def test_auditable_cargo_runner_wraps_cargo_on_posix(
    build_shell_module,
    tmp_path: Path,
) -> None:
    runner = build_shell_module.write_auditable_cargo_runner(tmp_path, windows=False)

    assert runner.read_text(encoding="ascii") == '#!/bin/sh\nexec cargo auditable "$@"\n'
    assert os.access(runner, os.X_OK)


def test_auditable_cargo_runner_wraps_cargo_on_windows(
    build_shell_module,
    tmp_path: Path,
) -> None:
    runner = build_shell_module.write_auditable_cargo_runner(tmp_path, windows=True)

    assert runner.suffix == ".cmd"
    assert runner.read_bytes() == b"@echo off\r\ncargo auditable %*\r\n"
