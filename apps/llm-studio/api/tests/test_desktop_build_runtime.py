from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import json

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "desktop" / "build_runtime.py"
SPEC = importlib.util.spec_from_file_location("desktop_build_runtime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
build_runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_runtime)


def test_release_portable_runtime_requires_hashed_lock_and_wheelhouse(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="require --wheelhouse and --lock"):
        build_runtime.validate_build_options(
            portable=True,
            install_dependencies=True,
            wheelhouse=None,
            lock=None,
            allow_unlocked_development=False,
        )

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "package.whl").write_bytes(b"wheel")
    lock = tmp_path / "lock.txt"
    lock.write_text("package==1.0 --hash=sha256:abc\n", encoding="utf-8")

    assert (
        build_runtime.validate_build_options(
            portable=True,
            install_dependencies=True,
            wheelhouse=wheelhouse,
            lock=lock,
            allow_unlocked_development=False,
        )
        == "portable"
    )


def test_unlocked_portable_runtime_is_explicitly_non_release() -> None:
    assert (
        build_runtime.validate_build_options(
            portable=True,
            install_dependencies=True,
            wheelhouse=None,
            lock=None,
            allow_unlocked_development=True,
        )
        == "portable-unlocked-development"
    )
    assert (
        build_runtime.validate_build_options(
            portable=True,
            install_dependencies=True,
            wheelhouse=None,
            lock=None,
            allow_unlocked_development=True,
            development_cpu_torch=True,
        )
        == "portable-unlocked-development"
    )


def test_development_cpu_torch_is_rejected_outside_unlocked_characterization(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="requires --allow-unlocked-development"):
        build_runtime.validate_build_options(
            portable=True,
            install_dependencies=True,
            wheelhouse=None,
            lock=None,
            allow_unlocked_development=False,
            development_cpu_torch=True,
        )

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "package.whl").write_bytes(b"wheel")
    lock = tmp_path / "lock.txt"
    lock.write_text("package==1.0 --hash=sha256:abc\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="incompatible with reviewed release inputs"):
        build_runtime.validate_build_options(
            portable=True,
            install_dependencies=True,
            wheelhouse=wheelhouse,
            lock=lock,
            allow_unlocked_development=False,
            development_cpu_torch=True,
        )


def test_unlocked_cpu_torch_install_pins_requirements_to_selected_cpu_wheel(
    monkeypatch,
) -> None:
    calls: list[list[str]] = []
    captured_constraint: list[str] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "-c" in command:
            return subprocess.CompletedProcess(command, 0, stdout="2.9.1+cpu\n")
        if "--constraint" in command:
            constraint = Path(command[command.index("--constraint") + 1])
            captured_constraint.append(constraint.read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(build_runtime.subprocess, "run", fake_run)

    build_runtime.install_unlocked_development_dependencies(
        Path("/runtime/python"),
        cpu_torch=True,
    )

    assert calls[0][-3:] == [
        "--index-url",
        build_runtime.PYTORCH_CPU_INDEX_URL,
        build_runtime.PYTORCH_RUNTIME_REQUIREMENT,
    ]
    assert calls[1][1:3] == ["-c", "import importlib.metadata; print(importlib.metadata.version('torch'))"]
    assert "--constraint" in calls[2]
    assert calls[2][-3:] == [
        build_runtime.PYTORCH_SAFE_SETUPTOOLS_REQUIREMENT,
        "--requirement",
        str(build_runtime.API_DIR / "requirements.txt"),
    ]
    assert captured_constraint == [
        "torch==2.9.1+cpu\nsetuptools>=78.1.1,<82\n"
    ]


def test_release_dependency_inputs_reject_unhashed_lock_and_symlink(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "package.whl"
    wheel.write_bytes(b"wheel")
    lock = tmp_path / "lock.txt"
    lock.write_text("package==1.0\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="sha256 hashes"):
        build_runtime.validate_release_dependency_inputs(wheelhouse, lock)

    lock.write_text("package==1.0 --hash=sha256:abc\n", encoding="utf-8")
    (wheelhouse / "linked.whl").symlink_to(wheel)
    with pytest.raises(SystemExit, match="contains symlinks"):
        build_runtime.validate_release_dependency_inputs(wheelhouse, lock)


def test_release_dependency_inputs_are_copied_and_hashed(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "package.whl"
    wheel.write_bytes(b"wheel")
    lock = tmp_path / "lock.txt"
    lock.write_text("package==1.0 --hash=sha256:abc\n", encoding="utf-8")
    output = tmp_path / "runtime"
    output.mkdir()

    provenance = build_runtime.write_dependency_inputs(
        output,
        wheelhouse=wheelhouse,
        lock=lock,
    )

    assert provenance["dependency_lock_sha256"] == build_runtime.sha256(
        output / "python-lock.txt"
    )
    assert provenance["wheelhouse_inventory_sha256"] == build_runtime.sha256(
        output / "wheelhouse-inventory.json"
    )
    assert "python-lock.txt" in build_runtime.required_runtime_files(
        Path("python/bin/python"),
        include_release_inputs=True,
    )


def test_portable_runtime_package_manager_is_removed(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=True, symlinks=False, clear=True).create(runtime / "python")
    runtime_python = runtime / build_runtime.python_relative_path()

    assert _module_exists(runtime_python, "pip")
    build_runtime.remove_runtime_package_manager(runtime_python)
    build_runtime.sanitize_portable_runtime(runtime / "python", runtime_python)
    assert not _module_exists(runtime_python, "pip")
    expected_executables = {runtime_python.name}
    if os.name == "nt":
        expected_executables.add("pythonw.exe")
    assert {path.name for path in runtime_python.parent.iterdir()} == expected_executables
    assert not list((runtime / "python").rglob("*.pyc"))
    pyvenv_config = (runtime / "python" / "pyvenv.cfg").read_text(encoding="utf-8")
    assert "command = " not in pyvenv_config
    assert "executable = " not in pyvenv_config


def test_portable_runtime_sanitizer_removes_dependency_tests_and_bytecode(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "python"
    executable = runtime / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"python")
    tests = runtime / "lib" / "package" / "tests"
    tests.mkdir(parents=True)
    (tests / "fixture.txt").write_text("fixture", encoding="utf-8")
    cache = runtime / "lib" / "package" / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"bytecode")
    (runtime / "bin" / "console-script").write_text("#! /tmp/python\n", encoding="utf-8")

    build_runtime.sanitize_portable_runtime(runtime, executable)

    assert executable.is_file()
    assert not tests.exists()
    assert not cache.exists()
    assert not (runtime / "bin" / "console-script").exists()


def test_runtime_inventory_probes_disable_bytecode_writes() -> None:
    environment = build_runtime.immutable_python_environment()

    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_runtime_size_policy_requires_release_threshold_and_enforces_limits(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "size-policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "development_guardrails": {
                    "linked-development": {
                        "max_payload_bytes": 10,
                        "max_payload_files": 1,
                    }
                },
                "release_thresholds": {},
            }
        ),
        encoding="utf-8",
    )

    limit = build_runtime.load_size_limit(
        policy,
        build_mode="linked-development",
        target="macos-aarch64",
    )
    assert limit["threshold_kind"] == "development_guardrail"
    build_runtime.enforce_runtime_size(
        {"payload_file_count": 1, "payload_total_bytes": 10},
        limit,
    )
    with pytest.raises(SystemExit, match="exceeds its reviewed size policy"):
        build_runtime.enforce_runtime_size(
            {"payload_file_count": 2, "payload_total_bytes": 11},
            limit,
        )
    with pytest.raises(SystemExit, match="not approved"):
        build_runtime.load_size_limit(
            policy,
            build_mode="portable",
            target="macos-aarch64",
        )

    malformed = tmp_path / "malformed-size-policy.json"
    malformed.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "development_guardrails": [],
                "release_thresholds": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="development_guardrails must be an object"):
        build_runtime.load_size_limit(
            malformed,
            build_mode="linked-development",
            target="macos-aarch64",
        )


def test_runtime_payload_measurement_excludes_manifest(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "payload.bin").write_bytes(b"payload")
    (runtime / "manifest.json").write_bytes(b"manifest")

    assert build_runtime.measure_runtime_payload(runtime) == {
        "payload_file_count": 1,
        "payload_total_bytes": 7,
    }


def test_linked_runtime_build_is_content_deterministic() -> None:
    build_root = ROOT / "build" / "desktop"
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="runtime-determinism-", dir=build_root) as temporary:
        temporary_root = Path(temporary)
        first = temporary_root / "first"
        second = temporary_root / "second"
        for output in (first, second):
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--python",
                    sys.executable,
                    "--output",
                    str(output),
                    "--runtime-version",
                    "determinism-test",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

        assert _runtime_content_snapshot(first) == _runtime_content_snapshot(second)


def _runtime_content_snapshot(root: Path) -> dict[str, tuple[str, str]]:
    snapshot: dict[str, tuple[str, str]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            snapshot[relative] = ("file", build_runtime.sha256(path))
        elif path.is_dir():
            snapshot[relative] = ("directory", "")
    return snapshot


def _module_exists(python: Path, module: str) -> bool:
    result = subprocess.run(
        [
            str(python),
            "-c",
            f"import importlib.util,sys;sys.exit(0 if importlib.util.find_spec({module!r}) else 1)",
        ],
        check=False,
    )
    return result.returncode == 0
