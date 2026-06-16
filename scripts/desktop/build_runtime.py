#!/usr/bin/env python3
"""Build a target-native LLM Studio desktop runtime with a validated manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[2]
API_DIR = ROOT / "apps" / "llm-studio" / "api"
DEFAULT_BUILD_ROOT = ROOT / "build" / "desktop" / "runtime"
DEFAULT_SIZE_POLICY = ROOT / "scripts" / "desktop" / "runtime-size-policy.json"
PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
PYTORCH_RUNTIME_REQUIREMENT = "torch>=2.2.0"
PYTORCH_SAFE_SETUPTOOLS_REQUIREMENT = "setuptools>=78.1.1,<82"
RUNTIME_MANIFEST_SCHEMA_VERSION = 1
API_CONTRACT_VERSION = "1"
DATA_SCHEMA_VERSION = "3"
SOURCE_TREES = (
    (API_DIR / "app", Path("source/apps/llm-studio/api/app")),
    (API_DIR / "templates", Path("source/apps/llm-studio/api/templates")),
    (ROOT / "model", Path("source/model")),
    (ROOT / "tokenizer", Path("source/tokenizer")),
    (ROOT / "training", Path("source/training")),
)
STARTER_DATASET_FILES = (
    (
        API_DIR / "datasets" / "shake.txt",
        Path("source/apps/llm-studio/api/datasets/shake.txt"),
    ),
)
IGNORE_NAMES = {
    ".DS_Store",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "build",
    "datasets",
    "node_modules",
    "out",
    "tests",
}
IGNORE_SUFFIXES = {".pyc", ".pyo"}


def parse_args() -> argparse.Namespace:
    target = f"{normalized_platform()}-{normalized_architecture()}"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_BUILD_ROOT / target)
    parser.add_argument("--runtime-version", default="0.1.0-dev")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--portable",
        action="store_true",
        help="Create a copied venv and install locked requirements for release packaging.",
    )
    parser.add_argument(
        "--install-dependencies",
        action="store_true",
        help="Install API dependencies into a portable runtime.",
    )
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        help="Install release runtime dependencies offline from this reviewed wheelhouse.",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        help="Fully pinned, hash-checked requirements lock for the target runtime.",
    )
    parser.add_argument(
        "--allow-unlocked-development",
        action="store_true",
        help="Allow explicit non-release network resolution for target characterization only.",
    )
    parser.add_argument(
        "--development-cpu-torch",
        action="store_true",
        help=(
            "Install PyTorch from its CPU-only channel for an explicitly unlocked, "
            "non-release characterization runtime."
        ),
    )
    parser.add_argument(
        "--size-policy",
        type=Path,
        default=DEFAULT_SIZE_POLICY,
        help="Checked-in runtime size guardrails and approved release thresholds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    ensure_build_output_is_safe(output)
    source_python = resolve_python_executable(args.python)
    build_mode = validate_build_options(
        portable=args.portable,
        install_dependencies=args.install_dependencies,
        wheelhouse=args.wheelhouse,
        lock=args.lock,
        allow_unlocked_development=args.allow_unlocked_development,
        development_cpu_torch=args.development_cpu_torch,
    )
    target = f"{normalized_platform()}-{normalized_architecture()}"
    size_limit = load_size_limit(args.size_policy, build_mode=build_mode, target=target)

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for source, relative in SOURCE_TREES:
        copy_tree(source, output / relative)
    for source, relative in STARTER_DATASET_FILES:
        copy_file(source, output / relative)
    shutil.copy2(API_DIR / "requirements.txt", output / "source/apps/llm-studio/api/requirements.txt")

    python_relative = create_python_runtime(
        output,
        source_python,
        portable=args.portable,
        install_dependencies=args.install_dependencies,
        wheelhouse=args.wheelhouse,
        lock=args.lock,
        allow_unlocked_development=args.allow_unlocked_development,
        development_cpu_torch=args.development_cpu_torch,
    )
    runtime_python = output / python_relative
    dependencies = dependency_versions(runtime_python)
    dependency_inputs = write_dependency_inputs(
        output,
        wheelhouse=args.wheelhouse,
        lock=args.lock,
    )
    provenance = {
        "build_mode": build_mode,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_tree_state": "dirty" if git_value("status", "--porcelain") else "clean",
        "builder_platform": platform.platform(),
        "builder_python": platform.python_version(),
        **(
            {"development_torch_channel": "pytorch-cpu"}
            if args.development_cpu_torch
            else {}
        ),
        **dependency_inputs,
    }
    write_json(output / "sbom.json", build_sbom(dependencies, provenance))
    write_json(output / "licenses.json", build_license_inventory(runtime_python))
    (output / "VERSION").write_text(f"{args.runtime_version}\n", encoding="utf-8")

    required_files = required_runtime_files(
        python_relative,
        include_release_inputs=build_mode == "portable",
    )
    payload_size = measure_runtime_payload(output)
    enforce_runtime_size(payload_size, size_limit)
    file_hashes = {
        path.relative_to(output).as_posix(): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "runtime_version": args.runtime_version,
        "shell_compatibility": {
            "minimum": "0.1.0",
            "maximum_exclusive": "0.2.0",
        },
        "api_contract_version": API_CONTRACT_VERSION,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "platform": normalized_platform(),
        "architecture": normalized_architecture(),
        "python_version": python_version(runtime_python),
        "source_root": "source",
        "python_executable": python_relative.as_posix(),
        "required_files": required_files,
        "file_hashes": file_hashes,
        "dependency_versions": dependencies,
        "provenance": provenance,
        "size": {
            **payload_size,
            **size_limit,
        },
    }
    write_json(output / "manifest.json", manifest)

    total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"Built {provenance['build_mode']} runtime: {output}")
    print(f"Files: {len(file_hashes) + 1}; size: {total_bytes / 1024 / 1024:.1f} MiB")


def resolve_python_executable(value: Path) -> Path:
    expanded = value.expanduser()
    if expanded.parent == Path("."):
        resolved = shutil.which(str(expanded))
        if resolved is None:
            raise SystemExit(f"Python executable is unavailable on PATH: {value}")
        path = Path(resolved).resolve()
    else:
        path = expanded.resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"Python executable is missing or not executable: {path}")
    return path


def load_size_limit(
    policy_path: Path,
    *,
    build_mode: str,
    target: str,
) -> dict[str, object]:
    path = policy_path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Runtime size policy is missing: {path}")
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Runtime size policy is invalid: {path}: {exc}") from exc
    if policy.get("schema_version") != 1:
        raise SystemExit("Runtime size policy has an unsupported schema version.")
    if build_mode == "portable":
        release_thresholds = policy.get("release_thresholds")
        if not isinstance(release_thresholds, dict):
            raise SystemExit("Runtime size policy release_thresholds must be an object.")
        limits = release_thresholds.get(target)
        threshold_kind = "release_threshold"
        if limits is None:
            raise SystemExit(
                f"Release runtime size threshold is not approved for {target}. "
                "Add a reviewed target-specific release_thresholds entry before building."
            )
    else:
        development_guardrails = policy.get("development_guardrails")
        if not isinstance(development_guardrails, dict):
            raise SystemExit("Runtime size policy development_guardrails must be an object.")
        limits = development_guardrails.get(build_mode)
        threshold_kind = "development_guardrail"
        if limits is None:
            raise SystemExit(f"Runtime size development guardrail is missing for {build_mode}.")
    if not isinstance(limits, dict):
        raise SystemExit(f"Runtime size limit for {build_mode}/{target} must be an object.")
    max_bytes = positive_integer(limits.get("max_payload_bytes"), "max_payload_bytes")
    max_files = positive_integer(limits.get("max_payload_files"), "max_payload_files")
    return {
        "threshold_kind": threshold_kind,
        "target": target,
        "max_payload_bytes": max_bytes,
        "max_payload_files": max_files,
        "policy_file": safe_policy_display(path),
    }


def positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SystemExit(f"Runtime size policy {label} must be a positive integer.")
    return value


def safe_policy_display(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def measure_runtime_payload(runtime: Path) -> dict[str, int]:
    files = [
        path
        for path in runtime.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ]
    return {
        "payload_file_count": len(files),
        "payload_total_bytes": sum(path.stat().st_size for path in files),
    }


def enforce_runtime_size(
    payload_size: dict[str, int],
    size_limit: dict[str, object],
) -> None:
    failures: list[str] = []
    if payload_size["payload_file_count"] > int(size_limit["max_payload_files"]):
        failures.append(
            f"{payload_size['payload_file_count']} files exceeds "
            f"{size_limit['max_payload_files']}"
        )
    if payload_size["payload_total_bytes"] > int(size_limit["max_payload_bytes"]):
        failures.append(
            f"{payload_size['payload_total_bytes']} bytes exceeds "
            f"{size_limit['max_payload_bytes']}"
        )
    if failures:
        raise SystemExit(
            "Runtime payload exceeds its reviewed size policy: " + "; ".join(failures)
        )


def create_python_runtime(
    output: Path,
    source_python: Path,
    *,
    portable: bool,
    install_dependencies: bool,
    wheelhouse: Path | None,
    lock: Path | None,
    allow_unlocked_development: bool,
    development_cpu_torch: bool,
) -> Path:
    if portable:
        runtime_dir = output / "python"
        venv.EnvBuilder(with_pip=True, symlinks=False, clear=True).create(runtime_dir)
        relative = python_relative_path()
        runtime_python = output / relative
        if install_dependencies:
            command = [str(runtime_python), "-m", "pip", "install", "--disable-pip-version-check"]
            command.append("--no-compile")
            if wheelhouse is not None and lock is not None:
                command.extend(
                    [
                        "--no-index",
                        "--find-links",
                        str(wheelhouse.expanduser().resolve()),
                        "--require-hashes",
                        "--requirement",
                        str(lock.expanduser().resolve()),
                    ]
                )
            elif allow_unlocked_development:
                install_unlocked_development_dependencies(
                    runtime_python,
                    cpu_torch=development_cpu_torch,
                )
                command = []
            else:
                raise AssertionError("Portable dependency inputs were not validated.")
            if command:
                subprocess.run(command, check=True)
            subprocess.run([str(runtime_python), "-m", "pip", "check"], check=True)
            remove_runtime_package_manager(runtime_python)
            sanitize_portable_runtime(runtime_dir, runtime_python)
        return relative

    relative = python_relative_path()
    destination = output / relative
    destination.parent.mkdir(parents=True)
    try:
        destination.symlink_to(source_python)
    except OSError as error:
        raise SystemExit(
            "Linked development runtime requires symlink support. "
            "Use --portable for a copied target-native environment."
        ) from error
    return relative


def install_unlocked_development_dependencies(
    runtime_python: Path,
    *,
    cpu_torch: bool,
) -> None:
    pip_install = [
        str(runtime_python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-compile",
    ]
    requirements = str(API_DIR / "requirements.txt")
    if not cpu_torch:
        subprocess.run([*pip_install, "--requirement", requirements], check=True)
        return

    subprocess.run(
        [
            *pip_install,
            "--index-url",
            PYTORCH_CPU_INDEX_URL,
            PYTORCH_RUNTIME_REQUIREMENT,
        ],
        check=True,
    )
    installed_version = subprocess.run(
        [
            str(runtime_python),
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('torch'))",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=immutable_python_environment(),
    ).stdout.strip()
    if not installed_version:
        raise RuntimeError("CPU-only PyTorch installation did not report an installed version.")

    with tempfile.TemporaryDirectory(prefix="llm-studio-torch-constraint-") as temporary:
        constraint = Path(temporary) / "torch-constraint.txt"
        constraint.write_text(
            f"torch=={installed_version}\n{PYTORCH_SAFE_SETUPTOOLS_REQUIREMENT}\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                *pip_install,
                "--constraint",
                str(constraint),
                PYTORCH_SAFE_SETUPTOOLS_REQUIREMENT,
                "--requirement",
                requirements,
            ],
            check=True,
        )


def remove_runtime_package_manager(runtime_python: Path) -> None:
    """Remove build-only pip from a portable runtime and prove it is absent."""
    subprocess.run(
        [str(runtime_python), "-m", "pip", "uninstall", "--yes", "pip"],
        check=True,
    )
    result = subprocess.run(
        [
            str(runtime_python),
            "-c",
            "import importlib.util,sys;"
            "sys.exit(0 if importlib.util.find_spec('pip') is None else 1)",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Portable runtime still contains the build-only pip package manager.")


def sanitize_portable_runtime(runtime_dir: Path, runtime_python: Path) -> None:
    """Remove build-only venv artifacts before hashing a portable runtime."""
    for path in sorted(runtime_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and path.name in {"__pycache__", ".pytest_cache", "tests"}:
            shutil.rmtree(path)
        elif path.is_file() and path.suffix.lower() in IGNORE_SUFFIXES:
            path.unlink()

    executable_dir = runtime_python.parent
    allowed_executables = {runtime_python.name}
    if os.name == "nt":
        allowed_executables.add("pythonw.exe")
    for path in executable_dir.iterdir():
        if path.name in allowed_executables or (
            os.name == "nt" and path.suffix.lower() in {".dll", ".pyd"}
        ):
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()

    pyvenv_config = runtime_dir / "pyvenv.cfg"
    if pyvenv_config.is_file():
        lines = pyvenv_config.read_text(encoding="utf-8").splitlines()
        retained = [
            line
            for line in lines
            if line.partition("=")[0].strip().lower() not in {"command", "executable"}
        ]
        pyvenv_config.write_text("\n".join(retained) + "\n", encoding="utf-8")


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise SystemExit(f"Required source tree is missing: {source}")
    shutil.copytree(source, destination, ignore=copy_ignore)


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"Required source file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in IGNORE_NAMES
        or Path(name).suffix.lower() in IGNORE_SUFFIXES
        or (name.startswith("test_") and Path(name).suffix.lower() == ".py")
    }


def dependency_versions(python: Path) -> dict[str, str]:
    script = (
        "import importlib.metadata,json;"
        "print(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions() "
        "if d.metadata.get('Name')},sort_keys=True))"
    )
    result = subprocess.run(
        [str(python), "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=immutable_python_environment(),
    )
    return json.loads(result.stdout)


def build_sbom(dependencies: dict[str, str], provenance: dict[str, str]) -> dict[str, object]:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"component": {"name": "llm-studio-runtime"}, "properties": provenance},
        "components": [
            {"type": "library", "name": name, "version": version, "purl": f"pkg:pypi/{name}@{version}"}
            for name, version in sorted(dependencies.items())
        ],
    }


def build_license_inventory(python: Path) -> dict[str, object]:
    script = """
import importlib.metadata, json
items = []
for dist in importlib.metadata.distributions():
    name = dist.metadata.get("Name")
    if not name:
        continue
    items.append({
        "name": name,
        "version": dist.version,
        "license": dist.metadata.get("License") or "UNKNOWN",
        "homepage": dist.metadata.get("Home-page") or "",
    })
print(json.dumps({"schema_version": 1, "packages": sorted(items, key=lambda x: x["name"].lower())}))
"""
    result = subprocess.run(
        [str(python), "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=immutable_python_environment(),
    )
    return json.loads(result.stdout)


def required_runtime_files(
    python_relative: Path,
    *,
    include_release_inputs: bool = False,
) -> list[str]:
    required = [
        python_relative.as_posix(),
        "source/apps/llm-studio/api/app/main.py",
        "source/apps/llm-studio/api/app/serve.py",
        "source/apps/llm-studio/api/templates/model_config.json",
        "source/apps/llm-studio/api/templates/model_config_schema.json",
        "source/apps/llm-studio/api/templates/tok_config.json",
        "source/apps/llm-studio/api/templates/tokenizer_config_schema.json",
        "source/apps/llm-studio/api/templates/dataloader_config.json",
        "source/apps/llm-studio/api/templates/dataloader_config_schema.json",
        "source/apps/llm-studio/api/datasets/shake.txt",
        "source/model/model.py",
        "source/tokenizer/loader.py",
        "source/training/training_config.json",
        "source/training/training_config_schema.json",
        "source/training/dataloader_config.json",
        "source/training/dataloader_config_schema.json",
        "VERSION",
        "sbom.json",
        "licenses.json",
    ]
    if include_release_inputs:
        required.extend(["python-lock.txt", "wheelhouse-inventory.json"])
    return required


def python_relative_path() -> Path:
    return Path("python/Scripts/python.exe" if os.name == "nt" else "python/bin/python")


def python_version(python: Path) -> str:
    return subprocess.run(
        [str(python), "-c", "import platform; print(platform.python_version())"],
        check=True,
        capture_output=True,
        text=True,
        env=immutable_python_environment(),
    ).stdout.strip()


def immutable_python_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def normalized_platform() -> str:
    value = platform.system().lower()
    return {"darwin": "macos"}.get(value, value)


def normalized_architecture() -> str:
    value = platform.machine().lower()
    return {"arm64": "aarch64", "amd64": "x86_64"}.get(value, value)


def ensure_build_output_is_safe(output: Path) -> None:
    allowed = (ROOT / "build" / "desktop").resolve()
    try:
        output.relative_to(allowed)
    except ValueError as error:
        raise SystemExit(f"Runtime output must remain under {allowed}: {output}") from error
    if output == allowed:
        raise SystemExit("Refusing to replace the desktop build root itself.")


def validate_build_options(
    *,
    portable: bool,
    install_dependencies: bool,
    wheelhouse: Path | None,
    lock: Path | None,
    allow_unlocked_development: bool,
    development_cpu_torch: bool = False,
) -> str:
    if not portable:
        if (
            install_dependencies
            or wheelhouse
            or lock
            or allow_unlocked_development
            or development_cpu_torch
        ):
            raise SystemExit(
                "Portable dependency options require --portable. "
                "Linked-development runtimes use the selected interpreter unchanged."
            )
        return "linked-development"
    if not install_dependencies:
        raise SystemExit("Portable runtimes require --install-dependencies.")
    if (wheelhouse is None) != (lock is None):
        raise SystemExit("--wheelhouse and --lock must be provided together.")
    if wheelhouse is not None and lock is not None:
        validate_release_dependency_inputs(wheelhouse, lock)
        if allow_unlocked_development or development_cpu_torch:
            raise SystemExit(
                "Unlocked development options are incompatible with reviewed release inputs."
            )
        return "portable"
    if development_cpu_torch and not allow_unlocked_development:
        raise SystemExit(
            "--development-cpu-torch requires --allow-unlocked-development."
        )
    if not allow_unlocked_development:
        raise SystemExit(
            "Release-portable runtimes require --wheelhouse and --lock. "
            "For non-release target characterization only, pass "
            "--allow-unlocked-development explicitly."
        )
    return "portable-unlocked-development"


def validate_release_dependency_inputs(wheelhouse: Path, lock: Path) -> None:
    raw_wheelhouse = wheelhouse.expanduser().absolute()
    raw_lock = lock.expanduser().absolute()
    if raw_wheelhouse.is_symlink() or raw_lock.is_symlink():
        raise SystemExit("Reviewed wheelhouse and runtime lock must not be symlinks.")
    wheelhouse = raw_wheelhouse.resolve()
    lock = raw_lock.resolve()
    if not wheelhouse.is_dir():
        raise SystemExit(f"Reviewed wheelhouse is missing: {wheelhouse}")
    if lock.is_symlink() or not lock.is_file():
        raise SystemExit(f"Reviewed runtime lock must be a regular file: {lock}")
    wheels = [path for path in wheelhouse.iterdir() if path.is_file()]
    if not wheels:
        raise SystemExit(f"Reviewed wheelhouse contains no files: {wheelhouse}")
    non_wheels = [path.name for path in wheels if path.suffix.lower() != ".whl"]
    if non_wheels:
        raise SystemExit(
            f"Reviewed wheelhouse contains non-wheel files: {', '.join(non_wheels[:5])}"
        )
    symlinks = [path.name for path in wheelhouse.rglob("*") if path.is_symlink()]
    if symlinks:
        raise SystemExit(f"Reviewed wheelhouse contains symlinks: {', '.join(symlinks[:5])}")
    lock_text = lock.read_text(encoding="utf-8")
    if "==" not in lock_text or "--hash=sha256:" not in lock_text:
        raise SystemExit(
            "Reviewed runtime lock must contain exact versions and sha256 hashes."
        )


def write_dependency_inputs(
    output: Path,
    *,
    wheelhouse: Path | None,
    lock: Path | None,
) -> dict[str, str]:
    if wheelhouse is None or lock is None:
        return {}
    lock = lock.expanduser().resolve()
    wheelhouse = wheelhouse.expanduser().resolve()
    copied_lock = output / "python-lock.txt"
    shutil.copy2(lock, copied_lock)
    inventory = {
        path.relative_to(wheelhouse).as_posix(): sha256(path)
        for path in sorted(wheelhouse.rglob("*"))
        if path.is_file()
    }
    inventory_path = output / "wheelhouse-inventory.json"
    write_json(inventory_path, inventory)
    return {
        "dependency_lock_sha256": sha256(copied_lock),
        "wheelhouse_inventory_sha256": sha256(inventory_path),
    }


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
