from __future__ import annotations

import importlib.util
from pathlib import Path
import re

from app import data_migrations, desktop_runtime

ROOT = Path(__file__).resolve().parents[4]


def load_script(name: str):
    path = ROOT / "scripts" / "desktop" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"desktop_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rust_string_constant(name: str) -> str:
    source = (
        ROOT / "apps" / "llm-studio" / "desktop" / "src-tauri" / "src" / "main.rs"
    ).read_text(encoding="utf-8")
    match = re.search(rf'^const {re.escape(name)}: &str = "([^"]+)";$', source, re.MULTILINE)
    assert match is not None, f"missing Rust contract constant {name}"
    return match.group(1)


def rust_integer_constant(name: str) -> int:
    source = (
        ROOT / "apps" / "llm-studio" / "desktop" / "src-tauri" / "src" / "main.rs"
    ).read_text(encoding="utf-8")
    match = re.search(rf"^const {re.escape(name)}: u32 = (\d+);$", source, re.MULTILINE)
    assert match is not None, f"missing Rust contract constant {name}"
    return int(match.group(1))


def test_desktop_contract_versions_are_consistent_across_runtime_and_shell() -> None:
    build_runtime = load_script("build_runtime")
    audit_dependencies = load_script("audit_dependencies")
    print_versions = load_script("print_versions")
    stage_runtime = load_script("stage_runtime")

    expected_manifest = int(desktop_runtime.RUNTIME_MANIFEST_SCHEMA_VERSION)
    expected_api = desktop_runtime.API_CONTRACT_VERSION
    expected_data = str(data_migrations.DATA_SCHEMA_VERSION)

    assert build_runtime.RUNTIME_MANIFEST_SCHEMA_VERSION == expected_manifest
    assert build_runtime.API_CONTRACT_VERSION == expected_api
    assert build_runtime.DATA_SCHEMA_VERSION == expected_data

    assert audit_dependencies.SUPPORTED_RUNTIME_MANIFEST_SCHEMA == expected_manifest

    assert print_versions.RUNTIME_MANIFEST_SCHEMA_VERSION == str(expected_manifest)
    assert print_versions.API_CONTRACT_VERSION == expected_api
    assert print_versions.DATA_SCHEMA_VERSION == expected_data

    assert stage_runtime.SUPPORTED_RUNTIME_MANIFEST_SCHEMA == expected_manifest
    assert stage_runtime.SUPPORTED_API_CONTRACT == expected_api
    assert stage_runtime.SUPPORTED_DATA_SCHEMA == expected_data

    assert rust_integer_constant("SUPPORTED_MANIFEST_SCHEMA") == expected_manifest
    assert rust_string_constant("SUPPORTED_API_CONTRACT") == expected_api
    assert rust_string_constant("SUPPORTED_DATA_SCHEMA") == expected_data
