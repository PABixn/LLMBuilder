#!/usr/bin/env python3
"""Print desktop shell, web, backend, runtime, platform, and schema versions."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_MANIFEST_SCHEMA_VERSION = "1"
API_CONTRACT_VERSION = "1"
DATA_SCHEMA_VERSION = "3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()

    web = read_json(ROOT / "apps/llm-studio/web/package.json")
    desktop = read_json(ROOT / "apps/llm-studio/desktop/package.json")
    report: dict[str, object] = {
        "shell_version": desktop["version"],
        "web_version": web["version"],
        "backend_version": "0.1.0",
        "api_contract_version": API_CONTRACT_VERSION,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "runtime_manifest_schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "torch_version": package_version("torch"),
        "platform": platform.system().lower(),
        "architecture": platform.machine().lower(),
    }
    if args.runtime is not None:
        runtime = args.runtime.expanduser().resolve()
        manifest = read_json(runtime / "manifest.json")
        report["runtime"] = {
            key: manifest[key]
            for key in (
                "runtime_version",
                "schema_version",
                "api_contract_version",
                "data_schema_version",
                "platform",
                "architecture",
                "python_version",
            )
        }
        report["runtime"]["dependency_count"] = len(manifest.get("dependency_versions", {}))
        report["runtime"]["hashed_file_count"] = len(manifest.get("file_hashes", {}))
        report["runtime"]["build_mode"] = manifest.get("provenance", {}).get("build_mode")
        report["runtime"]["size"] = manifest.get("size")
    print(json.dumps(report, indent=2, sort_keys=True))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


if __name__ == "__main__":
    main()
