#!/usr/bin/env python3
"""Generate or validate the frozen desktop API request/response contract catalog."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
API_DIR = ROOT / "apps" / "llm-studio" / "api"
CATALOG_PATH = ROOT / "docs" / "llm-studio-desktop-api-contract-fixtures.json"
HTTP_METHODS = {"delete", "get", "patch", "post", "put"}
CONCRETE_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:rpa|rps)_[a-z0-9_-]{16,}\b"),
    re.compile(r"(?i)\bBearer\s+[a-z0-9._~+/-]{24,}\b"),
    re.compile(r"(?i)LLM_STUDIO_RUNTIME_TOKEN\s*=\s*[\"']?[a-f0-9]{32,}"),
)


def _starts(prefix: str) -> Callable[[str], bool]:
    return lambda path: path.startswith(prefix)


GROUPS: tuple[dict[str, Any], ...] = (
    {
        "id": "remote-lifecycle",
        "title": "Remote lifecycle",
        "matches": lambda path: "/remote/" in path,
        "evidence": [
            "apps/llm-studio/api/tests/test_training_routes.py",
            "apps/llm-studio/api/tests/test_runpod_training.py",
        ],
        "representative": {
            "method": "POST",
            "path": "/api/v1/training/jobs/{job_id}/remote/cleanup",
            "request_json": None,
            "response_status": "200",
            "response_json_subset": {
                "id": "<job-id>",
                "status": "running",
                "executor_kind": "runpod_pod",
                "executor_status": "cleaned_up",
            },
        },
    },
    {
        "id": "inference",
        "title": "Inference",
        "matches": lambda path: "/generate" in path,
        "evidence": [
            "scripts/desktop/smoke_runtime.py",
            "apps/llm-studio/api/tests/test_training_artifacts_and_responses.py",
        ],
        "representative": {
            "method": "POST",
            "path": "/api/v1/training/jobs/{job_id}/generate",
            "request_json": {
                "prompt": "hello",
                "checkpoint_step": 1,
                "max_tokens": 2,
            },
            "response_status": "200",
            "response_json_subset": {
                "job_id": "<job-id>",
                "completion": "<generated-text>",
                "generated_token_ids": [1, 2],
            },
        },
    },
    {
        "id": "runpod-provider",
        "title": "RunPod provider",
        "matches": _starts("/api/v1/training/providers/runpod/"),
        "evidence": [
            "apps/llm-studio/api/tests/test_training_routes.py",
            "apps/llm-studio/api/tests/test_training_manager_characterization.py",
        ],
        "representative": {
            "method": "GET",
            "path": "/api/v1/training/providers/runpod/status",
            "request_json": None,
            "response_status": "200",
            "response_json_subset": {
                "configured": False,
                "validated": False,
                "source": "none",
                "defaults": {},
            },
        },
    },
    {
        "id": "training-config-preflight",
        "title": "Training config and preflight",
        "matches": lambda path: path.startswith("/api/v1/training/")
        and (
            path.endswith("/health")
            or "/config/" in path
            or "/validate/" in path
        ),
        "evidence": [
            "scripts/desktop/smoke_runtime.py",
            "apps/llm-studio/api/tests/test_training_preflight_modules.py",
        ],
        "representative": {
            "method": "POST",
            "path": "/api/v1/training/validate/training-config",
            "request_json": {"config": {"max_steps": 1}},
            "response_status": "200",
            "response_json_subset": {
                "valid": True,
                "normalized_config": {},
            },
        },
    },
    {
        "id": "training-jobs",
        "title": "Training jobs",
        "matches": _starts("/api/v1/training/jobs"),
        "evidence": [
            "scripts/desktop/smoke_runtime.py",
            "apps/llm-studio/api/tests/test_training_manager_characterization.py",
        ],
        "representative": {
            "method": "GET",
            "path": "/api/v1/training/jobs",
            "request_json": None,
            "response_status": "200",
            "response_json_subset": {"jobs": []},
        },
    },
    {
        "id": "tokenizer",
        "title": "Tokenizer",
        "matches": _starts("/api/v1/tokenizer/"),
        "evidence": [
            "scripts/desktop/smoke_runtime.py",
            "apps/llm-studio/api/tests/test_main.py",
        ],
        "representative": {
            "method": "POST",
            "path": "/api/v1/tokenizer/validate/tokenizer",
            "request_json": {"config": {"name": "desktop-contract-fixture"}},
            "response_status": "200",
            "response_json_subset": {
                "valid": True,
                "normalized_config": {},
            },
        },
    },
    {
        "id": "model-studio",
        "title": "Model Studio",
        "matches": lambda path: path.startswith("/api/v1/projects")
        or path in {"/api/v1/validate/model", "/api/v1/analyze/model"},
        "evidence": [
            "scripts/desktop/smoke_runtime.py",
            "apps/llm-studio/api/tests/test_main.py",
        ],
        "representative": {
            "method": "POST",
            "path": "/api/v1/validate/model",
            "request_json": {"config": {"context_length": 16, "vocab_size": 64}},
            "response_status": "200",
            "response_json_subset": {
                "valid": True,
                "normalized_config": {},
                "warnings": [],
                "errors": [],
            },
        },
    },
    {
        "id": "system-config",
        "title": "System and config",
        "matches": lambda path: path == "/health"
        or path.startswith("/api/v1/health")
        or path.startswith("/api/v1/desktop/")
        or path.startswith("/api/v1/config/"),
        "evidence": [
            "scripts/desktop/smoke_runtime.py",
            "apps/llm-studio/api/tests/test_main.py",
        ],
        "representative": {
            "method": "GET",
            "path": "/api/v1/health",
            "request_json": None,
            "response_status": "200",
            "response_json_subset": {
                "ready": True,
                "desktop_mode": True,
                "api_contract_version": "1",
            },
        },
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Replace the frozen catalog with the current authoritative API contract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog_path = args.catalog.expanduser().resolve()
    catalog = build_catalog(load_application_openapi())
    validate_catalog(catalog)
    if args.update:
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(
            json.dumps(catalog, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Updated frozen desktop API contract catalog: {catalog_path}")
        return
    if not catalog_path.is_file():
        raise SystemExit(
            f"Frozen desktop API contract catalog is missing: {catalog_path}. "
            "Run with --update after reviewing the API change."
        )
    frozen = json.loads(catalog_path.read_text(encoding="utf-8"))
    if frozen != catalog:
        raise SystemExit(
            "Desktop API contract drifted from the reviewed frozen catalog. "
            "Review the request/response change, then run "
            "scripts/desktop/validate_api_contract_fixtures.py --update."
        )
    print(
        "Frozen desktop API contract catalog passed: "
        f"{catalog['operation_count']} operations across {len(catalog['groups'])} groups."
    )


def load_application_openapi() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="llm-studio-api-contract-") as temporary:
        root = Path(temporary)
        environment = {
            "LLM_STUDIO_DATA_DIR": str(root / "data"),
            "LLM_STUDIO_CACHE_DIR": str(root / "cache"),
            "LLM_STUDIO_LOG_DIR": str(root / "logs"),
        }
        cleared_keys = (
            "LLM_STUDIO_DESKTOP",
            "LLM_STUDIO_RUNTIME_TOKEN",
            "LLM_STUDIO_RUNPOD_API_KEY",
        )
        changed_keys = (*cleared_keys, *environment)
        previous = {key: os.environ.get(key) for key in changed_keys}
        for key in cleared_keys:
            os.environ.pop(key, None)
        os.environ.update(environment)
        sys.path.insert(0, str(API_DIR))
        try:
            from app.main import app

            return app.openapi()
        finally:
            sys.path.remove(str(API_DIR))
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def build_catalog(openapi: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {group["id"]: [] for group in GROUPS}
    unclassified: list[str] = []
    for path, path_item in sorted(openapi.get("paths", {}).items()):
        for method, operation in sorted(path_item.items()):
            if method.lower() not in HTTP_METHODS:
                continue
            group = next((item for item in GROUPS if item["matches"](path)), None)
            if group is None:
                unclassified.append(f"{method.upper()} {path}")
                continue
            grouped[group["id"]].append(
                {
                    "method": method.upper(),
                    "path": path,
                    "operation": deepcopy(operation),
                }
            )
    if unclassified:
        raise ValueError(
            "Unclassified API operations: " + ", ".join(unclassified)
        )

    groups = []
    for group in GROUPS:
        operations = grouped[group["id"]]
        if not operations:
            raise ValueError(f"API contract group has no operations: {group['id']}")
        groups.append(
            {
                "id": group["id"],
                "title": group["title"],
                "evidence": group["evidence"],
                "representative": group["representative"],
                "operations": operations,
            }
        )
    return {
        "schema_version": 1,
        "api_contract_version": "1",
        "description": (
            "Frozen, sanitized request/response schema fixtures generated from "
            "FastAPI OpenAPI. Representative values contain no credentials or user data."
        ),
        "operation_count": sum(len(group["operations"]) for group in groups),
        "groups": groups,
        "components": {"schemas": deepcopy(openapi.get("components", {}).get("schemas", {}))},
    }


def validate_catalog(catalog: dict[str, Any]) -> None:
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported API contract catalog schema.")
    expected_groups = {group["id"] for group in GROUPS}
    actual_groups = {group["id"] for group in catalog.get("groups", [])}
    if actual_groups != expected_groups:
        raise ValueError(
            f"API contract groups do not match the parity matrix: {sorted(actual_groups)}"
        )
    operations = {
        (operation["method"], operation["path"]): operation["operation"]
        for group in catalog["groups"]
        for operation in group["operations"]
    }
    if len(operations) != catalog.get("operation_count"):
        raise ValueError("API contract catalog has duplicate or miscounted operations.")
    for group in catalog["groups"]:
        for evidence in group["evidence"]:
            if not (ROOT / evidence).is_file():
                raise ValueError(f"API contract evidence file is missing: {evidence}")
        validate_representative(catalog, operations, group["representative"])
    reject_concrete_secrets(catalog)


def validate_representative(
    catalog: dict[str, Any],
    operations: dict[tuple[str, str], dict[str, Any]],
    representative: dict[str, Any],
) -> None:
    key = (representative["method"], representative["path"])
    operation = operations.get(key)
    if operation is None:
        raise ValueError(f"Representative API operation is missing: {key[0]} {key[1]}")
    request = representative.get("request_json")
    if request is not None:
        request_schema = content_schema(operation.get("requestBody", {}))
        validate_top_level_keys(catalog, request_schema, request, f"{key[0]} {key[1]} request")
    status = representative["response_status"]
    response = operation.get("responses", {}).get(status)
    if response is None:
        raise ValueError(f"Representative response status is missing: {key[0]} {key[1]} {status}")
    response_subset = representative.get("response_json_subset")
    if response_subset is not None:
        response_schema = content_schema(response)
        validate_top_level_keys(
            catalog,
            response_schema,
            response_subset,
            f"{key[0]} {key[1]} response",
        )


def content_schema(container: dict[str, Any]) -> dict[str, Any]:
    content = container.get("content", {})
    for media_type in ("application/json", "application/x-ndjson"):
        if media_type in content:
            return content[media_type].get("schema", {})
    raise ValueError("Representative request/response has no JSON schema.")


def validate_top_level_keys(
    catalog: dict[str, Any],
    schema: dict[str, Any],
    value: dict[str, Any],
    label: str,
) -> None:
    resolved = resolve_schema(catalog, schema)
    properties = resolved.get("properties", {})
    unknown = sorted(set(value) - set(properties))
    if unknown:
        raise ValueError(f"{label} contains keys absent from its schema: {unknown}")


def resolve_schema(catalog: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    current = schema
    seen: set[str] = set()
    while "$ref" in current:
        reference = current["$ref"]
        if reference in seen or not reference.startswith("#/components/schemas/"):
            raise ValueError(f"Unsupported or cyclic schema reference: {reference}")
        seen.add(reference)
        name = reference.rsplit("/", 1)[-1]
        current = catalog["components"]["schemas"][name]
    return current


def reject_concrete_secrets(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            reject_concrete_secrets(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_concrete_secrets(item, f"{path}[{index}]")
    elif isinstance(value, str):
        for pattern in CONCRETE_SECRET_PATTERNS:
            if pattern.search(value):
                raise ValueError(f"Concrete secret found in API contract catalog at {path}.")


if __name__ == "__main__":
    main()
