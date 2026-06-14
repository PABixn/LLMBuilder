from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from app.main import app

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "desktop" / "validate_api_contract_fixtures.py"
CATALOG = ROOT / "docs" / "llm-studio-desktop-api-contract-fixtures.json"
SPEC = importlib.util.spec_from_file_location("desktop_api_contract_fixtures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
api_contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api_contract)


def test_frozen_api_contract_catalog_matches_authoritative_openapi() -> None:
    generated = api_contract.build_catalog(app.openapi())
    api_contract.validate_catalog(generated)

    assert json.loads(CATALOG.read_text(encoding="utf-8")) == generated
    assert generated["operation_count"] >= 40
    assert {group["id"] for group in generated["groups"]} == {
        "system-config",
        "model-studio",
        "tokenizer",
        "training-config-preflight",
        "runpod-provider",
        "training-jobs",
        "remote-lifecycle",
        "inference",
    }


def test_api_contract_catalog_rejects_unclassified_operation_and_concrete_secret() -> None:
    openapi = app.openapi()
    openapi["paths"]["/api/v1/unclassified"] = {
        "get": {
            "responses": {
                "200": {
                    "description": "ok",
                    "content": {"application/json": {"schema": {"type": "object"}}},
                }
            }
        }
    }
    with pytest.raises(ValueError, match="Unclassified API operations"):
        api_contract.build_catalog(openapi)

    with pytest.raises(ValueError, match="Concrete secret"):
        api_contract.reject_concrete_secrets(
            {"accidental": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345"}
        )
