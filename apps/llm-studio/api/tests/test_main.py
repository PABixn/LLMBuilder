from __future__ import annotations

import copy
import importlib
from pathlib import Path

from fastapi.testclient import TestClient

from app import config
from app.schemas import load_json


REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_TEMPLATE_PATH = REPO_ROOT / "model" / "gpt2_config.json"


def _load_app_module() -> object:
    import app.main as main_module

    return importlib.reload(main_module)


def test_health_and_static_fallback(monkeypatch, tmp_path: Path) -> None:
    web_dist = tmp_path / "web-dist"
    web_dist.mkdir(parents=True, exist_ok=True)
    (web_dist / "index.html").write_text(
        "<html><body><h1>LLM Studio</h1></body></html>",
        encoding="utf-8",
    )
    (web_dist / "asset.txt").write_text("asset-ok", encoding="utf-8")

    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LLM_STUDIO_WEB_DIST_DIR", str(web_dist))
    monkeypatch.setenv("LLM_STUDIO_SERVE_WEB", "1")
    config.reset_settings_cache()

    module = _load_app_module()

    with TestClient(module.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"ok": True}

        api_health = client.get("/api/v1/health")
        assert api_health.status_code == 200
        assert api_health.json() == {"ok": True}

        index = client.get("/")
        assert index.status_code == 200
        assert "LLM Studio" in index.text

        asset = client.get("/asset.txt")
        assert asset.status_code == 200
        assert asset.text == "asset-ok"

        spa_fallback = client.get("/deep/link/that/does/not/exist")
        assert spa_fallback.status_code == 200
        assert "LLM Studio" in spa_fallback.text

        missing_api = client.get("/api/v1/unknown-route")
        assert missing_api.status_code == 404


def test_validate_model_endpoint_reports_semantic_issues(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(tmp_path / "data"))
    config.reset_settings_cache()
    module = _load_app_module()
    base_config = load_json(MODEL_TEMPLATE_PATH)

    with TestClient(module.app) as client:
        warning_config = copy.deepcopy(base_config)
        warning_config["n_embd"] = 780  # 780 / 12 = 65 (odd head_dim)
        warning_resp = client.post("/api/v1/validate/model", json={"config": warning_config})
        assert warning_resp.status_code == 200
        warning_body = warning_resp.json()
        assert warning_body["valid"] is True
        warning_codes = {issue["code"] for issue in warning_body["warnings"]}
        assert "head_dim_not_even" in warning_codes
        assert warning_body["errors"] == []

        error_config = copy.deepcopy(base_config)
        error_config["blocks"][0]["components"][1]["attention"]["n_kv_head"] = 16
        error_resp = client.post("/api/v1/validate/model", json={"config": error_config})
        assert error_resp.status_code == 200
        error_body = error_resp.json()
        assert error_body["valid"] is False
        error_codes = {issue["code"] for issue in error_body["errors"]}
        assert "n_kv_head_gt_n_head" in error_codes
        assert "n_head_not_divisible_by_n_kv_head" in error_codes


def test_analyze_model_endpoint_instantiates_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(tmp_path / "data"))
    config.reset_settings_cache()
    module = _load_app_module()
    base_config = load_json(MODEL_TEMPLATE_PATH)

    with TestClient(module.app) as client:
        response = client.post("/api/v1/analyze/model", json={"config": base_config})
        assert response.status_code == 200
        body = response.json()
        assert body["valid"] is True
        assert body["instantiated"] is True
        assert body["instantiation_error"] is None
        assert body["analysis"]["total_parameters"] > 0
        assert body["analysis"]["block_count"] == len(base_config["blocks"])
        assert body["analysis"]["mlp_activation_step_count"] > 0
        breakdown = body["analysis"]["parameter_breakdown"]
        assert isinstance(breakdown, list)
        assert breakdown
        assert sum(item["parameters"] for item in breakdown) == body["analysis"]["total_parameters"]
        assert (
            sum(item["trainable_parameters"] for item in breakdown)
            == body["analysis"]["trainable_parameters"]
        )
        assert all(item["percentage"] >= 0 for item in breakdown)
        assert all(item["trainable_percentage"] >= 0 for item in breakdown)


def test_projects_endpoints_round_trip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(tmp_path / "data"))
    config.reset_settings_cache()
    module = _load_app_module()
    base_config = load_json(MODEL_TEMPLATE_PATH)

    with TestClient(module.app) as client:
        create = client.post(
            "/api/v1/projects",
            json={"name": "GPT-2 baseline", "model_config": base_config},
        )
        assert create.status_code == 201
        created = create.json()
        project_id = created["id"]
        assert created["name"] == "GPT-2 baseline"
        assert created["valid"] is True
        assert created["model_config"]["n_embd"] == base_config["n_embd"]

        listing = client.get("/api/v1/projects")
        assert listing.status_code == 200
        listed_ids = [item["id"] for item in listing.json()["projects"]]
        assert project_id in listed_ids

        detail = client.get(f"/api/v1/projects/{project_id}")
        assert detail.status_code == 200
        assert detail.json()["id"] == project_id

        artifact = client.get(f"/api/v1/projects/{project_id}/artifact")
        assert artifact.status_code == 200
        assert artifact.json()["n_embd"] == base_config["n_embd"]
