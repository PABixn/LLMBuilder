from __future__ import annotations

import importlib
from pathlib import Path

from fastapi.testclient import TestClient

from app import config


def _load_app_module() -> object:
    import app.main as main_module

    return importlib.reload(main_module)


def test_health_and_static_fallback(monkeypatch, tmp_path: Path) -> None:
    web_dist = tmp_path / "web-dist"
    web_dist.mkdir(parents=True, exist_ok=True)
    (web_dist / "index.html").write_text(
        "<html><body><h1>Tokenizer Studio</h1></body></html>",
        encoding="utf-8",
    )
    (web_dist / "asset.txt").write_text("asset-ok", encoding="utf-8")

    monkeypatch.setenv("TOKENIZER_STUDIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TOKENIZER_STUDIO_WEB_DIST_DIR", str(web_dist))
    monkeypatch.setenv("TOKENIZER_STUDIO_SERVE_WEB", "1")
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
        assert "Tokenizer Studio" in index.text

        asset = client.get("/asset.txt")
        assert asset.status_code == 200
        assert asset.text == "asset-ok"

        spa_fallback = client.get("/deep/link/that/does/not/exist")
        assert spa_fallback.status_code == 200
        assert "Tokenizer Studio" in spa_fallback.text

        missing_api = client.get("/api/v1/unknown-route")
        assert missing_api.status_code == 404


def test_local_file_stats_endpoint_counts_chars(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    dataset_file = data_dir / "datasets" / "sample.txt"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    content = "ab\né🙂"
    dataset_file.write_text(content, encoding="utf-8")

    monkeypatch.setenv("TOKENIZER_STUDIO_DATA_DIR", str(data_dir))
    config.reset_settings_cache()

    module = _load_app_module()

    with TestClient(module.app) as client:
        response = client.get(
          "/api/v1/files/stats",
          params={"file_path": str(dataset_file)},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["file_name"] == dataset_file.name
        assert body["size_chars"] == len(content)
        assert body["size_bytes"] == len(content.encode("utf-8"))
