.PHONY: help api api-prod web web-build install-api install-web install api-compile api-test api-smoke web-typecheck check

ROOT_DIR := $(abspath $(CURDIR)/../..)
API_DIR := $(CURDIR)/api
WEB_DIR := $(CURDIR)/web
ROOT_VENV_PY := $(ROOT_DIR)/.venv/bin/python3
API_VENV_PY := $(API_DIR)/.venv/bin/python3
PYTHON_BIN ?= $(if $(wildcard $(ROOT_VENV_PY)),$(ROOT_VENV_PY),$(if $(wildcard $(API_VENV_PY)),$(API_VENV_PY),python3))

help:
	@echo "LLM Studio targets:"
	@echo "  make api           - Run FastAPI in reload mode on port 8000"
	@echo "  make api-prod      - Run production API server (app.serve)"
	@echo "  make web           - Run Next.js dev server"
	@echo "  make web-build     - Build web app for desktop embedding/static export flow"
	@echo "  make install-api   - Install API Python requirements"
	@echo "  make install-web   - Install web npm dependencies"
	@echo "  make install       - Install both API and web dependencies"
	@echo "  make api-compile   - Syntax-check API modules with py_compile"
	@echo "  make api-test      - Run API pytest suite"
	@echo "  make api-smoke     - Smoke-check validate/analyze endpoints"
	@echo "  make web-typecheck - Run TypeScript typecheck"
	@echo "  make check         - Run API compile + web typecheck"

api:
	@echo "Using Python interpreter: $(PYTHON_BIN)"
	cd $(API_DIR) && $(PYTHON_BIN) -m uvicorn app.main:app --reload --port 8000

api-prod:
	@echo "Using Python interpreter: $(PYTHON_BIN)"
	cd $(API_DIR) && $(PYTHON_BIN) -m app.serve

web:
	cd $(WEB_DIR) && npm run dev

web-build:
	cd $(WEB_DIR) && npm run build:desktop

install-api:
	cd $(API_DIR) && $(PYTHON_BIN) -m pip install -r requirements.txt

install-web:
	cd $(WEB_DIR) && npm install

install: install-api install-web

api-compile:
	cd $(API_DIR) && $(PYTHON_BIN) -m py_compile app/main.py app/models.py app/config.py app/schemas.py app/serve.py

api-test:
	cd $(API_DIR) && $(PYTHON_BIN) -m pytest -q

api-smoke:
	cd $(API_DIR) && LLM_STUDIO_DATA_DIR=$$(pwd)/.tmp-smoke $(PYTHON_BIN) -c 'import importlib; from pathlib import Path; from fastapi.testclient import TestClient; import app.config as config; from app.schemas import load_json; config.reset_settings_cache(); import app.main as main; main = importlib.reload(main); cfg = load_json(Path("../../..") / "model" / "gpt2_config.json"); client = TestClient(main.app); assert client.get("/api/v1/health").status_code == 200; assert client.post("/api/v1/validate/model", json={"config": cfg}).status_code == 200; analyze = client.post("/api/v1/analyze/model", json={"config": cfg}); assert analyze.status_code == 200; body = analyze.json(); assert body["instantiated"] is True; assert body["analysis"]["total_parameters"] > 0; print("LLM Studio API smoke passed.")'

web-typecheck:
	cd $(WEB_DIR) && npm run typecheck

check: api-compile web-typecheck
