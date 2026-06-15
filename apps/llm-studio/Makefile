.PHONY: help api api-prod web web-build install-api install-web install install-desktop api-compile api-test api-smoke web-typecheck check desktop-check desktop-versions desktop-check-api-contract desktop-web-quality desktop-build-web-offline desktop-build-runtime desktop-build-release-runtime desktop-smoke-runtime desktop-test-shell desktop-build-shell desktop-audit-dependencies desktop-verify-nonbrowser desktop-test-e2e desktop-dev desktop-package desktop-release-manifest desktop-audit-payload

ROOT_DIR := $(abspath $(CURDIR)/../..)
API_DIR := $(CURDIR)/api
WEB_DIR := $(CURDIR)/web
DESKTOP_DIR := $(CURDIR)/desktop
DESKTOP_SCRIPTS := $(ROOT_DIR)/scripts/desktop
ROOT_VENV_PY := $(ROOT_DIR)/.venv/bin/python3
API_VENV_PY := $(API_DIR)/.venv/bin/python3
PYTHON_BIN ?= $(if $(wildcard $(ROOT_VENV_PY)),$(ROOT_VENV_PY),$(if $(wildcard $(API_VENV_PY)),$(API_VENV_PY),python3))
DESKTOP_TARGET := $(shell $(PYTHON_BIN) -c 'import platform; print(("macos" if platform.system().lower()=="darwin" else platform.system().lower()) + "-" + ("aarch64" if platform.machine().lower()=="arm64" else "x86_64" if platform.machine().lower()=="amd64" else platform.machine().lower()))')
DESKTOP_RUNTIME ?= $(ROOT_DIR)/build/desktop/runtime/$(DESKTOP_TARGET)
DESKTOP_PERFORMANCE_REPORT ?= $(ROOT_DIR)/build/desktop/release/performance-characterization-$(DESKTOP_TARGET).json
DESKTOP_SHELL_BINARY := $(DESKTOP_DIR)/src-tauri/target/release/llm_studio_desktop$(if $(findstring windows,$(DESKTOP_TARGET)),.exe,)

help:
	@echo "LLM Builder Studio targets:"
	@echo "  make api           - Run FastAPI in reload mode on port 8000"
	@echo "  make api-prod      - Run production API server (app.serve)"
	@echo "  make web           - Run Next.js dev server"
	@echo "  make web-build     - Build web app for desktop embedding/static export flow"
	@echo "  make install-api   - Install API Python requirements"
	@echo "  make install-web   - Install web npm dependencies"
	@echo "  make install       - Install both API and web dependencies"
	@echo "  make install-desktop - Install locked desktop shell npm dependencies"
	@echo "  make api-compile   - Syntax-check API modules with py_compile"
	@echo "  make api-test      - Run API pytest suite"
	@echo "  make api-smoke     - Smoke-check model + tokenizer endpoints"
	@echo "  make web-typecheck - Run TypeScript typecheck"
	@echo "  make check         - Run API compile + web typecheck"
	@echo "  make desktop-check - Validate desktop toolchains and lockfiles"
	@echo "  make desktop-versions - Print desktop/runtime version and schema information"
	@echo "  make desktop-check-api-contract - Validate the frozen API request/response contract catalog"
	@echo "  make desktop-web-quality - Run web lint, typecheck, and non-browser regressions"
	@echo "  make desktop-build-web-offline - Build and validate static frontend without network"
	@echo "  make desktop-build-runtime - Build a linked target-native development runtime"
	@echo "  make desktop-build-release-runtime DESKTOP_LOCK=... DESKTOP_WHEELHOUSE=... - Build reviewed portable runtime"
	@echo "  make desktop-smoke-runtime - Smoke and characterize the authenticated packaged runtime"
	@echo "  make desktop-test-shell - Format, lint, and test the Tauri shell"
	@echo "  make desktop-build-shell - Compile the release shell without creating an installer"
	@echo "  make desktop-audit-dependencies - Audit Python and Rust dependencies with reviewed policy"
	@echo "  make desktop-verify-nonbrowser - Run the complete local non-browser desktop verification"
	@echo "  make desktop-dev - Run Tauri against the built development runtime"
	@echo "  make desktop-package - Stage runtime and build a platform-native installer"
	@echo "  make desktop-release-manifest ARTIFACTS='...' - Generate release manifest and checksums"
	@echo "  make desktop-audit-payload PAYLOADS='...' - Audit extracted release payloads"

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

install-desktop:
	cd $(DESKTOP_DIR) && npm ci

api-compile:
	cd $(API_DIR) && $(PYTHON_BIN) -m compileall -q app

api-test:
	cd $(API_DIR) && $(PYTHON_BIN) -m pytest -q

api-smoke:
	cd $(API_DIR) && LLM_STUDIO_DATA_DIR=$$(pwd)/.tmp-smoke $(PYTHON_BIN) -c 'import importlib; from pathlib import Path; from fastapi.testclient import TestClient; import app.config as config; from app.schemas import load_json; config.reset_settings_cache(); import app.main as main; main = importlib.reload(main); cfg = load_json(Path("../../..") / "model" / "gpt2_config.json"); client = TestClient(main.app); assert client.get("/api/v1/health").status_code == 200; assert client.post("/api/v1/validate/model", json={"config": cfg}).status_code == 200; analyze = client.post("/api/v1/analyze/model", json={"config": cfg}); assert analyze.status_code == 200; body = analyze.json(); assert body["instantiated"] is True; assert body["analysis"]["total_parameters"] > 0; tokenizer_templates = client.get("/api/v1/tokenizer/config/templates"); assert tokenizer_templates.status_code == 200; tokenizer_validate = client.post("/api/v1/tokenizer/validate/tokenizer", json={"config": tokenizer_templates.json()["tokenizer_config_template"]}); assert tokenizer_validate.status_code == 200; print("LLM Builder Studio API smoke passed.")'

web-typecheck:
	cd $(WEB_DIR) && npm run typecheck

check: api-compile web-typecheck

desktop-check:
	$(PYTHON_BIN) $(DESKTOP_SCRIPTS)/check_toolchains.py

desktop-versions:
	$(PYTHON_BIN) $(DESKTOP_SCRIPTS)/print_versions.py --runtime $(DESKTOP_RUNTIME)

desktop-check-api-contract:
	$(PYTHON_BIN) $(DESKTOP_SCRIPTS)/validate_api_contract_fixtures.py

desktop-web-quality:
	cd $(WEB_DIR) && npm run lint && npm run typecheck && npm run test:regression

desktop-build-web-offline:
	cd $(WEB_DIR) && npm_config_offline=true npm run build:desktop

desktop-build-runtime:
	$(PYTHON_BIN) $(DESKTOP_SCRIPTS)/build_runtime.py --python $(PYTHON_BIN) --output $(DESKTOP_RUNTIME)

desktop-build-release-runtime:
	@test -n "$(DESKTOP_LOCK)" || (echo "DESKTOP_LOCK must point to a reviewed hashed lock" && exit 2)
	@test -n "$(DESKTOP_WHEELHOUSE)" || (echo "DESKTOP_WHEELHOUSE must point to a reviewed offline wheelhouse" && exit 2)
	$(PYTHON_BIN) $(DESKTOP_SCRIPTS)/build_runtime.py --portable --install-dependencies --lock $(DESKTOP_LOCK) --wheelhouse $(DESKTOP_WHEELHOUSE) --output $(DESKTOP_RUNTIME)

desktop-smoke-runtime:
	$(PYTHON_BIN) $(DESKTOP_SCRIPTS)/smoke_runtime.py $(DESKTOP_RUNTIME) --performance-report $(DESKTOP_PERFORMANCE_REPORT)

desktop-test-shell:
	cd $(DESKTOP_DIR) && npm run fmt:check && npm run clippy && npm test

desktop-build-shell:
	$(PYTHON_BIN) $(DESKTOP_SCRIPTS)/build_shell.py --no-bundle

desktop-audit-dependencies:
	$(PYTHON_BIN) $(DESKTOP_SCRIPTS)/audit_dependencies.py --python $(PYTHON_BIN) --cargo-binary $(DESKTOP_SHELL_BINARY) --report $(ROOT_DIR)/build/desktop/release/dependency-audit.json

desktop-verify-nonbrowser: desktop-check desktop-check-api-contract api-test desktop-web-quality desktop-build-web-offline desktop-build-runtime desktop-smoke-runtime desktop-test-shell desktop-build-shell desktop-audit-dependencies
	$(PYTHON_BIN) $(DESKTOP_SCRIPTS)/audit_release_payload.py $(WEB_DIR)/out $(DESKTOP_SHELL_BINARY)
	cd $(WEB_DIR) && npm audit --audit-level=high
	cd $(DESKTOP_DIR) && npm audit --audit-level=high
	$(PYTHON_BIN) -m pip check
	@if $(PYTHON_BIN) $(DESKTOP_SCRIPTS)/stage_runtime.py $(DESKTOP_RUNTIME) >/dev/null 2>&1; then echo "ERROR: linked development runtime unexpectedly passed release staging"; exit 1; else echo "Release staging correctly rejected linked development runtime."; fi
	git -C $(ROOT_DIR) diff --check

desktop-test-e2e:
	cd $(WEB_DIR) && npm run lint && npm run typecheck && npm run test:regression && npm run build:desktop
	cd $(WEB_DIR) && npm run test:e2e
	$(MAKE) desktop-smoke-runtime

desktop-dev:
	cd $(DESKTOP_DIR) && LLM_STUDIO_RUNTIME_DIR=$(DESKTOP_RUNTIME) npm run dev

desktop-package:
	$(PYTHON_BIN) $(DESKTOP_SCRIPTS)/stage_runtime.py $(DESKTOP_RUNTIME)
	$(PYTHON_BIN) $(DESKTOP_SCRIPTS)/build_shell.py

desktop-release-manifest:
	@test -n "$(ARTIFACTS)" || (echo "ARTIFACTS must list one or more release artifact files" && exit 2)
	$(PYTHON_BIN) $(DESKTOP_SCRIPTS)/release_manifest.py $(ARTIFACTS)

desktop-audit-payload:
	@test -n "$(PAYLOADS)" || (echo "PAYLOADS must list one or more extracted release payloads" && exit 2)
	$(PYTHON_BIN) $(DESKTOP_SCRIPTS)/audit_release_payload.py $(PAYLOADS)
