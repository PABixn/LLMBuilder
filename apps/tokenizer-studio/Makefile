.PHONY: api api-prod web web-build desktop-dev desktop-build

ROOT_DIR := $(abspath $(CURDIR)/../..)
API_DIR := $(CURDIR)/api
WEB_DIR := $(CURDIR)/web
DESKTOP_DIR := $(CURDIR)/desktop
ROOT_VENV_PY := $(ROOT_DIR)/.venv/bin/python3
API_VENV_PY := $(API_DIR)/.venv/bin/python3
PYTHON_BIN ?= $(if $(wildcard $(ROOT_VENV_PY)),$(ROOT_VENV_PY),$(if $(wildcard $(API_VENV_PY)),$(API_VENV_PY),python3))
DESKTOP_SCRIPT_DIR := $(ROOT_DIR)/scripts/desktop

api:
	@echo "Using Python interpreter: $(PYTHON_BIN)"
	cd api && $(PYTHON_BIN) -m uvicorn app.main:app --reload --port 8000

api-prod:
	@echo "Using Python interpreter: $(PYTHON_BIN)"
	cd api && $(PYTHON_BIN) -m app.serve

web:
	cd $(WEB_DIR) && npm run dev

web-build:
	cd $(WEB_DIR) && npm run build:desktop

desktop-dev:
	cd $(DESKTOP_DIR) && npm run tauri dev

desktop-build:
	$(DESKTOP_SCRIPT_DIR)/build_desktop.sh macos-arm64
