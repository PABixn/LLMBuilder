.PHONY: api web

ROOT_DIR := $(abspath $(CURDIR)/../..)
API_DIR := $(CURDIR)/api
ROOT_VENV_PY := $(ROOT_DIR)/.venv/bin/python3
API_VENV_PY := $(API_DIR)/.venv/bin/python3
PYTHON_BIN ?= $(if $(wildcard $(ROOT_VENV_PY)),$(ROOT_VENV_PY),$(if $(wildcard $(API_VENV_PY)),$(API_VENV_PY),python3))

api:
	@echo "Using Python interpreter: $(PYTHON_BIN)"
	cd api && $(PYTHON_BIN) -m uvicorn app.main:app --reload --port 8000

web:
	cd web && npm run dev
