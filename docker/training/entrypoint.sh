#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${LLM_STUDIO_REMOTE_WORKSPACE:-/workspace/llm-studio}"
exec python -m uvicorn remote_agent.app:app --host 0.0.0.0 --port "${LLM_STUDIO_RUNPOD_AGENT_PORT:-8021}"
