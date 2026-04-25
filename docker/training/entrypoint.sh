#!/usr/bin/env bash
set -euo pipefail

workspace="${LLM_STUDIO_REMOTE_WORKSPACE:-/workspace/llm-studio}"
port="${LLM_STUDIO_RUNPOD_AGENT_PORT:-8021}"
startup_log="${workspace}/logs/startup.log"

mkdir -p "${workspace}/logs"

log_event() {
  local event="$1"
  local timestamp
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  local line
  line="{\"timestamp\":\"${timestamp}\",\"event\":\"${event}\",\"job_id\":\"${LLM_STUDIO_REMOTE_JOB_ID:-}\",\"service\":\"llm-studio-training-entrypoint\",\"workspace\":\"${workspace}\",\"port\":\"${port}\"}"
  printf '[llm-studio-entrypoint] %s\n' "${line}" | tee -a "${startup_log}"
}

on_error() {
  local status="$?"
  log_event "entrypoint_error_status_${status}"
  exit "${status}"
}

trap on_error ERR

log_event "entrypoint_start"
log_event "startup_diagnostics_start"
if python -m remote_agent.diagnostics startup; then
  log_event "startup_diagnostics_done"
else
  diagnostics_status="$?"
  log_event "startup_diagnostics_failed_status_${diagnostics_status}"
fi
log_event "uvicorn_start"

exec python -m uvicorn remote_agent.app:app --host 0.0.0.0 --port "${port}" --log-level "${LLM_STUDIO_AGENT_LOG_LEVEL:-info}"
