# Tokenizer Studio Desktop Troubleshooting

## Backend Fails To Start

Checks:

1. Confirm runtime bundle exists (`runtime/python`, `runtime/app`, `runtime/web`, `runtime/VERSION`).
2. Confirm shell can find runtime:
   - set `TOKENIZER_STUDIO_RUNTIME_DIR` explicitly if needed.
3. Check latest log file:
   - `<app_data>/logs/backend-*.log`

## Health Check Timeout

Symptoms:

- Splash screen shows startup timeout.

Likely causes:

- Missing Python dependencies inside runtime.
- Invalid runtime `web/` path.
- Incompatible platform/runtime architecture.

Actions:

1. Rebuild runtime: `apps/tokenizer-studio/desktop/scripts/build_runtime_<platform>.sh`.
2. Run smoke test: `apps/tokenizer-studio/desktop/scripts/smoke_test_runtime.sh <platform>`.
3. Verify `python -m app.serve` works from `runtime/app`.

## Port Conflicts

The desktop shell selects a dynamic localhost port. If local firewall or endpoint security blocks loopback HTTP, sidecar health checks can fail.

Actions:

1. Allow localhost loopback traffic for the desktop app.
2. Review endpoint security policies for local process-to-process HTTP.

## Missing Artifacts or Job History

Data is stored in OS app-data path, not repository directories.

Default locations:

- macOS: `~/Library/Application Support/TokenizerStudio`
- Windows: `%AppData%\TokenizerStudio`
- Linux: `~/.local/share/TokenizerStudio`
