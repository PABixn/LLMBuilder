#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$ROOT_DIR/build/desktop}"
PLATFORM_NAME="${1:-macos-arm64}"
RUNTIME_DIR="$BUILD_ROOT/runtime/$PLATFORM_NAME"

if [[ ! -d "$RUNTIME_DIR" ]]; then
  echo "Runtime directory does not exist: $RUNTIME_DIR" >&2
  exit 1
fi

if [[ -x "$RUNTIME_DIR/python/bin/python" ]]; then
  PYTHON_BIN="$RUNTIME_DIR/python/bin/python"
elif [[ -x "$RUNTIME_DIR/python/Scripts/python.exe" ]]; then
  PYTHON_BIN="$RUNTIME_DIR/python/Scripts/python.exe"
else
  echo "Could not locate runtime python executable." >&2
  exit 1
fi

PORT="$("$PYTHON_BIN" - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"

TMP_DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tokenizer-studio-smoke.XXXXXX")"
LOG_FILE="$TMP_DATA_DIR/runtime-smoke.log"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
    wait "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
  rm -rf "$TMP_DATA_DIR"
}
trap cleanup EXIT

pushd "$RUNTIME_DIR/app" >/dev/null
TOKENIZER_STUDIO_HOST=127.0.0.1 \
TOKENIZER_STUDIO_PORT="$PORT" \
TOKENIZER_STUDIO_DATA_DIR="$TMP_DATA_DIR" \
TOKENIZER_STUDIO_WEB_DIST_DIR="$RUNTIME_DIR/web" \
"$PYTHON_BIN" -m app.serve >"$LOG_FILE" 2>&1 &
BACKEND_PID="$!"
popd >/dev/null

echo "Started runtime backend for smoke test on port $PORT (pid=$BACKEND_PID)"

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null
curl -fsS "http://127.0.0.1:$PORT/api/v1/health" >/dev/null
echo "Runtime smoke test passed."
