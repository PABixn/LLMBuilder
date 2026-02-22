#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <path-to-msi-or-exe>" >&2
  exit 1
fi

ARTIFACT_PATH="$1"
: "${WINDOWS_CERT_FILE:?WINDOWS_CERT_FILE is required}"
: "${WINDOWS_CERT_PASSWORD:?WINDOWS_CERT_PASSWORD is required}"

signtool sign \
  /f "$WINDOWS_CERT_FILE" \
  /p "$WINDOWS_CERT_PASSWORD" \
  /tr http://timestamp.digicert.com \
  /td sha256 \
  /fd sha256 \
  "$ARTIFACT_PATH"

echo "Signed: $ARTIFACT_PATH"
