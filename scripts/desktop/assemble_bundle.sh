#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$ROOT_DIR/build/desktop}"
PLATFORM_NAME="${1:-macos-arm64}"
RUNTIME_DIR="$BUILD_ROOT/runtime/$PLATFORM_NAME"
BUNDLE_OUT_DIR="$BUILD_ROOT/bundles/$PLATFORM_NAME"
RUNTIME_MANIFEST="$BUNDLE_OUT_DIR/runtime-manifest.json"

if [[ ! -d "$RUNTIME_DIR" ]]; then
  echo "Runtime directory does not exist: $RUNTIME_DIR" >&2
  echo "Run scripts/desktop/build_runtime_<platform>.sh first." >&2
  exit 1
fi

mkdir -p "$BUNDLE_OUT_DIR"

RUNTIME_VERSION="$(cat "$RUNTIME_DIR/VERSION" 2>/dev/null || echo "unknown")"
RUNTIME_ARCHIVE="$BUNDLE_OUT_DIR/tokenizer-studio-runtime-${PLATFORM_NAME}-${RUNTIME_VERSION}.tar.gz"
tar -czf "$RUNTIME_ARCHIVE" -C "$RUNTIME_DIR" .

if command -v shasum >/dev/null 2>&1; then
  RUNTIME_SHA256="$(shasum -a 256 "$RUNTIME_ARCHIVE" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
  RUNTIME_SHA256="$(sha256sum "$RUNTIME_ARCHIVE" | awk '{print $1}')"
else
  echo "No SHA-256 tool found (expected shasum or sha256sum)." >&2
  exit 1
fi

cat > "$RUNTIME_MANIFEST" <<EOF
{
  "app": "TokenizerStudio",
  "platform": "${PLATFORM_NAME}",
  "runtime_version": "${RUNTIME_VERSION}",
  "runtime_archive": "$(basename "$RUNTIME_ARCHIVE")",
  "sha256": "${RUNTIME_SHA256}"
}
EOF

echo "Runtime bundle archive: $RUNTIME_ARCHIVE"
echo "Runtime manifest: $RUNTIME_MANIFEST"
