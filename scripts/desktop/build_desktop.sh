#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/scripts/desktop"
DESKTOP_DIR="$ROOT_DIR/apps/tokenizer-studio/desktop"
BUILD_ROOT="${BUILD_ROOT:-$ROOT_DIR/build/desktop}"
PLATFORM_NAME="${1:-macos-arm64}"

case "$PLATFORM_NAME" in
  macos-*)
    RUNTIME_SCRIPT="$SCRIPT_DIR/build_runtime_macos.sh"
    ;;
  linux-*)
    RUNTIME_SCRIPT="$SCRIPT_DIR/build_runtime_linux.sh"
    ;;
  windows-*)
    RUNTIME_SCRIPT="$SCRIPT_DIR/build_runtime_windows.sh"
    ;;
  *)
    echo "Unsupported platform: $PLATFORM_NAME" >&2
    echo "Use one of: macos-arm64, linux-x64, windows-x64" >&2
    exit 1
    ;;
esac

embed_macos_runtime_in_app_bundle() {
  local runtime_dir="$BUILD_ROOT/runtime/$PLATFORM_NAME"
  local tauri_macos_bundle_dir="$DESKTOP_DIR/src-tauri/target/release/bundle/macos"
  local app_bundle
  local bundled_runtime_dir

  if [[ ! -d "$runtime_dir" ]]; then
    echo "Runtime directory missing; skipping macOS runtime embedding: $runtime_dir" >&2
    return
  fi

  app_bundle="$(find "$tauri_macos_bundle_dir" -maxdepth 1 -type d -name "*.app" | head -n 1 || true)"
  if [[ -z "$app_bundle" ]]; then
    echo "macOS app bundle not found; skipping runtime embedding in $tauri_macos_bundle_dir" >&2
    return
  fi

  bundled_runtime_dir="$app_bundle/Contents/Resources/runtime"
  rm -rf "$bundled_runtime_dir"
  mkdir -p "$bundled_runtime_dir"

  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$runtime_dir"/ "$bundled_runtime_dir"/
  else
    cp -R "$runtime_dir"/. "$bundled_runtime_dir"/
  fi

  echo "Embedded runtime into app bundle: $bundled_runtime_dir"
}

stage_runtime_for_tauri_resources() {
  local runtime_dir="$BUILD_ROOT/runtime/$PLATFORM_NAME"
  local tauri_resources_runtime_dir="$DESKTOP_DIR/src-tauri/resources/runtime"

  if [[ ! -d "$runtime_dir" ]]; then
    echo "Runtime directory missing; skipping Tauri resource staging: $runtime_dir" >&2
    return
  fi

  rm -rf "$tauri_resources_runtime_dir"
  mkdir -p "$tauri_resources_runtime_dir"

  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$runtime_dir"/ "$tauri_resources_runtime_dir"/
  else
    cp -R "$runtime_dir"/. "$tauri_resources_runtime_dir"/
  fi

  echo "Staged runtime for Tauri bundle resources: $tauri_resources_runtime_dir"
}

"$SCRIPT_DIR/build_web.sh"
"$RUNTIME_SCRIPT" "$PLATFORM_NAME"
stage_runtime_for_tauri_resources

pushd "$DESKTOP_DIR" >/dev/null
if [[ "${TOKENIZER_STUDIO_SKIP_NPM_INSTALL:-0}" != "1" ]]; then
  npm install
fi
if [[ "${TOKENIZER_STUDIO_SKIP_ICON_REFRESH:-0}" != "1" ]]; then
  npm run icons
fi
npm run tauri build
popd >/dev/null

if [[ "$PLATFORM_NAME" == macos-* ]]; then
  embed_macos_runtime_in_app_bundle
fi

"$SCRIPT_DIR/assemble_bundle.sh" "$PLATFORM_NAME"
echo "Desktop build pipeline completed for $PLATFORM_NAME"
