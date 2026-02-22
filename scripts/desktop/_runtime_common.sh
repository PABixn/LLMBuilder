#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
API_DIR="$ROOT_DIR/apps/tokenizer-studio/api"
WEB_DIST_DIR="${WEB_DIST_DIR:-$API_DIR/web-dist}"
BUILD_ROOT="${BUILD_ROOT:-$ROOT_DIR/build/desktop}"
RUNTIME_VERSION="${RUNTIME_VERSION:-0.1.0-dev}"

copy_tree() {
  local source_dir="$1"
  local target_dir="$2"

  rm -rf "$target_dir"
  mkdir -p "$target_dir"

  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$source_dir"/ "$target_dir"/
    return
  fi

  cp -R "$source_dir"/. "$target_dir"/
}

build_runtime_bundle() {
  local platform_name="$1"
  local runtime_dir="$BUILD_ROOT/runtime/$platform_name"
  local python_bin="${PYTHON_BIN:-python3}"

  rm -rf "$runtime_dir"
  mkdir -p "$runtime_dir"

  "$python_bin" -m venv "$runtime_dir/python"

  local runtime_python=""
  if [[ -x "$runtime_dir/python/bin/python" ]]; then
    runtime_python="$runtime_dir/python/bin/python"
  elif [[ -x "$runtime_dir/python/Scripts/python.exe" ]]; then
    runtime_python="$runtime_dir/python/Scripts/python.exe"
  else
    echo "Failed to locate runtime python executable in $runtime_dir/python" >&2
    exit 1
  fi

  "$runtime_python" -m pip install --upgrade pip setuptools wheel
  "$runtime_python" -m pip install --no-cache-dir -r "$API_DIR/requirements.txt"

  mkdir -p "$runtime_dir/app" "$runtime_dir/web" "$runtime_dir/wheelhouse"

  copy_tree "$API_DIR/app" "$runtime_dir/app/app"
  copy_tree "$API_DIR/templates" "$runtime_dir/app/templates"
  copy_tree "$ROOT_DIR/tokenizer" "$runtime_dir/app/tokenizer"

  cp "$API_DIR/requirements.txt" "$runtime_dir/app/requirements.txt"

  if [[ -d "$WEB_DIST_DIR" ]]; then
    copy_tree "$WEB_DIST_DIR" "$runtime_dir/web"
  else
    echo "Warning: web build output not found at $WEB_DIST_DIR" >&2
    echo "Run scripts/desktop/build_web.sh first." >&2
  fi

  printf "%s\n" "$RUNTIME_VERSION" > "$runtime_dir/VERSION"
  echo "Runtime bundle created at $runtime_dir"
}
