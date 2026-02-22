#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEB_DIR="$ROOT_DIR/apps/tokenizer-studio/web"
API_DIR="$ROOT_DIR/apps/tokenizer-studio/api"
WEB_DIST_TARGET="${1:-$API_DIR/web-dist}"

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

pushd "$WEB_DIR" >/dev/null
if [[ "${TOKENIZER_STUDIO_SKIP_NPM_CI:-0}" != "1" ]]; then
  npm ci
fi
TOKENIZER_STUDIO_DESKTOP_BUILD=1 npm run build:desktop
popd >/dev/null

copy_tree "$WEB_DIR/out" "$WEB_DIST_TARGET"
echo "Desktop web assets copied to $WEB_DIST_TARGET"
