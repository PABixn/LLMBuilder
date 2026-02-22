#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_runtime_common.sh"

PLATFORM_NAME="${1:-linux-x64}"
build_runtime_bundle "$PLATFORM_NAME"
