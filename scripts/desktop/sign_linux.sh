#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <path-to-appimage-or-deb>" >&2
  exit 1
fi

ARTIFACT_PATH="$1"
: "${LINUX_GPG_KEY_ID:?LINUX_GPG_KEY_ID is required}"

gpg --batch --yes --armor --detach-sign --local-user "$LINUX_GPG_KEY_ID" "$ARTIFACT_PATH"
echo "Signature created: ${ARTIFACT_PATH}.asc"
