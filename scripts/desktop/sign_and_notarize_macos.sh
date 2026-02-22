#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <path-to-app-or-dmg>" >&2
  exit 1
fi

ARTIFACT_PATH="$1"
: "${MACOS_CODESIGN_IDENTITY:?MACOS_CODESIGN_IDENTITY is required}"
: "${MACOS_NOTARY_PROFILE:?MACOS_NOTARY_PROFILE is required}"

codesign --force --options runtime --timestamp --sign "$MACOS_CODESIGN_IDENTITY" "$ARTIFACT_PATH"
xcrun notarytool submit "$ARTIFACT_PATH" --keychain-profile "$MACOS_NOTARY_PROFILE" --wait
xcrun stapler staple "$ARTIFACT_PATH"

echo "Signed and notarized: $ARTIFACT_PATH"
