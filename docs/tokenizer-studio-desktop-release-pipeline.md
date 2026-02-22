# Tokenizer Studio Desktop Release Pipeline

## Build Inputs

- `apps/tokenizer-studio/web`: frontend static bundle source.
- `apps/tokenizer-studio/api`: backend sidecar source + requirements.
- `apps/tokenizer-studio/desktop`: Tauri shell source.
- `scripts/desktop/*`: platform build and signing automation.

## Per-Platform Pipeline

1. Build web:
   - `scripts/desktop/build_web.sh`
2. Build runtime:
   - `scripts/desktop/build_runtime_macos.sh <platform>`
   - `scripts/desktop/build_runtime_windows.sh <platform>`
   - `scripts/desktop/build_runtime_linux.sh <platform>`
3. Smoke test runtime:
   - `scripts/desktop/smoke_test_runtime.sh <platform>`
4. Build shell:
   - `cd apps/tokenizer-studio/desktop && npm run tauri build`
5. Assemble runtime archive and manifest:
   - `scripts/desktop/assemble_bundle.sh <platform>`
6. Sign artifacts:
   - macOS: `scripts/desktop/sign_and_notarize_macos.sh <artifact>`
   - Windows: `scripts/desktop/sign_windows.sh <artifact>`
   - Linux: `scripts/desktop/sign_linux.sh <artifact>`

## Runtime Update Manifest

`assemble_bundle.sh` emits:

- `tokenizer-studio-runtime-<platform>-<version>.tar.gz`
- `runtime-manifest.json`

Desktop shell can use `TOKENIZER_STUDIO_RUNTIME_MANIFEST_URL` to auto-download runtime when no local runtime is available.

## Release Channels

- `alpha`: internal verification.
- `beta`: early users.
- `stable`: public rollout.

Recommended promotion flow:

1. Publish to `alpha`.
2. Monitor startup/health failures and rollback rate.
3. Promote to `beta`.
4. Promote to `stable` after validation window.
