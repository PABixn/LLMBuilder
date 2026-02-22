# Desktop Build Scripts

## Build

- `build_web.sh [output_dir]`
  - Builds static web assets for desktop mode.
  - Copies output to `apps/tokenizer-studio/api/web-dist` by default.

- `build_runtime_macos.sh [platform]`
- `build_runtime_windows.sh [platform]`
- `build_runtime_linux.sh [platform]`
  - Build Python runtime bundles with backend dependencies and app code.

- `smoke_test_runtime.sh [platform]`
  - Launches bundled runtime backend and verifies `/health` endpoints.

- `assemble_bundle.sh [platform]`
  - Produces runtime `.tar.gz` and `runtime-manifest.json`.

- `build_desktop.sh [platform]`
  - Runs web build, runtime build, Tauri build, and runtime assembly.
  - Refreshes Tauri icon set by default (`npm run icons`).

## Signing

- `sign_and_notarize_macos.sh <artifact>`
- `sign_windows.sh <artifact>`
- `sign_linux.sh <artifact>`

## Useful Environment Variables

- `BUILD_ROOT` (default: `build/desktop`)
- `RUNTIME_VERSION` (default: `0.1.0-dev`)
- `PYTHON_BIN` (runtime-builder Python executable)
- `WEB_DIST_DIR` (web assets input for runtime bundle)
- `TOKENIZER_STUDIO_SKIP_NPM_CI=1` (skip `npm ci` in `build_web.sh`)
- `TOKENIZER_STUDIO_SKIP_NPM_INSTALL=1` (skip `npm install` in `build_desktop.sh`)
- `TOKENIZER_STUDIO_SKIP_ICON_REFRESH=1` (skip `npm run icons` in `build_desktop.sh`)
