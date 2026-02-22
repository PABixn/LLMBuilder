# Tokenizer Studio Desktop Architecture RFC

## Status

- Date: 2026-02-21
- Status: Accepted
- Owners: Tokenizer Studio

## Summary

Tokenizer Studio is shipped as a local-first desktop app using a Tauri shell that supervises a packaged FastAPI sidecar backend and a production-built web UI.

The desktop shell starts the backend on `127.0.0.1:<dynamic_port>`, validates readiness via `/health`, and then navigates to the local app URL.

## Decisions

## 1. Installer Channel Strategy

- Default channel: thin installer.
- Optional channel: full offline bundle.

Rationale:

- Thin installer keeps shell installer small and allows faster runtime updates.
- Full bundle remains available for restricted/offline environments.

## 2. Runtime Layout Contract

Runtime bundle per platform:

```text
runtime/
  python/
  app/
    app/
    templates/
    tokenizer/
  web/
  wheelhouse/
  VERSION
```

Runtime discovery order in shell:

1. `TOKENIZER_STUDIO_RUNTIME_DIR` (explicit override).
2. `<app_data>/runtime/current`.
3. `<app_data>/runtime/default`.
4. Development fallback paths near current working directory.

## 3. Backend Network and Security Defaults

- Host bind: `127.0.0.1` only.
- Health endpoint: `/health` and `/api/v1/health`.
- Optional runtime token header supported via `TOKENIZER_STUDIO_RUNTIME_TOKEN`.
- CORS restricted to localhost defaults unless explicitly overridden.

## 4. Storage and Cache Policy

Default root is OS app-data:

- macOS: `~/Library/Application Support/TokenizerStudio`
- Windows: `%AppData%\TokenizerStudio`
- Linux: `~/.local/share/TokenizerStudio`

Managed paths:

- `db/tokenizer_studio.db`
- `uploads/`
- `artifacts/tokenizers/`
- `logs/`
- `cache/huggingface/`

HF env setup on startup:

- `HF_HOME`
- `HF_DATASETS_CACHE`
- `HUGGINGFACE_HUB_CACHE`

## 5. Platform Scope for v1

- macOS arm64: required.
- Windows x64: required.
- Linux x64: best effort.

## 6. CI Release Policy

Initial CI target:

- macOS arm64 unsigned internal artifacts (proof-of-concept).

Steps:

1. Build web desktop assets.
2. Build runtime bundle.
3. Smoke test runtime sidecar startup + health.
4. Build desktop shell (unsigned, no bundle for POC).
5. Assemble runtime archive + manifest.

Expansion targets:

- Signed installer generation.
- Notarization/signing automation per platform.
- Auto-update channels (`alpha`, `beta`, `stable`).

## Consequences

Benefits:

- No user Python/Node installation.
- Local compute and offline usage after runtime is present.
- Shared shell pattern for future local-first ML apps.

Tradeoffs:

- Runtime bundle size remains large due to ML dependencies.
- Packaging complexity increases across platforms.
- Runtime-update security/signing pipeline is mandatory for production rollout.
