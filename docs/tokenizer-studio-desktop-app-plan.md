# Tokenizer Studio Desktop App Plan

## Goal

Ship Tokenizer Studio as a non-technical-user desktop product:

- User installs one app (`.dmg` / `.msi` / Linux package).
- App launches locally with one click.
- App internally runs the FastAPI backend and web UI.
- User never installs Python/Node manually.
- All training/tokenization compute runs on user hardware.

This plan is intentionally designed to scale to future local-first apps (tokenizer training now, LLM training apps later).

## Current State (from this repo)

- Frontend is Next.js in `apps/tokenizer-studio/web/`.
- Backend is FastAPI in `apps/tokenizer-studio/api/`.
- Current workflow is developer-style local setup (`venv`, `npm install`, `uvicorn`, `npm run dev`) in `apps/tokenizer-studio/README.md`.
- Backend and frontend are wired to localhost.
- Data/artifacts currently default to app-relative directories under `apps/tokenizer-studio/api/`.

## Product Requirements

1. Zero technical setup:
- No terminal usage.
- No Python/Node installation by user.

2. Reliable local execution:
- Works offline for local files.
- Supports heavier dependencies (`torch`, `datasets`, `tokenizers`) without user intervention.

3. Multi-platform:
- macOS (Apple Silicon first, Intel optional).
- Windows x64.
- Linux x64 (best effort at first release).

4. Safe local behavior:
- Backend binds to localhost only.
- Clear data paths.
- Predictable update behavior.

5. Extensible foundation:
- Same desktop shell can host future local apps (LLM training modules).

## Architecture Decision

Use a desktop shell + local backend sidecar:

- Shell: Tauri app (small, native-feeling launcher).
- Backend: packaged Python runtime + FastAPI app as sidecar process.
- Frontend: production static build served by FastAPI (preferred), not `next dev`.

Why this model:

- Keeps current architecture mostly intact.
- Keeps app small compared with full Chromium shell.
- Preserves local compute model (you do not pay for user training workloads).
- Enables one installer for non-technical users.

## High-Level Runtime Design

1. User launches desktop app.
2. Shell starts backend sidecar with app-specific env vars.
3. Backend starts on `127.0.0.1:<dynamic_port>` and exposes `/health`.
4. Shell waits for health readiness.
5. Shell opens desktop window pointed to local URL.
6. User trains tokenizer locally.
7. Artifacts/caches are stored in OS app-data directories.
8. On app close, shell gracefully shuts down backend.

## Distribution Strategy for Heavy ML Dependencies

`torch` + `datasets` make installers large. Use two channels:

1. Full offline bundle:
- Installer includes runtime + dependencies.
- Largest binary size.
- Best for air-gapped users.

2. Thin installer (recommended default):
- Installer contains shell + bootstrap.
- First run downloads signed platform runtime bundle.
- Runtime is cached and versioned.
- Better install UX and faster iteration.

Both channels use the same runtime layout and startup contract.

## Runtime Bundle Layout (per platform)

Example layout:

```text
runtime/
  python/                 # embedded Python
  site-packages/          # installed deps (fastapi, torch, datasets, tokenizers, etc.)
  app/                    # backend source + startup module
  web/                    # built frontend assets (if served statically)
  wheelhouse/             # optional for repair/update flows
  VERSION                 # runtime semantic version
```

## Repository Changes

## 1) Frontend Production Build Changes

In `apps/tokenizer-studio/web/`:

- Add production build target for desktop distribution.
- Move API base URL to environment-driven config:
  - Dev default: `http://127.0.0.1:8000/api/v1`.
  - Desktop/prod default: relative `/api/v1` when served by backend.
- Ensure no dependency on Next dev server at runtime.

Deliverable:
- Reproducible `npm run build` output usable by backend static serving.

## 2) Backend Packaging and Runtime Changes

In `apps/tokenizer-studio/api/`:

- Add production settings profile:
  - bind host `127.0.0.1`.
  - dynamic port support.
  - stable `/health` readiness endpoint.
- Add static asset serving for built frontend.
- Move storage roots from repo-relative defaults to OS app-data paths:
  - uploads
  - artifacts
  - sqlite db
  - Hugging Face caches (`HF_HOME`, dataset cache)
- Keep local-only behavior as default.

Deliverable:
- Backend starts from packaged runtime without repo checkout.

## 3) Desktop Shell Project

Add new folder (example): `apps/tokenizer-studio/desktop/`.

Responsibilities:

- Detect/select runtime version.
- Start/monitor backend sidecar.
- Health check with timeout + retries.
- Open app window to local URL.
- Show startup/error UI if backend fails.
- Collect and rotate backend logs.
- Graceful shutdown signaling.

Deliverable:
- `tauri build` produces installable desktop binaries.

## 4) Build and Packaging Scripts

Add `apps/tokenizer-studio/desktop/scripts/`:

- `build_web.sh` (frontend production assets).
- `build_runtime_<platform>.sh` (embedded Python + pip install deps).
- `assemble_bundle.sh` (copy backend/web/runtime manifest).
- `sign_and_notarize_*` scripts for release.

Deliverable:
- One-command reproducible build in CI per platform.

## Security Model

1. Localhost-only backend:
- Bind only to `127.0.0.1`.
- Never listen on `0.0.0.0` in desktop mode.

2. Request hardening:
- Optionally use per-launch local auth token (desktop shell injects token header).
- Restrict CORS to desktop origin/localhost.

3. Input safety:
- Continue filename sanitization for uploads.
- Validate config payloads strictly before training.

4. Update trust:
- Signed app binaries.
- Signed runtime bundle manifest and checksums.

## Data and Storage Plan

Use per-OS app data locations, for example:

- macOS: `~/Library/Application Support/TokenizerStudio/`
- Windows: `%AppData%\\TokenizerStudio\\`
- Linux: `~/.local/share/TokenizerStudio/`

Subfolders:

- `db/`
- `uploads/`
- `artifacts/`
- `logs/`
- `cache/huggingface/`

Add export/import and clear-cache UI actions (post-MVP acceptable).

## UX Requirements for Non-Technical Users

1. Installer flow:
- Download installer.
- Install.
- Launch app.

2. First run:
- Splash screen with startup status.
- Optional runtime download progress (thin installer path).
- Friendly error recovery actions.

3. In-app operations:
- File picking via native dialogs.
- Clear training progress and log view.
- Artifact download/open-folder actions.

4. Failure handling:
- Detect missing disk space.
- Detect incompatible CPU/GPU/runtime.
- Show actionable error messages (no stack traces by default).

## Update Strategy

Two update streams:

1. Shell update:
- Tauri auto-updater with signed releases.

2. Runtime update:
- Versioned runtime bundle fetched by manifest.
- Background download + apply on next restart.
- Rollback to previous known-good runtime if health check fails.

## CI/CD and Release Pipeline

Create release workflow per platform:

1. Build web assets.
2. Build backend runtime bundle.
3. Build desktop shell installer.
4. Run smoke tests (launch, health, basic job).
5. Sign/notarize artifacts.
6. Publish release with checksums and runtime manifest.

Release channels:

- `alpha` (internal)
- `beta` (early users)
- `stable` (public)

## Testing Plan

## Unit

- Backend config resolution.
- Storage path initialization.
- Sidecar startup/teardown supervisor logic.

## Integration

- Desktop launch -> backend health -> UI loads.
- Upload dataset -> start job -> artifact appears.
- Restart app -> historical jobs shown from sqlite.

## E2E

- Fresh install on clean VM.
- Thin-installer runtime download flow.
- Offline launch after successful first run.

## Non-functional

- Cold start time target.
- Peak RAM and disk footprint tracking.
- Large dataset handling sanity tests.

## Phased Delivery Plan

## Phase 0: Technical Design (1 week)

- Finalize desktop architecture spec.
- Choose runtime bundling approach (full vs thin default).
- Freeze supported platforms for v1.

Exit criteria:
- Design doc approved.
- Dependency and signing requirements finalized.

## Phase 1: Backend and Frontend Productionization (1-2 weeks)

- Frontend production build output ready.
- Backend serves static frontend and `/api/v1`.
- App-data path migration completed.

Exit criteria:
- Run app locally without `next dev`.

## Phase 2: Desktop Shell MVP (2 weeks)

- Tauri shell starts sidecar backend.
- Health-checked startup + graceful shutdown.
- Basic error screens implemented.

Exit criteria:
- Internal users can install and run tokenizer training end-to-end.

## Phase 3: Packaging and Auto-Update (2-3 weeks)

- Per-platform installers produced in CI.
- Signing/notarization configured.
- Runtime update mechanism live.

Exit criteria:
- Installers distributed to beta users with reliable upgrades.

## Phase 4: Hardening and GA (2 weeks)

- Telemetry/log collection (privacy-safe).
- Crash recovery improvements.
- Docs/support playbooks finalized.

Exit criteria:
- Stable release criteria met.

## Risks and Mitigations

1. Large runtime size (torch):
- Mitigation: thin installer + runtime download and caching.

2. Platform-specific dependency issues:
- Mitigation: per-platform lockfiles/wheels; CI matrix on clean runners.

3. Startup failures hidden from user:
- Mitigation: explicit startup diagnostics screen + logs viewer.

4. Local port conflicts:
- Mitigation: dynamic port selection with retry.

5. Future app sprawl:
- Mitigation: define plugin-style job-type contract early.

## Definition of Done (v1 Desktop)

- Non-technical user can install and launch without terminal.
- Backend + UI run locally and reliably on supported platforms.
- Tokenizer training flow works end-to-end.
- Artifacts persist in OS app-data path.
- Signed installers and updater are operational.
- Troubleshooting docs exist for common failures.

## Post-v1 Extensions (for future LLM apps)

- Add app registry in desktop shell (Tokenizer Studio, Training Studio, etc.).
- Shared job runner and storage services.
- Optional GPU capability detection and optimization profiles.
- Advanced dataset/model cache management UI.
- Optional BYOC remote runner mode for power users.

## Immediate Next Actions

1. Create architecture RFC in `docs/` and choose:
- Thin installer default vs full-bundle default.
- Target platforms for v1 launch.

2. Start implementation branch:
- Frontend production API base refactor.
- Backend static serving + app-data path refactor.

3. Scaffold desktop shell:
- Tauri project with sidecar lifecycle prototype.

4. Stand up CI proof-of-concept:
- Build unsigned internal artifacts for one platform first (macOS arm64 recommended), then expand.
