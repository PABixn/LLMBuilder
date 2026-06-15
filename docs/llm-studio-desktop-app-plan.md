# LLM Studio Cross-Platform Desktop Application Execution Plan

**Document date:** 2026-06-14
**Plan status:** Locally feasible implementation complete; external and explicitly excluded release gates remain open
**Target product:** A cross-platform desktop edition of LLM Studio with the same user-visible behavior and capabilities as the current web edition

## Executor Completion Protocol

This document is an executable checklist. The executor must maintain it while implementing the desktop application.

- Every actionable step and acceptance gate starts with `[ ]`.
- Change `[ ]` to `[*]` only after the implementation, required tests, and documented acceptance evidence for that item are complete.
- Never mark an item `[*]` merely because code was written. Its tests and acceptance conditions must also pass.
- If an item is partially complete or blocked, leave it `[ ]` and append `BLOCKED:`, `PARTIAL:`, or `DECISION REQUIRED:` with the reason and a link to the issue or decision record.
- Mark items individually as they finish. Do not batch-mark an entire phase at the end.
- Keep implementation evidence close to the item: commit/PR link, command output summary, test report, screenshot, or release artifact URL.
- If implementation discoveries require a change to this plan, update the relevant section before changing the architecture or scope.
- A phase exit gate may be marked `[*]` only when every required item in that phase is `[*]`.
- “Same functionality” means functional parity with the web edition, not merely that every route renders. All parity checks in this plan are release-blocking unless explicitly labeled optional.

## Execution Record: 2026-06-12 through 2026-06-14

This execution implemented and verified every locally feasible source/runtime
step on a macOS arm64 development host. Items that require external credentials,
organizational approval, other target operating systems, clean non-developer
machines, publication infrastructure, or browser-driven validation remain open
and are not represented as complete.

### Verified Evidence

- Authoritative local command: `make -C apps/llm-studio
  desktop-verify-nonbrowser` passed after the final source changes.
- API: `python -m pytest -q` passed, **262 tests**, with one upstream
  Starlette/httpx deprecation warning.
- Frontend: `npm run lint`, `npm run typecheck`, and `npm run build:desktop`
  passed; static-output validation found all six routes and no configured
  secrets/developer paths. `npm run test:regression` passed, **66 tests**.
- Browser-driven checks were not run and Chromium was not installed, by explicit
  product-owner direction on 2026-06-13. Every browser/webview visual/parity gate
  remains open.
- Desktop shell: Rust format and clippy with `-D warnings` passed; **29 shell
  tests** passed; `tauri build --no-bundle` produced the macOS arm64 release
  binary with complete `cargo-auditable` metadata for **291 included crates**.
  The complete compiled-binary scan reports zero vulnerabilities and seven
  visible transitive maintenance/unsoundness warnings. The path-remapped binary
  and all six static routes passed the combined release payload audit across
  **97 files / 13,866,302 bytes**, with no secret, developer-path, cache, or
  unintended-file findings.
- Runtime: linked-development runtime built with **107 hashed files plus
  manifest**, reported **0.8 MiB**, and passed imports, authenticated real-sidecar
  startup, missing-token rejection, Unicode/space paths, model validation and
  analysis, project CRUD/artifact, tokenizer managed upload/stats/validation,
  tokenizer validation/training/history/preview/artifact/delete, training
  configuration/preflight/history/metrics/samples/logs/checkpoint, one-step
  local model training, valid long-run local-training cancellation, synchronous
  and streamed inference, artifact ZIP export, managed deletion, and configured
  backend static serving with safe asset, deep-link fallback, API-fallback,
  missing-asset, and traversal behavior. A second real-sidecar launch against
  the same data root rotates the launch token and verifies completed project,
  tokenizer, training history, and artifacts before managed cleanup. The smoke
  snapshots packaged runtime contents and metadata before imports and proves
  they remain unchanged after the complete workflow and shutdown.
- Fault/lifecycle evidence covers collision-safe writable probes, low disk,
  read-only roots, locked/corrupt SQLite databases, occupied loopback ports,
  corrupt/missing runtime metadata, fake-backend crashes, Unix forced
  local-training descendant cleanup, supervisor descendant cleanup, and abrupt
  backend-parent death while real local training is active, proving backend and
  owned training-tree cleanup without terminating an unrelated process.
- Data schema 3 preserves schema-2 relocatable typed SQLite paths and adds an
  idempotent credential cleanup that strips legacy Hugging Face dataset tokens
  and redacts provider credentials from active, retained legacy, and migration
  backup databases before checkpointing/removing WAL/SHM sidecars and physically
  compacting them. It also sanitizes managed job inputs/logs and removes
  transient remote bundles that could retain historical embedded tokens.
  Focused tests cover schema-1 migration, backup, transactional rollback,
  same-named database backup isolation, idempotency, relocation,
  external/remote-path preservation, and unsafe typed-location/symlink rejection.
  The checked-in pre-desktop/schema-1 golden fixture proves copy-only source-tree
  import and schema migration against durable test data.
  A separate copy-first, integrity-checked, idempotent migration moves the old
  default `tokenizer_studio.db` and `training_studio.db` contents into active
  LLM Studio-namespaced databases without overwriting current data. Frontend
  source regressions likewise prove copy-before-remove migration of legacy
  Tokenizer Studio form/theme storage into active LLM Studio keys.
- Dataset credentials are separated from every durable tokenizer/training
  configuration and passed only through in-memory execution configuration or
  child/pod environment. Job-scoped exact-value redaction protects arbitrary
  legacy token formats in backend/lifecycle errors and active log responses;
  runner state and remote diagnostics independently redact execution
  environment tokens. Terminal training transitions atomically scrub managed
  text outputs before releasing the redaction scope, and artifact export fails
  closed if scrubbing cannot complete. Adversarial tests inspect SQLite bytes,
  managed files, remote bundles, active/terminal logs, runner state, diagnostics,
  and artifact ZIPs.
- Runtime staging correctly rejected the linked-development runtime as
  nonportable. Focused tests also cover checksum mismatch, unsafe manifest paths,
  release symlinks, and target metadata checks.
- A full macOS arm64 portable-unlocked characterization runtime passed the same
  authenticated smoke and Python dependency audit. Build sanitization reduced it
  to **18,234 hashed files plus manifest / 602.4 MiB**, with no `pip`, console or
  activation scripts, dependency test/cache trees, bytecode, or build-command
  provenance. Staging correctly rejects it as non-release. Its payload audit
  records **80 upstream wheel/example home-path findings**, which require a
  reviewed release wheelhouse/binary triage rather than broad suppression.
- Release audit generation produced a path-redacted `release-manifest.json` and
  `SHA256SUMS`; focused tests cover hashes, symlinks, duplicate names, unsafe
  output paths, and atomic output.
- Dependency audit: web and desktop `npm audit --audit-level=high` each reported
  **0 vulnerabilities**. The reviewed audit gate reports zero blocking findings
  across 85 development Python packages and 65 portable-runtime packages, zero
  Rust vulnerabilities across 538 locked crates, one narrowly documented
  pip-audit/PyTorch affected-range mismatch, and 19 visible transitive Rust
  maintenance/unsoundness warnings requiring release review. Release shell
  builds use pinned `cargo-auditable` 0.7.4; the complete native-binary scan
  recovers all 291 included crates, reports zero vulnerabilities, and retains
  seven applicable transitive warnings for review.
- API contract fixtures: the checked-in sanitized catalog freezes all **55**
  registered request/response contracts across the eight parity-matrix groups.
  Its validator regenerates the catalog from FastAPI OpenAPI, rejects
  unclassified operations, validates representative request/response keys and
  evidence paths, rejects concrete secrets, and fails on any unreviewed drift.
- Runtime size policy: every runtime build records and enforces deterministic
  payload file/byte counts. Development builds use checked-in guardrails;
  release-portable builds fail before construction unless their exact target
  has an approved threshold, and staging independently recomputes the payload
  size and requires that approved release threshold.
- Startup profiling: authenticated readiness and shell diagnostics expose
  deterministic cold-start timing, including pre-lifespan application/PyTorch
  imports, runtime validation, managed-data preparation, migrations/store
  initialization, and total time to ready. The packaged-runtime smoke asserts
  and prints this profile so target-native CI evidence can identify bottlenecks.
  It also atomically writes a path-redacted per-target characterization report
  covering client-observed cold/warm first-API readiness, backend-only idle RSS,
  compute capabilities, runtime footprint, and deterministic tiny
  tokenizer/local-training create-to-terminal times. CI retains each target
  report; browser route timing, shell/webview memory, installer/update size, and
  approved thresholds remain external. The final linked macOS arm64 report
  measured **902 ms cold / 936 ms warm** client-observed first API readiness,
  **804 ms cold** backend readiness including **788 ms** application/PyTorch
  imports and **16 ms** migration/store initialization, **259,899,392 bytes**
  median backend-only idle RSS, **279 ms** tiny tokenizer time, **1,597 ms**
  tiny local-training time, and a **107-file / 896,641-byte** runtime payload.
  It recorded CPU and MPS available, CUDA unavailable, and no approved
  thresholds.
- Diagnostics now include a privacy-preserving, bounded aggregate inventory of
  data/cache/log file counts and bytes. Focused shell tests prove large sparse
  file reporting, entry/time-limit behavior, missing-root handling, and no
  symlink traversal without exporting filenames or paths. Collection runs
  off-thread from an owned supervisor snapshot and does not hold the lifecycle
  lock.
- Toolchain check passed with Python 3.12.7, pinned Rust 1.88.0, Cargo Audit
  0.22.1, pinned `cargo-auditable` 0.7.4, and required Python, Node/npm, Rust,
  audit, and lockfile inputs present.
- Checklist audit: **248 complete / 142 open / 0 unclassified**. Every open item
  is explicitly partial, externally blocked, or browser/webview-excluded.

### Open External or Explicitly Excluded Gates

- **BLOCKED EXTERNALLY:** product/security/architecture/release/QA/support
  approvals, issue/epic ownership records, signing identities, macOS
  notarization, Windows signing/reputation, Linux package selection/signing,
  protected release environments, publication, and updater rollout.
- **BLOCKED EXTERNALLY:** Windows x64, Linux x64, clean-machine, uninstall,
  upgrade/rollback, endpoint-security, proxy/custom-CA, and cross-platform
  performance validation require their target environments.
- **RELEASE BLOCKER:** select reviewed redistributable Python builds and
  platform-specific locked wheelhouses/constraints. The current portable builder
  characterizes CI behavior but is not an approved redistributable runtime
  strategy; the current unlocked macOS wheel payload also retains 80 upstream
  home-path strings that require wheel/binary review.
- **EXPLICITLY EXCLUDED:** browser installation and browser/webview checks.
  Static validation and non-browser regressions passed, but screenshots,
  interactions, visual accessibility, route refresh/deep-link behavior, and
  webview parity remain open.
- **BLOCKED EXTERNALLY:** opt-in real RunPod smoke requires approved credentials,
  budget, human approval, and cleanup monitoring. Default tests remain fake-only.

## Objective

Ship LLM Studio as a one-click desktop application that:

- Runs on supported macOS, Windows, and Linux targets without requiring the user to install Python, Node.js, Rust, or repository dependencies.
- Presents the same routes, workflows, data, validations, local training, RunPod training, tokenizer training, artifact management, and inference capabilities as the current web application.
- Preserves the current web application and its development workflow.
- Stores mutable data in OS-appropriate application-data locations, never in signed/read-only application resources.
- Starts, monitors, authenticates, and shuts down a packaged local FastAPI/Python runtime safely.
- Provides signed, reproducible, updateable installers and a supportable diagnostics path.

## Explicit Non-Goals

- Do not redesign or simplify the existing LLM Studio UI as part of desktop delivery.
- Do not replace FastAPI, Python model/tokenizer/training code, or existing API contracts with Rust implementations.
- Do not merge the normal web edition and desktop shell into a desktop-only code path.
- Do not require user-installed Python, Node.js, Rust, package managers, or command-line setup.
- Do not promise universal local GPU support. Support only the platform/runtime combinations explicitly tested and published in the compute matrix.
- Do not make background/tray training, multi-instance execution, thin-runtime delivery, telemetry upload, or OS-keychain RunPod persistence implicit requirements. Each requires an explicit approved decision and its corresponding checklist work.
- Do not package or execute the RunPod `remote_agent` locally unless a documented runtime requirement proves it necessary.
- Do not write mutable data into the installed application bundle or packaged Python runtime.

## Definition of Done

The desktop application is complete only when all of the following are true:

- [ ] A non-technical user can install, launch, use, update, and uninstall LLM Studio without a terminal. BLOCKED EXTERNALLY: reviewed portable runtimes, signed installers, updater, and clean-machine validation are not available.
- [ ] All route-level and API-level parity checks in this plan pass on every required platform. BLOCKED: required platforms and explicitly excluded browser/webview parity checks.
- [ ] The app launches from a clean machine with no preinstalled Python or Node.js. BLOCKED EXTERNALLY: clean-machine signed-installer validation requires reviewed portable runtimes and target environments.
- [*] The app can operate offline for workflows that use only already-local data and dependencies. Evidence: offline static build and local-fixture real-sidecar smoke pass without runtime dependency installation.
- [ ] Local tokenizer training, local model training, inference, and RunPod workflows behave the same as the web edition within the documented platform compute support matrix. PARTIAL: real-sidecar local tokenizer/training/inference pass; real RunPod and required target compute matrices remain external.
- [ ] Existing workspace data survives app restart and supported app upgrades. PARTIAL: real-sidecar restart persistence passes; released-version upgrades and live webview localStorage remain external/excluded.
- [ ] Closing the app cannot silently abandon a local training process or leave an unmanaged RunPod billing risk. PARTIAL: active-job query, explicit close warnings, owned local shutdown, and Unix backend parent-death cleanup are implemented/tested; abrupt target-native shell termination with active training and real RunPod lifecycle validation remain external.
- [*] The backend binds only to loopback, requires a per-launch secret for protected API routes, and does not expose broad Tauri capabilities. Evidence: bind/auth/CORS tests and event-listen-only Tauri capability.
- [ ] Installers are signed where the platform provides signing, macOS builds are notarized, release artifacts have checksums/signatures, and update rollback is tested. BLOCKED EXTERNALLY: signing identities, protected release infrastructure, publication, and updater rollout.
- [ ] Clean-machine end-to-end tests and release smoke tests pass for every required target. BLOCKED EXTERNALLY: target clean machines and signed installers.

## Codebase Analysis Summary

### Repository Shape

The desktop implementation must preserve and package the existing architecture rather than rewrite the product:

- `apps/llm-studio/web/`: Next.js 16 / React 19 frontend.
- `apps/llm-studio/api/`: FastAPI backend and local runtime orchestration.
- `model/`: model construction and configuration code used by validation, analysis, training, and inference.
- `tokenizer/`: tokenizer implementation used by tokenizer jobs.
- `training/`: local training runner, configuration, data loading, and inference support.
- `apps/llm-studio/remote_agent/`: RunPod pod-side agent; it is part of remote training infrastructure, not the local desktop runtime.
- `docker/training/`: remote training image.
- `docs/`: existing architecture, parity, security, and workflow documentation.

The analyzed frontend contains roughly 35,691 TypeScript/TSX lines, and the API application contains roughly 9,838 Python lines. Desktop work therefore must center on reuse, runtime adaptation, and parity verification.

### Existing Desktop Foundations to Reuse

The repository already contains several useful desktop-oriented foundations:

- `apps/llm-studio/web/package.json` defines `build:desktop`.
- `apps/llm-studio/web/next.config.ts` switches to Next static export for the cross-platform
  `build:desktop` npm lifecycle or when `LLM_STUDIO_DESKTOP_BUILD=1` is set explicitly.
- `apps/llm-studio/Makefile` describes the web build as a desktop embedding/static export flow.
- `apps/llm-studio/api/app/config.py` already resolves OS-specific application-data and cache roots.
- `apps/llm-studio/api/app/main.py` can serve a built frontend and has `/health` plus `/api/v1/health`.
- API runtime-token middleware already protects `/api/v1/*` when configured.
- `apps/llm-studio/web/app/tokenizer/lib/storage.ts` already detects Tauri invocation APIs.
- `apps/llm-studio/web/app/tokenizer/components/TokenizerPageContent.tsx` already attempts a `save_tokenizer_artifact` Tauri command before browser-download fallback.

Historical implementation work is available in Git history:

- Commit `4796a67` contains a Tauri v2 Tokenizer Studio shell, runtime builder scripts, desktop RFC, release pipeline, and troubleshooting documentation.
- Commit `f60cedd` contains follow-up shipping work.
- Commit `c4506f9` removed the Tokenizer Studio-specific desktop tree while merging applications into LLM Studio.
- The executor should recover useful patterns with `git show`, but must not restore the historical implementation unchanged.

### Historical Implementation Defects That Must Not Be Reintroduced

The removed Tauri implementation is a valuable prototype, but it is insufficient for LLM Studio:

- It navigated the main webview from a bundled startup page to `http://127.0.0.1:<dynamic-port>`. A changing origin would isolate or lose the current localStorage-backed workspace state between launches.
- Navigating to the loopback URL weakens the clean separation between trusted bundled UI and local HTTP content, complicates Tauri IPC availability, and broadens the security surface.
- It packaged only Tokenizer Studio portions rather than `model/`, `tokenizer/`, `training/`, and the full LLM Studio API.
- Its runtime builder created a generic virtual environment with unconstrained platform installs; that is not a reproducible cross-platform ML runtime strategy.
- It reserved a dynamic port and released it before the backend bound it, leaving a port-race window.
- Its native Save As implementation worked only on macOS.
- It used `csp: null`.
- Its thin-runtime downloader verified only a checksum, did not establish a signed manifest trust chain, and lacked robust rollback/progress/recovery behavior.
- Its process shutdown did not fully address the local-training process tree or Windows Job Objects.

### Verified Baseline on 2026-06-12

- `npm run lint` from `apps/llm-studio/web`: passed.
- `npm run typecheck` from `apps/llm-studio/web`: passed.
- `npm run test:regression` from `apps/llm-studio/web`: passed, 47 tests.
- `npm run build:desktop` from `apps/llm-studio/web`: passed when network access to Google Fonts was available.
- `npm run build:desktop` failed without network access because `next/font/google` attempted to fetch Sora and IBM Plex Mono. Release builds are not yet offline/reproducible.
- Static export currently produces all user routes: `/`, `/simple`, `/studio`, `/tokenizer`, `/training`, and `/inference`.
- Static export currently emits route files such as `studio.html`, `training.html`, and `tokenizer.html`; route navigation and refresh behavior must be verified inside Tauri's asset protocol.
- API tests could not be run in the analyzed local environment because the available Python interpreters did not have `pytest` installed. CI and a dedicated development environment must establish the authoritative API baseline before implementation.

### Desktop-Critical Current Behavior

The desktop implementation must account for the following current behavior:

- Frontend API settings are resolved at module load in three separate client surfaces:
  - `apps/llm-studio/web/lib/api.ts`
  - `apps/llm-studio/web/lib/tokenizerLegacyApi.ts`
  - `apps/llm-studio/web/lib/training/client.ts`
- Development clients default to `http://127.0.0.1:8000/api/v1`; production clients default to relative `/api/v1`.
- Runtime tokens currently come from build-time `NEXT_PUBLIC_RUNTIME_TOKEN`, which is unsuitable for a per-launch desktop secret.
- Direct artifact URLs are constructed synchronously in `lib/workspaceAssets.ts`, `lib/training/artifacts.ts`, and `lib/tokenizerLegacyApi.ts`.
- Inference streaming uses fetch plus NDJSON response streaming in `lib/training/generation.ts`.
- Browser behaviors include file inputs, drag/drop, Blob downloads, clipboard access, confirm dialogs, URL query parameters, custom events, and streamed fetch responses.
- Mutable workspace state is persisted in browser localStorage. The desktop webview origin must remain stable across launches and updates.
- Backend modules contain many fixed `Path(__file__).resolve().parents[N]` source-tree assumptions. A packaged layout will break these unless runtime-root discovery is centralized.
- `apps/llm-studio/api/app/config.py` still defaults tokenizer artifacts and uploads to API/repository-relative paths. Packaged applications must instead use writable app-data paths.
- Local model training starts `sys.executable -m training.runner` in a child process. The desktop supervisor must manage the full process tree.
- Tokenizer training runs through an in-process `ThreadPoolExecutor`.
- On API restart, incomplete local jobs are marked failed; incomplete RunPod jobs are marked recovery-limited because the raw pod-agent token is not persisted.
- RunPod keys pasted into the UI are process-memory only and are not persisted.
- Training and tokenizer SQLite stores persist multiple filesystem paths. Migrations and runtime-location changes must preserve or repair them.

## Architecture Decision

### Selected Architecture

Use a Tauri v2 desktop shell with a bundled static frontend and a packaged Python/FastAPI sidecar:

```text
┌──────────────────────────────────────────────────────────────────┐
│ Tauri v2 desktop process                                         │
│                                                                  │
│  ┌───────────────────────────┐   narrow commands/capabilities     │
│  │ Bundled Next static UI    │◄──────────────────────────────┐    │
│  │ stable Tauri asset origin │                               │    │
│  └─────────────┬─────────────┘                               │    │
│                │ authenticated HTTP on 127.0.0.1             │    │
│                ▼                                              │    │
│  ┌───────────────────────────┐      process supervision       │    │
│  │ Packaged FastAPI/Python   │◄───────────────────────────────┘    │
│  │ dynamic loopback port     │                                    │
│  └─────────────┬─────────────┘                                    │
└────────────────┼───────────────────────────────────────────────────┘
                 │
       ┌─────────┴─────────┐
       │ model/tokenizer/  │
       │ training runtime  │
       └───────────────────┘
```

Key decisions:

- The main UI loads from Tauri `frontendDist` / asset protocol and remains there. Do not navigate the main app to the backend's random HTTP port.
- The bundled asset origin remains stable so all existing localStorage state remains available across launches.
- Tauri starts a packaged Python runtime sidecar on an ephemeral loopback port.
- Tauri generates a cryptographically strong per-launch token, passes it to FastAPI through `LLM_STUDIO_RUNTIME_TOKEN`, and exposes the API base URL plus token only through a narrow trusted bootstrap command.
- The frontend obtains runtime connection details at startup and all API/download/streaming clients consume that runtime configuration.
- The backend remains independently runnable for web development and can continue serving static web output when desired, but the shipped desktop UI does not depend on backend static serving.
- Mutable app data, logs, databases, caches, jobs, artifacts, and downloaded runtimes live outside signed application resources.
- Full/offline installers are the required first release format. A thin installer/runtime downloader is optional only after its signing, rollback, progress, proxy, retry, and recovery requirements are complete.

### Rejected Alternatives

- **Electron shell:** rejected for the initial implementation because Tauri foundations already exist in repository history and Tauri avoids bundling another full browser runtime. Reconsider only if a documented WebView compatibility blocker prevents parity.
- **Rewrite the backend in Rust:** rejected because it would duplicate mature FastAPI, tokenizer, training, inference, and RunPod behavior and create unacceptable parity risk.
- **Navigate Tauri to FastAPI-served UI:** rejected because a dynamic loopback port changes origin, threatens current localStorage continuity, and increases IPC/security complexity.
- **Require user-installed Python:** rejected because it violates one-click installation and reproducibility requirements.
- **Thin installer as initial required distribution:** rejected until signed runtime manifests, atomic activation, rollback, and failure recovery are proven.

## Required Platform and Compute Support Matrix

The executor must turn this proposal into an approved support matrix during Phase 1.

| Target | Initial release status | Local compute expectation | Required package |
|---|---|---|---|
| macOS arm64 | Required | CPU and tested Apple Silicon MPS behavior supported by bundled PyTorch | Signed and notarized `.dmg` or platform-standard Tauri bundle |
| Windows x64 | Required | CPU required; GPU support only if a separate tested runtime channel is explicitly approved | Signed installer, with WebView2 prerequisite handling |
| Linux x64 | Required unless product decision explicitly downgrades it | CPU required; GPU support only through explicit tested runtime channels | AppImage plus one native package format selected after compatibility testing |
| macOS x64 | Optional post-v1 | CPU | Separate signed/notarized artifact if adopted |
| Linux CUDA variants | Optional post-v1 | Only explicitly tested CUDA/PyTorch combinations | Separate runtime channels; never claim universal CUDA compatibility |

RunPod training remains available on every platform with working network access. CI must not spend money on RunPod by default; use mocked/fake provider tests and opt-in real-provider smoke tests.

## Functional Parity Contract

### Route-Level Parity Matrix

| Route | Current behavior that must remain equivalent in desktop |
|---|---|
| `/` | Workspace inventory, model/tokenizer/training assets, refresh/error states, sorting/filtering, artifact download, navigation, theme, UI-mode selection |
| `/simple` | Guided model/tokenizer/training/inference flow, persisted progress, coordinated presets, artifact readiness, links into expert routes |
| `/studio` | Visual model builder, drag interactions, backend validation/analysis, saved projects, import/export, undo/redo, diagnostics, `?project=` deep link |
| `/tokenizer` | Persisted forms, local uploads and drag/drop, streaming datasets, filters/weights, validation, tokenizer jobs and polling, preview, recent jobs, artifact download, `?job=` deep link |
| `/training` | Model/tokenizer selection, local/streaming datasets, training config, preflight and recommendations, local and RunPod launch, polling, stop, metrics, logs, samples, checkpoints, remote actions, artifact download, `?project=`, `?tokenizerJob=`, and `?run=` deep links |
| `/inference` | Completed-run selection, checkpoint selection, search, generation controls, streamed generation, completion/error states |

### API-Level Parity Matrix

The packaged runtime must expose and pass characterization tests for all current API groups:

| API group | Required endpoints/workflows |
|---|---|
| System/config | `/health`, `/api/v1/health`, config templates, config schemas |
| Model Studio | model validation, model analysis, project create/list/get/update/delete, project artifact download |
| Tokenizer | health, templates, schemas, tokenizer validation, dataloader validation, train/validation file upload, file stats, job create/list/get/delete, preview, artifact metadata, artifact download |
| Training config/preflight | health, templates, schemas, dataloader validation, training-config validation, preflight |
| RunPod provider | defaults, status, catalog, key validation, pods, network volumes |
| Training jobs | create/list/get/delete, metrics, samples, logs, checkpoints, stop, artifact download |
| Remote lifecycle | resync, cleanup, reattach, restart/recovery behavior |
| Inference | synchronous generation and streamed generation |

### Persistence Compatibility Contract

The following active browser storage keys must not be renamed or silently discarded:

- `llm-studio-theme`
- `llm-studio-ui-mode-v1`
- `llm-studio-workspace-asset-cache-v2`
- `llm-studio-document`
- `llm-studio-import-draft`
- `llm-studio-component-prefabs`
- `llm-studio-simple-flow-v1`
- `llm-studio-tokenizer-form`
- `llm-studio-tokenizer-dataset-form`
- `llm-studio-tokenizer-training-form`
- `llm-studio-tokenizer-active-job-id`
- `llm-studio-tokenizer-preview-text`
- `llm-studio-tokenizer-hidden-recent-job-ids`
- `llm-training-config-v1`
- `llm-training-dataloader-v1`
- `llm-training-selection-v1`
- `llm-training-active-run-v1`

The first LLM Studio desktop release explicitly migrates
`tokenizer-studio-theme` and the six historical `tokenizer-studio-*` form/job
keys into their active LLM Studio equivalents. Migration copies before removal,
never overwrites an active value, preserves legacy data on write failure, and
runs before theme paint or tokenizer-state hydration. If the desktop app
introduces native persistence later, it must migrate active keys explicitly and
preserve browser fallback behavior.

## Target Runtime and Storage Contracts

### Packaged Runtime Layout

Use an explicit, versioned runtime layout rather than relying on source-tree parent counts:

```text
runtime/
  manifest.json
  VERSION
  python/
    ... platform Python runtime and installed packages ...
  source/
    apps/llm-studio/api/app/
    apps/llm-studio/api/templates/
    model/
    tokenizer/
    training/
  metadata/
    python-lock-or-constraints.txt
    dependency-licenses/
    sbom.*
    build-provenance.json
```

The desktop `frontendDist` contains the static UI. Do not duplicate it into the Python runtime unless backend-served static smoke testing requires a separate test fixture.

### Mutable Data Layout

All mutable paths resolve from a single OS app-data root:

```text
LLMStudio/
  db/
    llm_studio_tokenizer.db
    llm_studio_training.db
  projects/
  uploads/
  artifacts/
    tokenizers/
  training/
    jobs/
    exports/
  cache/
    huggingface/
  logs/
    shell/
    backend/
  diagnostics/
  backups/
  runtime/                 # only if optional downloaded runtimes are later enabled
    versions/
    staging/
    current.json
```

Required OS defaults:

| OS | Data root | Cache behavior |
|---|---|---|
| macOS | `~/Library/Application Support/LLMStudio` | Large cache may use `~/Library/Caches/LLMStudio` if the data model cleanly separates it |
| Windows | `%APPDATA%\LLMStudio` with appropriate `%LOCALAPPDATA%` use for cache | Confirm roaming versus local behavior |
| Linux | `$XDG_DATA_HOME/LLMStudio` or `~/.local/share/LLMStudio` | `$XDG_CACHE_HOME/LLMStudio` or `~/.cache/LLMStudio` |

### Version Compatibility Contract

Release metadata must version these independently:

- Desktop shell version.
- Static web build version/commit.
- Python runtime version.
- Backend API contract version.
- Mutable-data schema version.
- Runtime manifest schema version.

The shell must refuse to start an incompatible runtime with an actionable message. Every data migration must be forward-safe, backed up, and tested against supported upgrade paths.

### Target Sidecar Startup Contract

The approved RFC may refine names, but the shell/backend boundary must remain explicit and testable:

| Contract value | Owner | Requirement |
|---|---|---|
| Desktop mode | Shell -> backend | Set an explicit desktop-mode flag; backend fails closed on desktop-unsafe settings |
| Host | Shell -> backend | Always `127.0.0.1`; never wildcard or externally reachable |
| Port/startup handshake | Shell <-> backend | Collision-safe ephemeral bind with the actual port returned through a structured handshake |
| Runtime token | Shell -> backend -> UI bootstrap | Strong random per launch; header-only; memory-only; redacted |
| Source root | Shell -> backend | Immutable packaged `source/` root used instead of positional parent discovery |
| Data root | Shell -> backend | Writable OS app-data root |
| Cache root | Shell -> backend | Writable OS cache root or documented data-root cache |
| Log root | Shell -> backend | Writable rotated backend log directory |
| Web serving | Shell -> backend | Disabled for the shipped desktop UI unless a specific compatibility test needs it |
| CORS origin | Shell -> backend | Exact bundled Tauri origin(s) verified per platform |
| Runtime/shell compatibility | Manifest -> shell/backend | Validated before backend serves user workflows |
| Readiness | Backend -> shell | Structured staged readiness with versions/capabilities and no secrets |
| Active-job status | Backend -> shell | Queried before application exit |
| Shutdown | Shell -> backend/process tree | Graceful request followed by bounded full-tree termination |

## Step-by-Step Execution Plan

## Phase 0: Establish Baseline, Ownership, and Plan Governance

- [ ] Assign owners for desktop shell, web runtime integration, Python packaging, release engineering, security review, QA, and documentation. BLOCKED EXTERNALLY: organizational ownership assignment.
- [ ] Create a desktop epic and one issue per phase/gate in this plan; link each issue back to this document. BLOCKED EXTERNALLY: issue-tracker/project-governance access and owner approval.
- [*] Record the approved product name, bundle identifier, executable name, app-data directory name, and URL/deep-link scheme in a short ADR. Evidence: `docs/llm-studio-desktop-adr.md`.
- [*] Record whether Linux x64 is a required GA platform or an explicitly labeled beta platform. Evidence: ADR records Linux x64 as beta.
- [*] Record the supported macOS minimum version, Windows minimum version, Linux distributions/glibc baseline, and supported CPU architectures. Evidence: ADR platform matrix.
- [ ] Record the initial local-compute support promise per platform, including the exact PyTorch build/channel and whether GPU acceleration is supported or intentionally excluded. PARTIAL: platform/compute promise is documented; exact reviewed target PyTorch locks/channels remain a release decision.
- [*] Record the full/offline installer decision as the required first release channel; document thin-runtime delivery as deferred unless all optional-channel gates are completed. Evidence: ADR/RFC/release pipeline.
- [*] Create a dedicated reproducible Python development/test environment and install `apps/llm-studio/api/requirements-dev.txt`. Evidence: API `.venv` runs the full suite/runtime tooling and `pip check` passes.
- [*] Run and capture the API baseline with `python -m pytest -q` from `apps/llm-studio/api`. Evidence: current full suite passes 262 tests.
- [*] Run and capture `npm run lint`, `npm run typecheck`, `npm run test:regression`, and `npm run build:desktop` from `apps/llm-studio/web`. Evidence: all passed; 69 regression tests and six validated routes.
- [*] Create an offline desktop-build baseline test and capture the current Google Fonts failure as a tracked issue. Evidence: remote Google fonts removed; offline/static build command and CI gate pass.
- [ ] Capture baseline screenshots and interaction recordings for all six user routes using `docs/llm-studio-web-route-parity-checklist.md`. EXPLICITLY EXCLUDED: browser-driven checks per product-owner direction.
- [*] Capture representative API request/response fixtures for every API group in the parity matrix, excluding secrets and sensitive data. Evidence: `docs/llm-studio-desktop-api-contract-fixtures.json` freezes 55 sanitized OpenAPI request/response contracts across all eight groups; `validate_api_contract_fixtures.py` validates classification, representative keys/evidence, secret absence, and reviewed drift in tests, Make, and CI.
- [*] Create small deterministic fixtures for a model project, tokenizer job, local training run, completed checkpoint, inference request, and mocked RunPod run. Evidence: API test fixtures and `scripts/desktop/smoke_runtime.py`.
- [ ] Document baseline startup time, idle memory, runtime disk footprint, small tokenizer-job time, and small local-training-job time on each required target. PARTIAL: the packaged smoke now writes and CI retains a structured per-target report with cold/warm readiness, backend-only idle RSS, payload size, and tiny tokenizer/local-training timings; required target execution and approved baselines remain external.
- [*] Confirm no generated baseline artifacts are committed unless intentionally added as fixtures. Evidence: build/output/runtime targets are ignored; only runtime `.gitkeep` is retained.
- [ ] Phase 0 exit gate: baseline commands, screenshots, API fixtures, platform decisions, and ownership records are complete and review-approved. BLOCKED: browser screenshots are explicitly excluded and organizational ownership/review/platform decisions remain external.

## Phase 1: Finalize Desktop Architecture and Threat Model

- [*] Write `docs/llm-studio-desktop-rfc.md` using this plan's selected architecture as the default.
- [*] In the RFC, explicitly state that the main UI remains on the stable bundled Tauri asset origin and does not navigate to a random loopback URL.
- [*] In the RFC, specify how the frontend obtains per-launch runtime configuration from a narrow Tauri bootstrap command.
- [*] Define the startup-state machine: shell start, runtime validation, backend spawn, readiness stages, UI activation, failure/retry, and shutdown.
- [*] Define a robust ephemeral-port startup handshake that avoids reserve-then-release races. Evidence: backend binds `127.0.0.1:0`, retains the socket, and atomically reports the actual port/PID.
- [*] Define the per-launch runtime-token flow, header name, in-memory lifetime, log-redaction behavior, and invalid-token response behavior.
- [*] Determine the exact bundled webview origin on every required platform and document the precise CORS/CSP rules needed for loopback fetches. Evidence: exact Tauri origins and restrictive CSP are recorded; target webview execution remains open.
- [*] Define the Tauri capability allowlist and command surface; prohibit broad filesystem, shell, and arbitrary command execution.
- [*] Define process-tree ownership and termination behavior on Unix process groups and Windows Job Objects.
- [*] Define app-close behavior for active tokenizer jobs, active local training, active inference, and active RunPod jobs.
- [*] Require an active-local-training close dialog with explicit “return to app” and “stop local work and exit” semantics; returning cancels exit and keeps work alive, and closing never silently kills training.
- [*] Require a RunPod close warning that explains potential continued billing and the current recovery limitation.
- [*] Decide whether background/tray execution is in v1. If not, explicitly prohibit it and make close behavior consistent.
- [*] Define how secrets are handled: per-launch token and pasted RunPod key never persist; Hugging Face dataset tokens remain execution-memory-only locally and are excluded from local/browser persistence, normalized config, job inputs, databases, logs, diagnostics, and remote bundles. A private-dataset RunPod launch necessarily sends the least-privilege token to the created pod environment as a documented provider trust boundary.
- [*] Define proxy, custom CA, corporate-network, and TLS trust behavior for Hugging Face, RunPod, updates, and optional runtime downloads.
- [*] Define privacy policy for diagnostics and telemetry. Evidence: local-only diagnostics; no telemetry/upload.
- [*] Threat-model malicious local websites, untrusted uploaded files, loopback request forgery, path traversal, archive extraction, update compromise, log leakage, and process orphaning.
- [*] Document the rationale for reusing Tauri v2 and the useful historical files in commits `4796a67` and `f60cedd`.
- [*] Document the historical defects listed in this plan as prohibited implementation patterns.
- [ ] Obtain architecture, security, release-engineering, and product approval on the RFC. BLOCKED EXTERNALLY: organizational review/sign-off.
- [ ] Phase 1 exit gate: the RFC and threat model are approved, and no unresolved decision can invalidate later packaging work. BLOCKED EXTERNALLY: required organizational approvals and portable-runtime decisions.

## Phase 2: Make Backend Paths and Runtime Discovery Packaging-Safe

- [*] Add a single backend runtime/source-root resolver in `apps/llm-studio/api/app/` with an explicit packaged-runtime environment variable such as `LLM_STUDIO_SOURCE_ROOT`. Evidence: `runtime_paths.py`.
- [*] Preserve a source-tree fallback for existing developer workflows.
- [*] Replace fixed `Path(__file__).resolve().parents[N]` import/root discovery in `app/main.py`, `app/config.py`, `app/tokenizer_jobs.py`, training routes, training preflight modules, local executor, and RunPod dataset handling.
- [*] Adapt root `training/` path discovery where it assumes a repository checkout.
- [*] Ensure packaged imports for `model`, `tokenizer`, and `training` do not depend on the current working directory. Evidence: packaged runtime import smoke.
- [*] Define and validate all required immutable runtime resources at startup: API templates, model templates/schemas, training templates/schemas, Python packages, and source modules.
- [*] Return a structured startup error naming every missing or incompatible runtime resource. Evidence: runtime-layout tests aggregate missing resources.
- [*] Change default tokenizer output and upload paths from API/repository-relative locations to writable app-data locations.
- [*] Update `.env.example` comments and configuration documentation to reflect the new defaults.
- [*] Add a one-time migration/import strategy for existing repository-local tokenizer uploads and artifacts so developer/user data is not silently lost. Evidence: copy-only/idempotent migration and marker tests.
- [*] Add path-normalization helpers for persisted tokenizer and training paths. Evidence: managed-path normalization/root enforcement in `storage_safety.py` plus typed managed-location codec/store boundaries in `managed_locations.py`.
- [*] Decide whether database records store absolute paths, data-root-relative paths, or typed locations. Prefer relocatable data-root-relative paths for managed files. Decision: data schema 3 preserves schema-2 versioned `llm-studio-data:v1/` typed, data-root-relative locations while preserving external/sentinel/remote values, and adds credential cleanup/physical compaction for prior managed state.
- [*] Migrate managed absolute paths safely while preserving external user-selected paths when applicable. Evidence: backup-first schema-1-to-2 SQLite migration rewrites only paths proven under data root; relocation, external/remote preservation, unsafe typed-location rejection, rollback, and idempotency tests pass.
- [*] Add database schema/version metadata and backup-before-migration behavior.
- [*] Add tests for source-tree runtime resolution.
- [*] Add tests for a synthetic packaged-runtime directory with no repository checkout.
- [ ] Add tests for paths containing spaces, Unicode, long names, and Windows drive-letter/UNC semantics where supported. PARTIAL: spaces/Unicode pass real runtime smoke and a synthetic long Unicode packaged path passes; Windows drive-letter/UNC evidence requires Windows.
- [*] Add tests proving no mutable files are written beneath packaged runtime resources. Evidence: the real-sidecar smoke snapshots packaged contents, file hashes, modes, timestamps, directory structure, and symlink identities before imports and proves no drift after workflows/restart/shutdown; focused tests prove mutation detection and no symlink following.
- [*] Add tests for fresh app-data initialization and existing-data migration. Evidence: schema/path/import migrations plus copy-first legacy default-database-name migration cover current-wins, custom paths, integrity failure cleanup, corrupt metadata, symlink rejection, and idempotency.
- [*] Add tests for database rollback/recovery when a migration fails.
- [*] Run the full API suite and package-layout characterization tests. Evidence: 262 tests passed plus linked and sanitized-portable real packaged-runtime smokes.
- [ ] Phase 2 exit gate: the API and training/tokenizer modules run from a synthetic packaged layout and all mutable data lives under configured writable roots. PARTIAL: synthetic layout, configured-root tests, and immutable-resource smoke pass; Windows path evidence remains external.

## Phase 3: Harden Backend Lifecycle for a Supervised Desktop Runtime

- [*] Add a desktop runtime mode flag such as `LLM_STUDIO_DESKTOP=1` and document every behavior it changes.
- [*] Keep the default bind address `127.0.0.1`; reject or fail closed if desktop mode is configured to bind `0.0.0.0` or a non-loopback interface.
- [*] Extend the readiness response to include backend version, API contract version, runtime version, data-schema version, startup stage, and compute capability summary.
- [*] Separate liveness from readiness so the shell can distinguish “process alive but loading PyTorch/migrating” from “ready for UI requests.”
- [*] Add a startup handshake/reporting mechanism that returns the actual bound port without a port-race window.
- [*] Ensure the runtime token is mandatory for protected APIs in desktop mode.
- [*] Confirm health/readiness endpoints expose no sensitive data and define whether they require the token. Evidence: root liveness is minimal; structured readiness is authenticated.
- [*] Restrict CORS in desktop mode to the exact tested Tauri asset origins; do not use a broad localhost regex unless proven necessary and documented.
- [*] Limit allowed request headers and methods where practical without breaking current parity. Evidence: desktop mode accepts only the exact token header and explicit methods.
- [*] Add structured backend logs with stable event IDs for startup, migrations, job lifecycle, provider actions, and shutdown. Evidence: rotating JSONL logger plus startup/migration/tokenizer/training/RunPod/shutdown events and recursive event-field redaction tests.
- [*] Redact runtime tokens, RunPod API keys, pod-agent tokens, authorization headers, query secrets, and sensitive dataset content from logs. Evidence: value- and key-based redaction covers structured fields, raw/JSON assignment strings, bare provider credentials, provider exceptions, local lifecycle logs/stdout, and remote-agent diagnostics. Job-scoped in-memory exact-value redaction additionally covers arbitrary legacy Hugging Face token values without recognizable provider prefixes.
- [*] Add log rotation by size/age/count, not only timestamped startup files. Evidence: rotating backend JSONL and size/count-rotated shell JSONL.
- [*] Add a diagnostics endpoint or narrow shell-readable status file containing non-secret runtime health information.
- [*] Define graceful shutdown sequencing for HTTP server, training manager, tokenizer executor, stores, and child processes.
- [*] Add explicit active-job status reporting the shell can query before close.
- [*] Ensure local training subprocesses are placed in a process group/job object owned by the desktop runtime.
- [ ] Ensure forced shutdown terminates the entire owned local-training process tree and does not kill unrelated processes. PARTIAL: Unix forced local-training and shell-descendant cleanup tests pass with `SIGTERM`-resistant descendants; a real backend parent-SIGKILL test proves active local-training-tree cleanup and unrelated-process protection. Windows Job Object/taskkill ownership is implemented but target-native Windows integration remains open.
- [*] Preserve current behavior that marks interrupted local jobs failed, but improve the error message to identify desktop shutdown versus crash where possible.
- [*] Add a recovery strategy for incomplete RunPod jobs or clearly expose recovery-limited state and remediation actions after restart.
- [*] Add disk-space checks before uploads, tokenizer jobs, local training, artifact bundling, migrations, updates, and optional runtime installs. Evidence: active managed-write/job/bundle paths and size-aware database/WAL/SHM migration backup preflight are covered; update and optional-runtime install paths are disabled and cannot execute.
- [*] Add clean handling for read-only data roots, full disks, database locks, port exhaustion, and corrupted runtime metadata. Evidence: typed/actionable 507/503 storage/database failures and bind errors plus low-disk, read-only-probe, locked/corrupt SQLite, occupied-port, and corrupt/missing-runtime tests.
- [*] Add backend tests for wrong/missing runtime token, non-loopback bind rejection, readiness states, graceful shutdown, forced shutdown, and process-tree cleanup. Evidence: backend auth/bind/readiness/lifecycle tests, real stop workflow, Unix forced local-training tree cleanup, and shell fake-sidecar descendant cleanup.
- [*] Add tests proving secrets never appear in captured logs. Evidence: backend formatter, provider exception, RunPod lifecycle file/stdout, persisted error, route-response, and remote-agent diagnostics regressions cover arbitrary assignments, bare `hf_`/`rpa_`/`rps_` credentials, and arbitrary non-prefixed execution tokens. Active training log responses redact exact job credentials; terminal transitions atomically scrub managed text outputs before releasing the in-memory scope.
- [ ] Phase 3 exit gate: a supervisor can start, authenticate, inspect, and stop the backend repeatedly without leaked processes, ports, secrets, or corrupted jobs. PARTIAL: fake-supervisor, repeated real-sidecar lifecycle/token-rotation, current-target forced process-tree, Unix abrupt-parent-death cleanup with active local training and unrelated-process protection, and fault-injection tests pass; Windows/Linux and target-native abrupt-shell evidence remain open.

## Phase 4: Refactor the Frontend for Runtime-Provided Desktop Connectivity

- [*] Create a single runtime-config module under `apps/llm-studio/web/lib/` that represents environment kind, API base URL, runtime token, desktop capabilities, and version information.
- [*] Preserve web-development defaults and production web behavior.
- [*] Add a startup provider/hook that retrieves runtime configuration from Tauri only when running inside the desktop shell.
- [*] Prevent route workflows from issuing API requests until runtime configuration is ready; show a clear startup/retry state instead.
- [*] Replace module-load API-base/token constants in `lib/api.ts` with runtime-config consumption.
- [*] Replace module-load API-base/token constants in `lib/tokenizerLegacyApi.ts` with runtime-config consumption.
- [*] Replace module-load API-base/token constants in `lib/training/client.ts` with runtime-config consumption.
- [*] Refactor `lib/training/generation.ts` so streamed NDJSON requests use runtime config and token headers.
- [*] Centralize authenticated request creation, error normalization, abort handling, and desktop runtime-unavailable errors. Evidence: `runtimeConfig.ts` owns authenticated raw/JSON requests, validation-detail parsing, invalid-JSON/network/runtime-unavailable/abort normalization, while preserving domain-specific error factories; focused regressions pass.
- [*] Replace synchronous direct artifact URL construction with a download abstraction capable of authenticated fetch plus native Save As.
- [*] Update workspace asset downloads to use the shared download abstraction.
- [*] Update model project artifact downloads to use the shared download abstraction.
- [*] Update tokenizer artifact downloads and remove tokenizer-only desktop special casing once the shared abstraction is proven.
- [*] Update training artifact downloads to use the shared download abstraction.
- [*] Preserve browser anchor/Blob fallbacks for the web edition.
- [*] Create a narrow desktop bridge module; no route component should access raw `window.__TAURI__` or unrestricted invoke directly.
- [*] Preserve all current localStorage keys and ensure runtime bootstrapping does not clear or re-origin them.
- [*] Add a runtime-config failure/retry UI that provides “retry backend,” “open logs,” and “open diagnostics” through narrow desktop capabilities.
- [ ] Verify clipboard, file input, drag/drop, browser confirm behavior, keyboard shortcuts, custom events, and streamed responses in desktop webviews. EXPLICITLY EXCLUDED: browser/webview checks.
- [*] Add unit tests for web runtime defaults, desktop runtime bootstrap, token injection, runtime failure, and authenticated download behavior. Evidence: source-level regressions cover narrow bootstrap/retry contracts, URL normalization, fail-closed stale-token clearing/recovery, header-only token injection, no persistence, native cancellation, authenticated native download/reveal, and browser Blob fallback.
- [ ] Add tests proving secret tokens are not rendered, persisted to localStorage, included in URLs, or logged. PARTIAL: header-only URL tests, explicit no-localStorage bootstrap/retry source tests, Hugging Face form/config non-persistence regressions, log redaction tests, and static/release-output scans that reject secret-bearing localStorage/sessionStorage keys pass; live webview persistence inspection is excluded.
- [ ] Run web lint, typecheck, regression tests, and route-parity tests. PARTIAL: lint/typecheck/69 regressions/static build pass; browser route-parity tests are explicitly excluded.
- [ ] Phase 4 exit gate: one runtime-config path serves every API, streaming, and artifact workflow while the normal web edition remains behaviorally unchanged. PARTIAL: source/static/non-browser tests pass; behavioral webview/browser parity is explicitly excluded.

## Phase 5: Make the Static Frontend Deterministic and Tauri-Compatible

- [*] Vendor Sora and IBM Plex Mono locally or replace `next/font/google` with a reproducible local-font strategy. Evidence: remote font dependency replaced with system font stacks.
- [*] Prove `npm run build:desktop` succeeds with network access disabled. Evidence: offline build target/CI configuration and static build pass.
- [*] Pin/install frontend dependencies through the existing lockfile and fail CI on lockfile drift.
- [*] Define the exact static-export route strategy for Tauri, including whether `trailingSlash` or another resolver is required. Evidence: static export with `trailingSlash` and route validator.
- [ ] Verify direct navigation and in-app navigation for `/`, `/simple`, `/studio`, `/tokenizer`, `/training`, and `/inference`. EXPLICITLY EXCLUDED: browser/webview checks; all six route files are statically validated.
- [ ] Verify refresh/reload on every route inside packaged Tauri builds on every platform. EXPLICITLY EXCLUDED: browser/webview checks.
- [ ] Verify query-parameter deep links: `/studio?project=`, `/tokenizer?job=`, and `/training?project=&tokenizerJob=&run=`. EXPLICITLY EXCLUDED: browser/webview checks.
- [ ] Verify static assets, icons, CSS, Next client chunks, and route chunks load through the Tauri asset protocol without HTTP fallback. EXPLICITLY EXCLUDED: browser/webview checks.
- [*] Confirm no Next server, Node.js process, or dev server is required in the shipped app. Evidence: Tauri `frontendDist` consumes validated static output and native release build passes.
- [*] Add a static-export contents validator that fails when expected routes/assets are missing or server-only routes appear.
- [*] Add a test that searches desktop output for accidental build-time API tokens, local developer paths, and secrets.
- [ ] Define a stable desktop webview storage partition/identifier and test persistence across app restart and upgrade. PARTIAL: stable Tauri identifier `com.llmbuilder.studio` is defined and documented; live persistence tests are explicitly excluded.
- [*] Add a localStorage migration/version mechanism only if necessary; preserve every key listed in the persistence contract. Evidence: active tokenizer keys use the LLM Studio namespace; one-time source-tested migrations copy historical Tokenizer Studio form/job/theme state before removal, preserve active values, retain legacy data on failed writes, and migrate theme before first paint.
- [ ] Verify responsive layout at the desktop minimum window size and common high-DPI scaling levels. EXPLICITLY EXCLUDED: browser/webview checks.
- [ ] Verify WebView2 behavior on Windows and WebKit/WebKitGTK behavior on macOS/Linux for file inputs, drag/drop, charts, streaming fetch, and downloads. BLOCKED: target environments and browser/webview checks.
- [ ] Capture desktop screenshots and compare them against approved route baselines. EXPLICITLY EXCLUDED: browser checks.
- [ ] Phase 5 exit gate: the bundled static frontend builds offline, loads every route from a stable asset origin, and preserves current browser state and behavior. PARTIAL: offline static build and stable bundled origin pass; live route/state behavior is explicitly excluded.

## Phase 6: Restore and Adapt the Tauri v2 Desktop Shell

- [*] Recover useful Tauri configuration, icons, supervisor patterns, and documentation from commit `4796a67` into a new `apps/llm-studio/desktop/` implementation.
- [*] Rename all Tokenizer Studio identifiers, environment variables, commands, labels, paths, and documentation to LLM Studio equivalents. Evidence: all active desktop/product/storage/database surfaces use LLM Studio names; historical identifiers remain only in explicit, tested compatibility aliases, migration maps, fixtures, and historical-plan context.
- [*] Configure Tauri `frontendDist` to the validated LLM Studio static export.
- [*] Keep a small native/bundled startup view or overlay available until runtime readiness completes; do not navigate away from the asset origin.
- [*] Implement a typed supervisor state machine in Rust with explicit startup, ready, failed, stopping, and stopped states.
- [*] Implement runtime discovery and validation for development override and bundled runtime resources.
- [*] Validate runtime manifest schema, target platform/architecture, versions, required files, and checksums before spawn.
- [*] Generate a cryptographically secure per-launch API token in Rust.
- [*] Spawn the packaged Python backend with explicit desktop-mode, source-root, data-root, cache-root, token, log, and startup-handshake environment values.
- [*] Avoid inheriting unsafe or irrelevant development environment variables into the child process.
- [*] Implement staged startup progress that distinguishes runtime validation, migration, Python import/PyTorch initialization, backend bind, and API readiness.
- [*] Use realistic startup timeouts for first launch and slower machines, with progress and cancellation rather than a single unexplained 45-second timeout.
- [*] Implement bounded restart/retry behavior and prevent rapid crash loops.
- [*] Implement shell and backend log files with rotation and redaction.
- [*] Implement narrow commands for runtime bootstrap status, retry, stop, active-job status, open logs, open data folder, and diagnostics export.
- [*] Implement graceful application exit with the active-job decision behavior from the RFC.
- [*] Implement Unix process-group handling and Windows Job Object handling.
- [ ] Prove forced shell termination does not leave the backend or local training process tree running after OS cleanup semantics complete. PARTIAL: owned supervisor termination, OS process-group/Job Object setup, and real Unix abrupt-parent-death cleanup of the backend plus active local-training tree pass focused tests; target-native shell-kill and Windows/Linux validation remain open.
- [*] Implement single-instance behavior or explicitly support multiple instances with isolated ports and a database locking strategy. Prefer single-instance for v1.
- [*] Handle second-launch requests by focusing the existing window and forwarding supported deep links. Evidence: duplicate launches restore/show/focus the existing window; external URL schemes are intentionally unsupported/deferred.
- [ ] Configure minimum/default window size based on route-layout testing and persist window geometry only if it is robust. PARTIAL: default `1440x960` and minimum `1080x720` are configured; route-layout validation is explicitly excluded and geometry persistence is deferred until proven robust.
- [*] Implement native menu items for quit, reload/retry backend, open logs, open data folder, and about/version information where platform conventions require them. Evidence: narrow Tauri menu actions; quit reuses active-job close handling and retry uses the trusted event bridge.
- [ ] Add shell unit tests for runtime selection, manifest validation, token generation, startup transitions, log paths, and shutdown decisions. PARTIAL: 29 shell tests cover tokens, unique temporary names, runtime and managed-artifact path/symlink escape, compatibility/portability/size metadata, retries, cancellation, redaction, close warning, atomic writes, authenticated streaming, bounded log rotation, bounded aggregate storage inventory with large-file and symlink behavior, fake-sidecar transitions, invalid handshake/port, bad token, owned shutdown, forced descendant cleanup, and unrelated-process protection; target-native app-handle/log-directory integration remains open.
- [*] Add supervisor integration tests using a fake sidecar for ready, timeout, crash, wrong version, bad token, port collision, and shutdown cases. Evidence: fake-sidecar ready/crash/timeout/invalid schema/invalid port/bad token/owned termination tests plus manifest compatibility tests.
- [ ] Phase 6 exit gate: the Tauri shell reliably supervises a fake and real packaged backend while keeping the bundled UI origin stable. PARTIAL: fake-sidecar, real packaged-sidecar lifecycle, Unix abrupt-parent-death cleanup with active local training, and static origin configuration pass; live target-native shell/webview abrupt-kill gates remain open.

## Phase 7: Implement Cross-Platform Native Integrations

- [*] Select audited Tauri v2 plugins or narrow custom Rust commands for dialogs, filesystem writes, and opening folders/files.
- [*] Replace the historical macOS-only `osascript` Save As implementation with a supported cross-platform native dialog implementation.
- [*] Implement a single Save As flow for model project exports, tokenizer artifacts, training artifacts, diagnostics bundles, and any future downloads.
- [*] Ensure downloaded artifact responses remain authenticated and tokens never appear in URLs.
- [*] Sanitize suggested filenames and reject path traversal or invalid target paths.
- [*] Handle overwrite confirmation, cancellation, permission errors, full disks, and interrupted writes. Evidence: native dialog cancellation/overwrite and atomic write/error paths.
- [*] Use atomic writes where practical for exported files and diagnostics bundles.
- [*] Preserve browser download behavior in the normal web edition.
- [*] Implement “Open Data Folder,” “Open Logs Folder,” and “Reveal Artifact” with narrow allowed paths. Evidence: data/log commands plus typed artifact-route metadata lookup, canonical app-data containment, symlink-escape rejection, and desktop-only workspace reveal action.
- [*] Prevent arbitrary path opening from untrusted backend or frontend input.
- [ ] Verify native file upload/picker and browser file input flows both work; do not require native path access for parity. EXPLICITLY EXCLUDED: browser/webview interaction checks; multipart managed-upload real-sidecar smoke passes.
- [*] Decide whether native notifications are required for long jobs; if implemented, request minimal permission and avoid sensitive content. Decision: deferred for v1; status remains in-app.
- [ ] Verify clipboard behavior and keyboard shortcuts on every platform. EXPLICITLY EXCLUDED: browser/webview interaction checks and target-platform matrix.
- [ ] Verify drag/drop behavior does not accidentally navigate the webview away from the app. EXPLICITLY EXCLUDED: browser/webview interaction checks.
- [ ] Add unit/integration tests for filename sanitization, allowed-path enforcement, cancellation, overwrite, and write failures. PARTIAL: sanitization, managed-path and reveal-path/symlink enforcement, typed reveal endpoints, authenticated streaming, source-level native cancellation/browser-fallback contracts, atomic create/overwrite, missing-directory failure, low-disk/read-only storage failures, and bounded log rotation pass; native-dialog permission/cancellation integration remains target-native E2E work.
- [ ] Add end-to-end tests for every artifact/export/download action on every required platform. EXPLICITLY EXCLUDED/BLOCKED: live browser/webview checks and required target platforms; non-browser native command contracts pass.
- [ ] Phase 7 exit gate: every file-related user workflow works consistently on macOS, Windows, Linux, and the unchanged web edition. BLOCKED: target-native and explicitly excluded browser/webview E2E evidence.

## Phase 8: Build a Reproducible Platform Runtime

- [*] Define a Python version and support lifecycle compatible with FastAPI, datasets, tokenizers, PyTorch, and every required target. Evidence: Python 3.12 baseline in ADR and CI.
- [ ] Select the embedded/redistributable Python strategy for each platform and document licenses and redistribution obligations. BLOCKED EXTERNALLY: reviewed redistributable builds are not selected; generic venv construction is characterization-only.
- [ ] Create platform-specific locked constraints or wheel manifests instead of relying on open-ended `>=` requirements during release builds. BLOCKED EXTERNALLY: reviewed target wheelhouses/constraints are still required.
- [ ] Define the exact PyTorch package/channel per platform and compute support promise. PARTIAL: compute promise is documented; exact reviewed target package/channel remains part of locked-runtime work.
- [ ] Build each runtime on its target OS/architecture; do not cross-build ML runtimes and assume equivalence. PARTIAL: macOS arm64 linked runtime passed; target-native CI matrix is defined but Windows/Linux evidence is external.
- [*] Package `apps/llm-studio/api/app`, `apps/llm-studio/api/templates`, `model`, `tokenizer`, and `training`.
- [*] Exclude repository-only data, tests, caches, `.pyc`, credentials, local artifacts, remote-agent implementation, Docker files, and unrelated sources unless explicitly needed. Evidence: source-copy filters plus portable-runtime sanitization remove dependency tests/caches/bytecode and build-only entry points before hashing.
- [*] Include immutable templates/schemas and validate them in runtime smoke tests.
- [*] Include a runtime manifest with shell compatibility range, API version, schema version, platform, architecture, Python version, dependency versions, file hashes, and build provenance.
- [*] Generate an SBOM and dependency-license inventory for each platform runtime.
- [*] Scan runtime dependencies and bundled binaries for known vulnerabilities and establish a triage policy. Evidence: target-runtime pip-audit, Cargo Audit, npm audits, SBOM/license inventory, and pinned `cargo-auditable` native-shell scanning are gated; the complete shell scan recovers 291 included crates with zero vulnerabilities and seven visible applicable warnings. All 19 lock-level Rust warnings and 80 non-vulnerability wheel payload path findings remain visible release-review inputs rather than suppressed findings.
- [ ] Make runtime builds deterministic enough that unexpected file/version changes fail review. PARTIAL: two identical complete linked-runtime builds compare byte-for-byte and symlink-for-symlink, while manifest hashes/provenance expose changes; reproducible locked wheelhouses are still required.
- [*] Ensure the runtime never installs packages or mutates itself in signed application resources. Evidence: portable construction removes and verifies absence of `pip`, strips bytecode/build-only artifacts before hashing, and all post-sanitization probes plus the full smoke disable bytecode writes.
- [ ] Add runtime size reporting and fail on unexplained size regressions above an approved threshold. PARTIAL: builder records/enforces deterministic payload counts against checked-in development guardrails; release builds fail without an exact approved target threshold, and staging independently recomputes and enforces it. Approved release target baselines remain external.
- [*] Add runtime import smoke tests for FastAPI, SQLAlchemy, datasets, tokenizers, torch, `model`, `tokenizer`, and `training`.
- [*] Add runtime API smoke tests for health, model validation/analysis, tokenizer validation, and training preflight.
- [*] Add a tiny deterministic tokenizer-job smoke test.
- [*] Add a tiny deterministic local-training plus inference smoke test that completes within CI constraints.
- [ ] Add CPU capability and macOS MPS characterization tests matching the approved matrix. PARTIAL: readiness and each retained packaged-runtime characterization report record CPU/MPS/CUDA capabilities; approved target characterization matrix remains external.
- [*] Add tests for clean offline launch with already-local fixtures. Evidence: runtime smoke uses local fixtures and no runtime dependency install.
- [ ] Verify HTTPS/TLS requests use appropriate system/certifi trust behavior on each platform. BLOCKED: target-native corporate/TLS environments.
- [*] Verify runtime paths containing spaces and non-ASCII characters. Evidence: real runtime smoke uses a Unicode/space temporary root.
- [ ] Phase 8 exit gate: a versioned runtime built on each target passes import, API, tokenizer, training, inference, offline, and path smoke tests without a repository checkout. PARTIAL: macOS arm64 linked-development and sanitized portable-unlocked characterization runtimes pass the full smoke; reviewed locked portable runtimes and Windows/Linux evidence remain external.

## Phase 9: Complete Security Hardening

- [*] Set a restrictive Tauri CSP; prohibit `csp: null`.
- [*] Allow scripts, styles, images, fonts, and loopback connections only as required by tested app behavior.
- [*] Configure Tauri capabilities per window and command; remove unused default/plugin permissions. Evidence: main window grants event listen/unlisten only; native commands are narrow custom commands.
- [*] Ensure remote HTTP/HTTPS content cannot invoke desktop commands.
- [*] Ensure the backend accepts desktop requests only on loopback and rejects missing/invalid per-launch tokens.
- [*] Ensure CORS permits only exact required bundled origins.
- [*] Ensure tokens are carried in headers, never query strings or download URLs.
- [ ] Ensure tokens and secrets never persist in localStorage, sessionStorage, crash reports, diagnostics, shell logs, backend logs, or release artifacts. PARTIAL: unit/static/log/release-audit checks pass, including rejection of secret-bearing web-storage writes; execution-only Hugging Face credential tests prove exclusion from normalized responses, SQLite, managed job files, preflight data, and decompressed remote bundles. Exact-value regressions additionally prove arbitrary non-prefixed tokens are redacted from tokenizer failures, active training logs, runner state, remote diagnostics, terminal managed files, and artifact ZIPs. Schema v3 scrubs and compacts legacy active/retained/backup databases, checkpoints/removes WAL/SHM sidecars, and sanitizes managed job state. Live webview and signed installer inspection remain open.
- [*] Review upload filename sanitization, file stats, artifact downloads, and static-file serving for traversal and symlink escapes.
- [*] Review project deletion, tokenizer job deletion, training job deletion, cache cleanup, migrations, and update cleanup for allowed-root enforcement. Evidence: every enabled project/tokenizer/training deletion and migration cleanup path is confined to a typed or validated managed root with escape/symlink tests; cache cleanup is explicitly user-directed only while the app is closed, and updater/downloaded-runtime activation and cleanup surfaces remain disabled.
- [*] Validate archive extraction against traversal, symlinks, decompression bombs, and disk exhaustion before enabling optional runtime downloads. Evidence: downloaded-runtime extraction is disabled; release staging rejects unsafe paths/symlinks and optional channel cannot activate.
- [*] Sign runtime manifests separately from publishing checksums if optional downloaded runtimes are enabled. Decision: optional downloaded runtimes remain disabled until this external signing gate exists.
- [*] Pin update endpoints and require signed Tauri updater artifacts. Decision: updater remains disabled until signed endpoint/rollback work is approved.
- [*] Add dependency, secret, SBOM, and license scanning to CI. Evidence: target-native runtime pip-audit, Cargo Audit, npm audits, static-output/compiled-shell payload scans, SBOM/license generation, and release audit generation are required; native binary triage remains tracked separately in Phase 8.
- [ ] Perform a focused security review or penetration test of the desktop boundary. BLOCKED EXTERNALLY: independent security review/penetration-test owner and environment.
- [ ] Resolve all critical/high findings and explicitly accept any lower-severity residual findings. BLOCKED EXTERNALLY: depends on independent review findings and authorized risk acceptance.
- [ ] Phase 9 exit gate: security review approves loopback API, Tauri capability surface, file operations, secrets, and update trust. BLOCKED EXTERNALLY: security approval and signed update trust infrastructure.

## Phase 10: Preserve Every User Workflow Through Desktop Parity Tests

- [ ] Create a desktop parity test suite that maps one-to-one to `docs/llm-studio-web-route-parity-checklist.md`. EXPLICITLY EXCLUDED: browser/webview parity automation; non-browser API/runtime/static coverage is recorded separately.
- [ ] Add `/` tests for load, navigation, theme persistence, UI-mode persistence, workspace refresh/error states, filters/sorts, and all available artifact downloads. EXPLICITLY EXCLUDED: browser/webview route checks.
- [ ] Add `/simple` tests for fresh flow, persisted flow, preset selection, step readiness, tokenizer transition, training transition, inference transition, and expert-route links. EXPLICITLY EXCLUDED: browser/webview route checks; focused non-browser simple-mode regressions pass.
- [ ] Add `/studio` tests for builder load, drag interactions, config edits, backend validation, backend analysis, save/update/list/open/delete project, import, export, undo/redo, diagnostics, and `?project=` hydration. EXPLICITLY EXCLUDED: browser/webview route checks; API/static/non-browser coverage passes.
- [ ] Add `/tokenizer` tests for persisted forms, `?job=` hydration, tokenizer validation, dataloader validation, local upload, drag/drop, file stats, streaming dataset/filter/weight editing, job launch, polling, preview, recent-job behavior, delete, metadata, and native/browser artifact download. EXPLICITLY EXCLUDED: browser/webview route checks; real-sidecar tokenizer workflow passes.
- [ ] Add `/training` tests for persisted config/selection, query hydration, model/tokenizer pickers, local dataset, streaming datasets, validation, preflight, recommendations, local launch, RunPod launch with fake provider, polling, stop, metrics, logs, samples, checkpoints, recent runs, remote actions, delete, and artifact download. EXPLICITLY EXCLUDED: browser/webview route checks; real-sidecar local workflow and source fake-provider tests pass.
- [ ] Add `/inference` tests for completed-run search, checkpoint search, settings, synchronous error handling, streamed NDJSON generation, completion, cancellation, and missing-artifact behavior. EXPLICITLY EXCLUDED: browser/webview route checks; real-sidecar synchronous/streamed inference passes.
- [ ] Add app-restart tests proving all current localStorage keys and backend workspace history persist. PARTIAL: real packaged-sidecar restart preserves project/tokenizer/training history and artifacts while rotating the launch token; live webview localStorage persistence is explicitly excluded.
- [ ] Add app-upgrade tests proving localStorage, SQLite databases, paths, projects, jobs, artifacts, and caches survive supported upgrades. BLOCKED EXTERNALLY: no released predecessor exists and live webview localStorage checks are excluded.
- [ ] Add crash/restart tests during idle, tokenizer job, local training, inference, and mocked RunPod training. PARTIAL: real-sidecar restart persistence, interrupted-job recovery, and real Unix backend parent-death cleanup with active local training pass; abrupt target-native app/OS workflow matrix remains external.
- [ ] Add close-dialog tests for active local training and RunPod billing warnings. PARTIAL: Rust active-job warning/owned-shutdown logic passes; live dialog interaction is explicitly excluded.
- [ ] Add offline tests for local-only workflows and clear network-error behavior for Hugging Face/RunPod workflows. PARTIAL: real local-only runtime smoke passes; browser-facing network-error parity remains excluded.
- [*] Add low-disk, read-only data root, corrupt DB, corrupt runtime, missing runtime file, locked DB, and backend-crash tests. Evidence: focused storage/database/runtime tests plus fake-sidecar crash coverage pass.
- [ ] Add accessibility checks for startup/error dialogs and all existing route behavior. EXPLICITLY EXCLUDED: browser/webview interaction and accessibility checks.
- [ ] Add high-DPI, minimum-window, and common display-scaling visual checks. EXPLICITLY EXCLUDED: browser/webview visual checks and target display matrix.
- [ ] Add tests for two successive launches using different backend ports while retaining the same webview/localStorage state. PARTIAL: repeated real-sidecar launch/token rotation and stable Tauri identifier are proven; live webview localStorage persistence is explicitly excluded.
- [*] Ensure default CI RunPod tests use fakes and never create billable resources.
- [ ] Add an opt-in, budget-capped, auto-cleaned real RunPod smoke workflow with explicit credentials and human approval. BLOCKED EXTERNALLY: approved credentials, budget, protected workflow, and human cleanup monitoring.
- [ ] Record parity results in a platform matrix and attach screenshots/logs to release candidates. BLOCKED: release candidates/target environments and explicitly excluded screenshots.
- [ ] Phase 10 exit gate: every required route/API/workflow parity test passes on every required platform with no unexplained behavior difference from web. BLOCKED: required platform matrix, real RunPod, and explicitly excluded browser/webview parity checks.

## Phase 11: Build Developer Workflow and Diagnostics

- [*] Add documented one-command desktop development startup that runs the Tauri shell against the local API/runtime without modifying production security defaults.
- [*] Add one-command static frontend build, runtime build, runtime smoke, shell build, installer build, and complete local verification targets.
- [*] Extend `apps/llm-studio/Makefile` or add clearly named root-level desktop commands without breaking existing targets.
- [*] Keep platform-specific packaging logic in maintainable scripts or build tooling; do not pretend a Unix shell script is a real Windows build.
- [*] Add a command that validates required toolchains and prints actionable installation/version errors for contributors.
- [*] Add a command that prints desktop shell, web, backend, runtime, Python, PyTorch, platform, architecture, and schema versions.
- [*] Add a diagnostics export that includes redacted logs, manifests, version/capability data, health summaries, migration status, and bounded aggregate data/cache/log storage counts.
- [*] Ensure diagnostics never include tokens, API keys, raw user prompts/data, filenames, or full sensitive paths unless explicitly approved and redacted. Evidence: exported supervisor errors pass through secret/path redaction; storage inventory emits only bounded aggregate counts/bytes and never follows symlinks.
- [*] Add an in-app startup failure screen with retry, open logs, open data folder, export diagnostics, and quit.
- [ ] Add user-facing errors for missing runtime, incompatible runtime, failed migration, insufficient disk, failed backend start, unsupported compute, and unavailable WebView2/system dependencies. PARTIAL: runtime/backend/storage/compute failures are actionable; a missing system webview cannot render an in-app message and requires installer/platform handling.
- [*] Document how developers run API tests, web tests, shell tests, runtime tests, and desktop E2E tests. Evidence: release pipeline documents the authoritative non-browser command and explicitly records the browser exclusion.
- [*] Document how to update Python/runtime dependencies and regenerate locks/SBOMs.
- [ ] Phase 11 exit gate: a new contributor can build, run, test, diagnose, and package the desktop app from the documentation. PARTIAL: documented local commands pass; independent new-contributor and reviewed portable-package validation remain external.

## Phase 12: Create Cross-Platform CI and Release Packaging

- [*] Add CI workflows for macOS arm64, Windows x64, and Linux x64 using target-native runners.
- [*] Cache dependencies without caching mutable build outputs that could compromise reproducibility.
- [*] Run web lint, typecheck, regression, offline static export, and static-output validation in CI.
- [*] Run the full API test suite and packaged-layout tests in CI.
- [*] Run Tauri/Rust formatting, linting, unit tests, and supervisor integration tests in CI. Evidence: target-native workflow runs format, clippy `-D warnings`, all 29 shell/fake-sidecar tests, path-remapped release compile, and compiled-shell payload audit.
- [*] Build and smoke-test each platform runtime in CI. Evidence: target-native characterization builds are sanitized, fully smoked, and Python-dependency audited before release-manifest generation.
- [ ] Build desktop installers from the already-smoke-tested matching runtime and static frontend artifacts. BLOCKED: reviewed portable runtimes are not available; release staging correctly rejects the linked-development runtime.
- [ ] Run installer launch and basic E2E smoke tests on target-native runners or clean VMs. BLOCKED EXTERNALLY: target-native signed installers/clean VMs and browser/webview interaction checks.
- [ ] Configure macOS Developer ID signing, hardened runtime, entitlements, and notarization. BLOCKED EXTERNALLY: signing identity, entitlements review, protected release environment, and Apple notarization.
- [ ] Configure Windows code signing, timestamping, reputation strategy, and WebView2 prerequisite/bootstrap behavior. BLOCKED EXTERNALLY: signing identity, release infrastructure, Windows environment, and product/release decisions.
- [ ] Select and document Linux package targets, WebKitGTK/system dependency expectations, desktop entries, icons, and uninstall behavior. BLOCKED EXTERNALLY: release/product package-target decision and target validation.
- [ ] Generate SHA-256 checksums, signatures, SBOMs, license notices, provenance, and release manifests for every artifact. PARTIAL: generators and local evidence exist; signatures/final artifacts require external release infrastructure.
- [ ] Verify installers contain no credentials, developer paths, repository-local data, caches, or unintended files. PARTIAL: path-remapped compiled shell and static frontend payload audit pass locally and in CI; sanitized portable characterization payload has no build-generated cache/bytecode/entry-point findings but records 80 upstream wheel/example home-path findings requiring reviewed-wheelhouse triage; final extracted signed installers remain external.
- [ ] Verify uninstall removes application binaries but preserves user data by default, with documented manual data removal. BLOCKED EXTERNALLY: signed target-native installers and clean-machine uninstall tests; retention policy is documented.
- [*] Define `alpha`, `beta`, and `stable` release channels and promotion rules.
- [*] Configure the signed Tauri application updater only after signing and rollback tests pass. Decision enforced: updater remains disabled until these gates pass.
- [ ] Test shell updates that retain the bundled runtime and data. BLOCKED EXTERNALLY: signed updater remains intentionally disabled until release gates pass.
- [ ] Test coordinated shell/runtime compatibility failures with clear rollback. PARTIAL: compatibility failures are rejected before launch; signed updater/rollback testing remains external.
- [*] If optional downloaded runtimes are implemented, add signed manifests, atomic staging, disk checks, progress UI, retry/resume, proxy support, compatibility validation, previous-version retention, and automatic rollback. Decision enforced: optional downloaded runtimes are not implemented/enabled.
- [*] Never activate an optional downloaded runtime in place; promote an already-verified staged version atomically. Decision enforced by absence of a downloaded-runtime activation path.
- [*] Keep at least one known-good runtime until the new runtime has passed readiness and smoke checks. Decision enforced: only immutable bundled runtimes are allowed for v1; downloaded activation is disabled.
- [ ] Add release-candidate installation tests on clean, non-developer machines for every target. BLOCKED EXTERNALLY: release candidates, clean target machines, and signing.
- [ ] Phase 12 exit gate: CI produces signed/installable, target-native artifacts with complete provenance and passing clean-machine smoke tests. BLOCKED EXTERNALLY: reviewed portable runtimes, signing/notarization, publication, and clean-machine tests.

## Phase 13: Validate Upgrades, Recovery, Performance, and Supportability

- [*] Define supported upgrade paths, including at minimum previous stable to current stable.
- [*] Create golden test data from pre-desktop/source-tree layouts and every released desktop schema version. Evidence: checked-in `tests/fixtures/desktop-migrations/pre-desktop-schema-1.json` exercises copy-only source-tree import and schema-1 absolute-path migration; no desktop schema version has been released yet, and future released schemas must add fixtures before migration changes.
- [ ] Test data migration, backup, rollback, and re-run idempotency for each supported upgrade. PARTIAL: schema-1-to-3 typed-location and physical credential-cleanup migration, relocation, size-aware backup-capacity preflight, backup/retained-legacy sanitization, SQLite WAL checkpoint/sidecar cleanup/compaction, transactional rollback, failed-backup safety, copy-only import, unsafe-location rejection, and idempotency pass; released-version restore/upgrade testing requires release predecessors.
- [*] Test app downgrade behavior and block unsafe downgrade rather than corrupting newer data.
- [ ] Test update interruption at download, staging, install, first launch, migration, and readiness phases. BLOCKED EXTERNALLY: signed updater remains intentionally disabled.
- [ ] Test database corruption recovery messaging and backup restoration. PARTIAL: corrupt SQLite/schema metadata messaging, migration backup, failed-backup safety, downgrade rejection, and documented coherent-backup recovery pass; an automated full-database restore UX remains intentionally unimplemented pending product approval.
- [ ] Test app crash and OS restart behavior with active local and mocked RunPod jobs. PARTIAL: interrupted local/recovery-limited backend behavior and real Unix parent-SIGKILL cleanup with active local training are covered; target-native app/OS restart matrix remains external.
- [ ] Test recovery when the loopback port is unavailable or endpoint security software blocks the backend. PARTIAL: ephemeral bind, real occupied-port actionable failure, invalid-port handshake, and failed-start behavior pass; endpoint-security target tests remain external.
- [ ] Test very large caches/artifacts enough to validate disk reporting, cleanup, and diagnostics behavior. PARTIAL: shell tests prove privacy-preserving byte/count reporting for a 128 MiB sparse artifact, entry/time-bounded scans, missing roots, and no symlink traversal; approved scale thresholds plus long-duration/target storage and cleanup validation remain external.
- [*] Test corporate proxy/custom CA behavior or document unsupported configurations explicitly. Evidence: preserved environment-variable contract and troubleshooting guidance are documented; target validation remains a release matrix item.
- [ ] Measure cold/warm startup, first API readiness, idle memory, route navigation, tiny job times, installer size, runtime size, and update size on every target. PARTIAL: packaged smoke and target-native CI emit retained structured evidence for backend cold/warm first-API readiness, backend-only idle RSS, tiny tokenizer/local-training times, compute capabilities, and runtime payload size; browser route timing, shell/webview memory, reviewed installer/update size, and completed target execution remain external.
- [ ] Establish release thresholds for startup time, memory, disk, and unexplained regressions. BLOCKED EXTERNALLY: product/release approval and target baselines.
- [ ] Profile and address startup bottlenecks such as PyTorch import and data migrations. PARTIAL: authenticated readiness/logs/diagnostics and retained characterization reports record per-stage cold/warm timings, including pre-lifespan application/PyTorch imports and migrations/store initialization; target measurements, bottleneck-remediation decisions, and approved thresholds remain external.
- [ ] Verify logs rotate and app data does not grow without bounds except for user-created artifacts/caches with visible cleanup controls. PARTIAL: shell/backend logs rotate, diagnostics expose bounded aggregate data/cache/log growth, and cleanup guidance exists; large-data/long-duration growth testing remains open.
- [*] Add a cache-management/data-location view or provide clearly documented folder/cleanup controls before GA. Evidence: Open Data/Open Logs commands and troubleshooting cleanup guidance.
- [*] Write `docs/llm-studio-desktop-user-guide.md`.
- [*] Write `docs/llm-studio-desktop-troubleshooting.md`.
- [*] Write `docs/llm-studio-desktop-release-pipeline.md`.
- [*] Write support guidance for logs, diagnostics, data backup, cache cleanup, RunPod billing/recovery, reinstall, and complete uninstall.
- [ ] Phase 13 exit gate: upgrades, failure recovery, performance thresholds, diagnostics, and support documentation pass review. BLOCKED EXTERNALLY: released predecessors, target recovery/performance evidence, and organizational review.

## Phase 14: Staged Rollout and General Availability

- [ ] Publish an internal alpha only after Phases 0-12 gates are complete. BLOCKED EXTERNALLY: prior release gates and publication infrastructure.
- [ ] Run alpha on every required platform with real user workflows and collect structured parity/failure feedback. BLOCKED EXTERNALLY: alpha artifacts, target users/environments, and release program.
- [ ] Resolve all release-blocking alpha issues and repeat the full platform matrix. BLOCKED EXTERNALLY: depends on alpha execution and findings.
- [ ] Publish a limited beta with signed installers, update channel, release notes, known limitations, and support path. BLOCKED EXTERNALLY: signing, updater, publication, and support ownership.
- [ ] Monitor beta startup failures, crashes, migration failures, failed downloads, process leaks, and workflow parity defects using the approved privacy model. BLOCKED EXTERNALLY: beta release/monitoring program; telemetry upload is intentionally absent.
- [ ] Resolve all critical/high defects and explicitly disposition lower-severity defects before stable. BLOCKED EXTERNALLY: depends on alpha/beta/security findings and authorized acceptance.
- [ ] Run final clean-machine install, upgrade, rollback, uninstall, offline, local training, tokenizer, inference, and mocked/opt-in RunPod tests. BLOCKED EXTERNALLY: signed release candidates, clean target machines, approved RunPod credentials/budget, and browser checks.
- [ ] Confirm every phase exit gate and every Definition of Done item is `[*]`. BLOCKED: external and explicitly excluded gates remain open.
- [ ] Obtain final product, engineering, QA, security, release, and support sign-off. BLOCKED EXTERNALLY: organizational approvals.
- [ ] Publish stable signed artifacts, checksums/signatures, SBOMs, release notes, user guide, and troubleshooting guide. BLOCKED EXTERNALLY: signing/publication infrastructure and final approvals.
- [ ] Tag the release and archive the exact web/runtime/shell build provenance. BLOCKED EXTERNALLY: final approved release.
- [ ] Phase 14 exit gate: stable desktop LLM Studio is published for every approved required platform and meets the full parity contract. BLOCKED EXTERNALLY: all staged-rollout and publication gates.

## Detailed Acceptance Gates

These gates are intentionally redundant with phase steps. They provide a final, concise release audit.

### Startup and Process Gate

- [ ] Fresh install launches without Python, Node.js, or repository checkout. BLOCKED EXTERNALLY: reviewed portable runtime, signed installer, and clean-machine test.
- [*] Backend binds only to `127.0.0.1` on a collision-safe ephemeral port.
- [*] Bundled UI remains on a stable Tauri asset origin.
- [*] Per-launch token is required and remains memory-only.
- [*] Startup progress and failures are actionable.
- [ ] Normal and forced exits leave no owned backend/training processes. PARTIAL: owned graceful/forced supervisor, local-training descendant termination, and real Unix backend parent-death cleanup with active local training and unrelated-process protection pass focused tests; target-native abrupt app/OS termination remains open.

### Data Gate

- [*] All mutable data is stored in OS app-data/cache locations.
- [*] No mutable writes occur in application resources. Evidence: packaged runtime is hash-validated at startup, portable build/audit probes and runtime launch disable bytecode writes, mutable roots are separate, and complete linked plus sanitized-portable real-sidecar smokes prove runtime trees remain unchanged.
- [ ] Browser localStorage state persists across restarts and upgrades. EXPLICITLY EXCLUDED/BLOCKED: live webview persistence checks and released upgrade predecessor.
- [ ] Projects, tokenizer jobs, training jobs, artifacts, databases, and managed paths survive supported upgrades. PARTIAL: real-sidecar restart persistence and relocatable schema-1-to-3 typed-location/credential-cleanup migration tests pass; released-version upgrade evidence remains external.
- [*] Migrations create backups and fail safely.

### Functional Gate

- [ ] All six routes pass the parity matrix on every required platform. EXPLICITLY EXCLUDED/BLOCKED: browser/webview parity checks and required target environments; static route validation passes.
- [ ] All API groups pass packaged-runtime characterization tests. PARTIAL: every local core group, local stop/cancellation, static-serving compatibility, RunPod catalog, missing-key provider resources, and deterministic recovery-limited remote lifecycle contracts pass real-sidecar smoke; real RunPod provider actions remain external.
- [*] Local tokenizer training completes and exports an artifact.
- [*] Local model training completes and produces metrics/logs/checkpoints/artifacts.
- [*] Inference produces streamed output from a completed run/checkpoint.
- [ ] RunPod workflows pass fake-provider tests and the approved opt-in real smoke. PARTIAL: source fake-provider/recovery tests and packaged missing-key/recovery-limited lifecycle contracts pass; approved real smoke is external.
- [ ] Every download/export uses a working native flow with web fallback preserved. PARTIAL: all artifact/export call sites use the shared abstraction; narrow authenticated native streaming, managed reveal, and browser fallback contract tests pass; live native-dialog/webview E2E remains explicitly excluded or target-native.

### Security Gate

- [*] CSP and Tauri capability allowlists are restrictive and reviewed.
- [*] Loopback API rejects unauthenticated requests.
- [ ] Secrets are absent from URLs, persistence, logs, diagnostics, and artifacts. PARTIAL: automated unit/static/log/diagnostics/release-audit evidence passes, including exact arbitrary-token redaction/scrubbing, execution-only dataset-credential and legacy physical-cleanup regressions, artifact-ZIP inspection, and secret-bearing web-storage-key rejection; live webview and signed installer inspection remain open.
- [*] File/archive operations resist traversal and allowed-root escapes.
- [ ] Installers, updates, and optional runtime manifests are signed and verified. BLOCKED EXTERNALLY: signing/update infrastructure; optional downloaded runtimes remain disabled.

### Release Gate

- [*] Offline static frontend build passes.
- [ ] Platform-native runtime build and smoke tests pass. PARTIAL: macOS arm64 linked and sanitized portable-unlocked characterization runtimes pass and target-native CI is defined; reviewed locked Windows/Linux/macOS release runtimes remain external.
- [ ] Clean-machine installer tests pass. BLOCKED EXTERNALLY: signed installers and clean target machines.
- [ ] Signing/notarization and updater rollback pass. BLOCKED EXTERNALLY: signing identities, protected release infrastructure, and approved updater rollout.
- [ ] Checksums, signatures, SBOM, licenses, provenance, docs, and release notes are published. PARTIAL: local checksums/SBOM/licenses/provenance/docs generation passes; signatures, final release notes, and publication remain external.

## Risk Register

| Risk | Impact | Required mitigation |
|---|---|---|
| PyTorch/runtime size and platform variance | Large installers, broken local compute | Target-native locked runtimes, explicit compute matrix, size gates, full/offline first |
| Dynamic backend port changes UI origin | Lost localStorage and broken IPC | Keep UI on stable bundled Tauri asset origin; runtime-config bootstrap |
| Fixed source-tree parent paths | Packaged runtime crashes | Central explicit source-root resolver plus synthetic packaged-layout tests |
| Tokenizer output/upload defaults point into API tree | Writes fail in signed/read-only bundle | Move defaults to app data and migrate existing data |
| Local training spawns child processes | Orphaned jobs or silent interruption | Process groups/Windows Job Objects, active-job close dialog, forced cleanup tests |
| RunPod process-memory credentials | Limited restart recovery and billing risk | Explicit warnings, recovery-state UX, remote lifecycle tests; keychain only via approved design |
| Broad Tauri/loopback permissions | Local privilege/security exposure | Narrow commands/capabilities, exact CORS/CSP, per-launch token, security review |
| Google Fonts build dependency | Non-reproducible/offline release failure | Vendor fonts and enforce offline static build |
| SQLite/path migrations | Lost workspace history/artifacts | Versioned migrations, backups, relative managed paths, upgrade/downgrade tests |
| Thin-runtime download compromise/failure | Code execution or unusable first launch | Defer for first release; signed manifests, atomic staging, rollback, proxy/retry/disk handling |
| WebView platform differences | Route/workflow behavior drift | Target-native E2E and visual parity matrix |
| Linux dependency fragmentation | Install/launch failures | Explicit distro/glibc/WebKitGTK support matrix and clean-machine tests |
| Code signing/notarization delays | Release blocked | Establish credentials and dry-run pipeline early |

## Expected File/Directory Changes

The exact structure may be refined by the approved RFC, but implementation should expect:

```text
apps/llm-studio/
  desktop/
    package.json
    src/
    src-tauri/
      Cargo.toml
      capabilities/
      src/
      tauri.conf.json
  api/
    app/
      runtime_paths.py            # or equivalent centralized resolver
      desktop_runtime.py          # or equivalent desktop lifecycle/status support
  web/
    lib/
      runtimeConfig.ts            # or equivalent single runtime config surface
      desktopBridge.ts            # narrow bridge only
      downloads.ts                # shared authenticated/native download path
scripts/
  desktop/
    ... target-aware build, package, smoke, and release tooling ...
docs/
  llm-studio-desktop-rfc.md
  llm-studio-desktop-user-guide.md
  llm-studio-desktop-troubleshooting.md
  llm-studio-desktop-release-pipeline.md
```

Do not create generic abstractions unless they remove real duplication across the existing clients/routes or are required by the desktop boundary.

## Required Verification Commands

The executor may add commands, but the final pipeline must provide stable equivalents for:

```text
# Existing web checks
cd apps/llm-studio/web
npm run lint
npm run typecheck
npm run test:regression
npm run build:desktop

# Existing API checks
cd apps/llm-studio/api
python -m pytest -q

# Planned desktop checks, names finalized during implementation
make desktop-check
make desktop-build-web-offline
make desktop-build-runtime
make desktop-smoke-runtime
make desktop-test-shell
make desktop-test-e2e
make desktop-package
```

Every release build must run from a clean checkout or controlled build workspace and must produce provenance tying the installer to exact source, web, runtime, lockfile, and toolchain versions.

## Appendix A: Endpoint-by-Endpoint Packaged Runtime Parity Checklist

Every endpoint below must have a packaged-runtime characterization or end-to-end test. Mark an endpoint `[*]` only after authentication, success behavior, relevant failure behavior, and desktop caller behavior are verified.

### System and Model Studio

- [*] `GET /health` passes packaged-runtime liveness checks without exposing secrets.
- [*] `GET /api/v1/health` passes authenticated API health checks.
- [*] `GET /api/v1/config/templates` returns the packaged model template.
- [*] `GET /api/v1/config/schemas` returns the packaged model schema.
- [*] `POST /api/v1/validate/model` preserves validation behavior and errors.
- [*] `POST /api/v1/analyze/model` imports packaged model code and preserves analysis behavior.
- [*] `POST /api/v1/projects` creates a project under managed app data.
- [*] `PUT /api/v1/projects/{project_id}` updates a managed project.
- [ ] `GET /api/v1/projects` lists projects after restart and upgrade. PARTIAL: real packaged-runtime restart/list succeeds; released-version upgrade evidence remains open.
- [*] `GET /api/v1/projects/{project_id}` returns a managed project.
- [ ] `GET /api/v1/projects/{project_id}/artifact` downloads through the authenticated/native desktop path. PARTIAL: packaged authenticated response, narrow native streaming, and managed reveal contract tests pass; live webview Save As is excluded.
- [*] `DELETE /api/v1/projects/{project_id}` deletes only the allowed managed project path.

### Tokenizer Workspace

- [*] `GET /api/v1/tokenizer/health` passes packaged tokenizer health checks.
- [*] `GET /api/v1/tokenizer/config/templates` returns packaged tokenizer/dataloader templates.
- [*] `GET /api/v1/tokenizer/config/schemas` returns packaged tokenizer/dataloader schemas.
- [*] `POST /api/v1/tokenizer/validate/tokenizer` preserves validation behavior and errors.
- [*] `POST /api/v1/tokenizer/validate/dataloader` preserves validation behavior and errors.
- [*] `POST /api/v1/tokenizer/files/train` stores uploaded data in managed app data. Evidence: real packaged-runtime multipart smoke with traversal-like source filename.
- [*] `GET /api/v1/tokenizer/files/stats` resolves and reports allowed files. Evidence: real packaged-runtime stats smoke.
- [*] `POST /api/v1/tokenizer/files/validation` stores validation data in managed app data. Evidence: real packaged-runtime multipart smoke.
- [*] `POST /api/v1/tokenizer/jobs` runs against the packaged tokenizer implementation.
- [ ] `GET /api/v1/tokenizer/jobs` preserves completed/recent job history across restart and upgrade. PARTIAL: real packaged-runtime restart/list succeeds; released-version upgrade evidence remains open.
- [*] `GET /api/v1/tokenizer/jobs/{job_id}` preserves polling/status/error behavior.
- [*] `DELETE /api/v1/tokenizer/jobs/{job_id}` deletes only allowed managed job/artifact paths.
- [*] `POST /api/v1/tokenizer/jobs/{job_id}/preview` preserves tokenizer preview behavior.
- [*] `GET /api/v1/tokenizer/jobs/{job_id}/artifact/meta` preserves artifact metadata.
- [ ] `GET /api/v1/tokenizer/jobs/{job_id}/artifact` downloads through the authenticated/native desktop path. PARTIAL: packaged authenticated response, narrow native streaming, and managed reveal contract tests pass; live webview Save As is excluded.

### Training Configuration and Providers

- [*] `GET /api/v1/training/health` passes packaged training health checks.
- [*] `GET /api/v1/training/config/templates` returns packaged training/dataloader templates.
- [*] `GET /api/v1/training/config/schemas` returns packaged training/dataloader schemas.
- [*] `POST /api/v1/training/validate/dataloader` preserves validation behavior and errors.
- [*] `POST /api/v1/training/validate/training-config` preserves validation behavior and errors.
- [*] `POST /api/v1/training/validate/preflight` imports packaged model/training code and preserves recommendations/issues.
- [*] `GET /api/v1/training/providers/runpod/defaults` preserves provider defaults.
- [*] `GET /api/v1/training/providers/runpod/status` preserves configured/memory-only key status behavior.
- [ ] `GET /api/v1/training/providers/runpod/catalog` preserves catalog behavior through fake and opt-in real-provider tests. PARTIAL: source fake-provider catalog tests pass; approved opt-in real-provider smoke is external.
- [ ] `POST /api/v1/training/providers/runpod/validate-key` preserves validation while preventing secret persistence/logging. PARTIAL: mocked validation plus structured/raw provider-error redaction and memory-only persistence tests pass; approved real credential smoke is external.
- [ ] `GET /api/v1/training/providers/runpod/pods` preserves provider resource listing. PARTIAL: source fake-provider route tests pass and recursively redact provider-echoed credentials; approved real provider smoke is external.
- [ ] `GET /api/v1/training/providers/runpod/network-volumes` preserves provider volume listing. PARTIAL: source fake-provider route tests pass and recursively redact provider-echoed credentials; approved real provider smoke is external.

### Training Jobs and Inference

- [ ] `POST /api/v1/training/jobs` launches packaged local training and fake/opt-in RunPod training. PARTIAL: real packaged local training and source fake-provider tests pass; opt-in real RunPod is external.
- [ ] `GET /api/v1/training/jobs` preserves run history across restart and upgrade. PARTIAL: real packaged-runtime completed-run restart/list succeeds; released-version upgrade evidence remains open.
- [ ] `GET /api/v1/training/jobs/{job_id}` preserves polling, local, RunPod, and recovery-limited states. PARTIAL: packaged local polling/completed/restart states and source RunPod/recovery-limited tests pass; real provider lifecycle remains external.
- [*] `DELETE /api/v1/training/jobs/{job_id}` deletes only allowed managed run/artifact paths.
- [*] `GET /api/v1/training/jobs/{job_id}/metrics` preserves metrics behavior.
- [*] `GET /api/v1/training/jobs/{job_id}/samples` preserves samples behavior.
- [*] `GET /api/v1/training/jobs/{job_id}/logs` preserves logs behavior without exposing secrets.
- [*] `GET /api/v1/training/jobs/{job_id}/checkpoints` preserves checkpoint behavior.
- [*] `POST /api/v1/training/jobs/{job_id}/generate` preserves synchronous generation behavior.
- [*] `POST /api/v1/training/jobs/{job_id}/generate/stream` preserves NDJSON streaming, errors, and cancellation. Evidence: packaged start/token/done streaming passes; focused async regressions cover setup/iteration errors, disconnect detection between token computations, deterministic iterator close, and sanitized started/completed/cancelled/failed lifecycle events.
- [ ] `POST /api/v1/training/jobs/{job_id}/stop` stops local process trees and preserves remote stop behavior. PARTIAL: valid active local run is cancelled/deleted in real packaged-runtime smoke and a forced descendant-tree test passes; remote provider stop remains external.
- [ ] `POST /api/v1/training/jobs/{job_id}/remote/resync` preserves remote artifact resync. PARTIAL: source fake-provider behavior passes; real provider resync is external.
- [ ] `POST /api/v1/training/jobs/{job_id}/remote/cleanup` preserves explicit provider cleanup semantics. PARTIAL: source fake-provider cleanup behavior passes; real provider cleanup is external.
- [ ] `POST /api/v1/training/jobs/{job_id}/remote/reattach` preserves supported reattach behavior and clear recovery limitations. PARTIAL: source recovery-limited behavior passes; real provider lifecycle validation is external.
- [ ] `GET /api/v1/training/jobs/{job_id}/artifact` downloads through the authenticated/native desktop path. PARTIAL: packaged authenticated ZIP, narrow native streaming, and managed reveal contract tests pass; live webview Save As is excluded.

### Backend Static-Web Compatibility

The shipped desktop UI uses Tauri `frontendDist`, but existing web/static-serving behavior must not regress:

- [*] `GET /` still serves configured static web output when backend web serving is enabled. Evidence: real packaged-runtime static-serving smoke.
- [*] `GET /{asset_path}` still serves safe configured assets and rejects traversal/API fallback misuse. Evidence: real packaged-runtime asset/deep-link/missing-asset/API-fallback/traversal smoke.
- [*] Normal web development and backend-served static deployments remain functional after desktop changes. Evidence: web quality/static build and packaged backend static-serving smoke pass.

## Primary Technical References

- Existing repository parity baseline: `docs/llm-studio-web-baseline.md`
- Existing route parity checklist: `docs/llm-studio-web-route-parity-checklist.md`
- Existing frontend architecture conventions: `docs/llm-studio-web-architecture-note.md`
- Existing RunPod architecture/security/troubleshooting documents under `docs/`
- Historical desktop work: `git show 4796a67:<path>` and `git show f60cedd:<path>`
- Tauri v2 + Next.js: <https://v2.tauri.app/start/frontend/nextjs/>
- Tauri sidecars: <https://v2.tauri.app/develop/sidecar/>
- Tauri capabilities/security: <https://v2.tauri.app/security/capabilities/>
- Tauri updater: <https://v2.tauri.app/plugin/updater/>
- Next.js static export: <https://nextjs.org/docs/app/guides/static-exports>

## Final Executor Rule

- [ ] Before declaring the desktop application complete, re-read this entire plan, confirm every required item is marked `[*]` with evidence, rerun the full clean-machine platform matrix, and obtain all final sign-offs. BLOCKED: external and explicitly excluded gates remain open, so the application must not be declared complete.
