# LLM Studio Desktop Architecture and Threat Model

- Date: 2026-06-12
- Status: Implemented baseline; architecture/security/release approval pending

## Architecture

LLM Studio Desktop uses Tauri v2, the existing Next.js static export, and a
supervised target-native Python/FastAPI runtime. The main webview always remains
on the bundled Tauri asset origin. It never navigates to the random loopback
backend port.

The frontend calls only a narrow bridge in `web/lib/desktopBridge.ts`.
`runtime_bootstrap` returns the live API base URL, memory-only per-launch token,
capabilities, and compatible versions. `web/lib/runtimeConfig.ts` is the only
connectivity source for normal API requests, streamed inference, and downloads.

The desktop application identifier is fixed as `com.llmbuilder.studio`. It is
the stable webview/storage identity and must not change after release without an
explicit storage-migration plan. Active tokenizer and theme storage use LLM
Studio-namespaced keys; first-run compatibility migration copies historical
Tokenizer Studio values before removing them and never overwrites current data.

## Runtime Layout

```text
runtime/
  manifest.json
  VERSION
  sbom.json
  licenses.json
  python/
    bin/python | Scripts/python.exe
  source/
    apps/llm-studio/api/{app,templates,requirements.txt}
    model/
    tokenizer/
    training/
```

`manifest.json` declares schema/API/data versions, shell compatibility, target,
Python/dependency versions, required files, SHA-256 hashes, and provenance. The
shell validates it before spawning Python. Development may set
`LLM_STUDIO_RUNTIME_DIR`; production resolves only bundled resources.

Release-portable runtimes require a target-specific exact requirements lock with
SHA-256 hashes and a reviewed offline wheelhouse. The lock and a hashed
wheelhouse inventory are copied into the runtime and covered by its manifest.
Unlocked network-resolved characterization runtimes are explicitly non-release
and cannot be staged into an installer.
Portable construction uses `pip` only as a build tool, verifies dependency
consistency, removes `pip`, console/activation scripts, tests, caches, and
bytecode, then inventories and hashes the sanitized runtime. Post-sanitization
probes and runtime launch disable bytecode writes.

## Managed Data Locations

Data schema 3 preserves schema-2 paths owned by the application as typed,
relocatable locations in SQLite:
`llm-studio-data:v1/<data-root-relative-posix-path>`.
Store boundaries resolve those locations against the current configured data
root and enforce containment before returning usable absolute paths. Sentinels,
external user-selected paths, artifact bundle filenames, and RunPod remote
workspace paths remain unchanged.

The schema migration creates a coherent pre-migration database backup, then
transactionally rewrites only absolute paths proven to be below the configured
data root. Schema 3 additionally removes legacy Hugging Face dataset tokens and
redacts provider credentials from active, retained legacy, and backup databases,
checkpoints/removes WAL/SHM sidecars, physically compacts sanitized databases,
sanitizes managed job inputs/logs, and removes transient remote bundles that
could retain credentials. Active tokenizer/training jobs additionally register
their exact execution-only token values in a job-scoped, process-memory
redaction registry. This protects arbitrary legacy token formats in backend
logs and API errors; training terminal transitions scrub managed text outputs
before releasing the scope. It is
idempotent, fails without advancing schema metadata, and blocks unsafe downgrade
from a newer schema.

Active default databases are `db/llm_studio_tokenizer.db` and
`db/llm_studio_training.db`. Before stores initialize, a separate idempotent
name migration copies historical Tokenizer Studio defaults through SQLite's
backup API, verifies the copy, atomically promotes it, retains the legacy source,
and never overwrites a current or custom database.

## Startup State Machine

1. Tauri loads the bundled static UI and displays the runtime gate.
2. Shell resolves and validates the target runtime and hashes.
3. Shell creates managed data/cache/log directories and rotates backend logs.
4. Shell generates a cryptographically random 256-bit token.
5. Shell starts Python in an owned Unix process group or Windows Job Object with
   an allowlisted environment and explicit desktop paths. On Unix it also passes
   its exact PID so the backend can reject an unmanaged launch and monitor parent
   identity.
6. Python binds `127.0.0.1:0` itself, retains the socket, and atomically writes a
   PID/host/port handshake. This avoids reserve-then-release races.
7. Shell validates the handshake PID/loopback URL and polls authenticated
   `/api/v1/health` through staged readiness. Readiness includes sanitized
   per-stage milliseconds and total time to ready, beginning before application
   and PyTorch imports so cold-start bottlenecks are visible in diagnostics.
8. Shell emits non-secret startup progress for data-path preparation, runtime
   validation, Python/compute loading, loopback bind, migration/readiness, and
   completion. Startup runs off the UI thread and can be cancelled.
9. Shell returns runtime config to the bundled UI. API workflows remain blocked
   until this succeeds.
10. Failure retains the bundled UI and exposes retry, logs, data, diagnostics,
   and quit actions.

Startup timeout is 180 seconds to cover cold PyTorch imports on slower machines.
Retry terminates the owned process tree first and does not create a crash loop.
The shell enforces one application instance. A second launch restores and
focuses the existing main window. External URL schemes and argument forwarding
remain disabled until their parsing, authorization, and route contracts are
reviewed.

## Non-Browser Characterization

The authoritative packaged-runtime smoke can write an atomic, path-redacted
characterization report outside the immutable runtime. The report records
target/runtime identity and payload size, backend compute capabilities,
client-observed cold and warm first-API readiness, backend startup-stage timing,
three backend-only idle RSS samples, deterministic tiny tokenizer and local
training create-to-terminal times, and total smoke time.

These measurements are characterization evidence, not approved release
thresholds. They intentionally exclude shell/webview memory, route-navigation
timing, installer size, and update size. Browser/webview checks remain excluded
by product-owner direction, while target matrices and thresholds require release
review. CI retains one report per target-native characterization runtime.

## Network and Secret Contract

- Backend desktop bind: loopback only; non-loopback configuration fails closed.
- Protected API header: `X-LLM-Studio-Token`.
- Historical web-development token-header aliases are rejected in desktop mode.
- Token lifetime: generated in Rust per backend launch, retained only in Rust and
  frontend memory, never placed in URLs or persisted.
- Desktop CORS origins: exactly `tauri://localhost` and
  `https://tauri.localhost`; no desktop localhost regex.
- Tauri CSP allows bundled assets and loopback API connections only. It prohibits
  frames, objects, remote scripts, and a null CSP.
- Desktop frontend API requests cross the authenticated `runtime_request` Tauri
  command instead of relying on WebView loopback `fetch`, which WKWebView can
  reject for custom-scheme pages. The shell accepts only relative runtime paths,
  four API methods, and the `Accept` and `Content-Type` request headers, then
  injects the memory-only runtime token before forwarding to its owned backend.
- Pasted RunPod keys remain API-process memory only. They are not included in
  diagnostics or persistence.
- Proxy/custom-CA behavior follows explicitly preserved OS environment variables:
  proxy variables, `SSL_CERT_FILE`, `SSL_CERT_DIR`, `REQUESTS_CA_BUNDLE`, and
  `CURL_CA_BUNDLE`.

## Narrow Native Surface

Allowed application commands:

- `runtime_bootstrap`, `runtime_request`, `retry_runtime`, `active_jobs`
- `runtime_status`, `cancel_runtime_start`
- `save_file`, `save_api_artifact`, `reveal_api_artifact`
- `open_logs_folder`, `open_data_folder`
- `export_diagnostics`
- `quit_app`, `stop_and_exit`

There is no arbitrary shell execution, arbitrary path open, unrestricted
filesystem command, or remote-content command access. Save As uses a
cross-platform native dialog, sanitizes suggested names, and writes through a
same-directory temporary file before promotion.

## Shutdown and Recovery

The shell queries authenticated active-job state before closing. Active work
prevents close and presents a return-to-app action plus an explicit exit action.
For local work, exit terminates the entire owned runtime/process tree. For
RunPod work, the exit label and warning state that the remote pod may continue
billing and that automatic reattach is recovery-limited because raw pod-agent
tokens are not persisted.
Background/tray execution is prohibited in v1.

On normal or forced owned shutdown, Unix process groups and Windows Job Objects
cover backend/local-training descendants. On Unix, the backend also watches the
exact supervising shell parent; abrupt parent death requests graceful Uvicorn
shutdown so lifespan cleanup stops executors and closes stores. Existing
interrupted local jobs are marked failed at next startup. Mutable stores remain
in app data and survive restart/update.

## Threat Model

| Threat | Control |
| --- | --- |
| Malicious local website calls loopback API | Mandatory random header token, exact CORS, loopback bind |
| Remote content invokes native commands | Bundled-only UI, restrictive CSP, minimal capability file |
| Runtime/update compromise | Manifest target/version checks, SHA-256 files, target-native pipeline; signing remains release gate |
| Path traversal from manifests or filenames | Reject absolute/parent manifest paths; sanitize Save As suggestion; typed managed locations and backend root validation |
| Archive traversal/decompression bomb | Downloaded runtimes are deferred; no archive extraction exists in v1 |
| Secret leakage | Token in memory/header only; diagnostics omit token/key/path detail; clean child environment |
| Process orphaning | Unix process group, exact-parent watchdog, and Windows Job Object ownership |
| Silent RunPod billing | Active-job close interception and explicit warning |
| Corrupt or incomplete runtime | Fail before spawn with all validation failures and recovery actions |
| Untrusted upload | Backend filename sanitization and managed upload directory; content remains application input |

## Historical Patterns Explicitly Prohibited

The useful Tauri v2/supervisor structure from commits `4796a67` and `f60cedd` was
reviewed, but these defects must not return:

- navigating the main UI to a loopback URL;
- reserve-then-release port selection;
- `csp: null`;
- optional desktop API authentication;
- macOS-only `osascript` Save As;
- repository-relative mutable data;
- an open-ended runtime dependency install during app launch.

## Privacy and Diagnostics

Diagnostics are local, user-triggered JSON exports. There is no telemetry or
automatic upload. Diagnostics contain versions, lifecycle state, target, a
redacted last error, non-secret health/migration state, capability declarations,
a redacted runtime-manifest summary, log-file counts, aggregate data/cache/log
file counts and bytes, and a bounded set of structured redacted shell events.
Storage inventory scans are capped by both entry count and elapsed time, report
whether they completed, and count but never follow symlinks. They exclude
tokens, API keys, prompts, datasets, raw backend logs, filenames, and full
sensitive paths. Diagnostics snapshot
supervisor state first, then collect health/log/storage data off-thread without
holding the lifecycle lock.

The native supervisor writes stable-event, redacted `shell.jsonl` records with
size/count rotation. The backend independently writes rotating structured JSONL
logs. Neither log intentionally records runtime tokens or user content.
