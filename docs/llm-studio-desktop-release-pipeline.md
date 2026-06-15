# LLM Studio Desktop Build and Release Pipeline

## Local Verification

```bash
make -C apps/llm-studio desktop-verify-nonbrowser
make -C apps/llm-studio desktop-check
make -C apps/llm-studio desktop-versions
make -C apps/llm-studio desktop-check-api-contract
make -C apps/llm-studio desktop-build-web-offline
make -C apps/llm-studio desktop-build-runtime
make -C apps/llm-studio desktop-smoke-runtime
make -C apps/llm-studio desktop-test-shell
make -C apps/llm-studio desktop-build-shell
make -C apps/llm-studio desktop-audit-dependencies
make -C apps/llm-studio desktop-release-manifest ARTIFACTS="path/to/installer ..."
make -C apps/llm-studio desktop-audit-payload PAYLOADS="path/to/extracted-payload ..."
```

`desktop-verify-nonbrowser` is the authoritative complete local verification
command under the current product-owner restriction. It runs API tests, web
lint/typecheck/regressions/offline static build, runtime build/smoke, shell
format/clippy/tests/release compile, npm audits, Python dependency consistency,
Python and Rust vulnerability audits with the reviewed narrow policy, release
payload audits for the static frontend and compiled shell, release-staging
rejection for the linked runtime, frozen API request/response contract
validation, runtime-size guardrails, and `git diff --check`.

Authenticated runtime readiness includes a sanitized startup profile with
per-stage milliseconds and total time to ready. It begins before FastAPI imports
the application so expensive Python/PyTorch import cost is not hidden, and
separately records runtime validation, managed-data preparation, and
migrations/store initialization. `smoke_runtime.py` asserts and prints this
profile for every target-native runtime; release threshold decisions still
require reviewed target measurements.

`desktop-smoke-runtime` also writes
`build/desktop/release/performance-characterization-<target>.json`. The atomic,
path-redacted report records runtime payload size and compute capabilities,
client-observed cold/warm first-API readiness, backend startup-stage timing,
three backend-only idle RSS samples, deterministic tiny tokenizer/local-training
create-to-terminal times, and total smoke time. It explicitly marks shell/webview
memory, browser route timing, installer size, update size, and approved
thresholds as unavailable or external. A report path inside the immutable
runtime is rejected.

`validate_api_contract_fixtures.py` regenerates the authoritative API contract
from FastAPI OpenAPI and compares it to
`docs/llm-studio-desktop-api-contract-fixtures.json`. The catalog classifies all
55 registered operations into the eight parity-matrix groups, retains sanitized
request/response schemas plus representative request/response subsets, rejects
unclassified operations or concrete secrets, and requires explicit review
before contract drift is accepted:

```bash
python scripts/desktop/validate_api_contract_fixtures.py
python scripts/desktop/validate_api_contract_fixtures.py --update  # only after review
```

Release shell builds run through `scripts/desktop/build_shell.py`, which preserves
explicit Rust compiler flags while remapping workspace and developer-home source
paths. It invokes Tauri through a temporary cross-platform Cargo runner backed
by pinned `cargo-auditable` 0.7.4, so the actual release binary embeds complete
dependency metadata while retaining path remapping. The builder resolves the
platform-native npm executable explicitly, including `npm.cmd` on Windows. This prevents Rust
dependency panic-location strings from leaking local build paths and lets the
native-binary vulnerability gate inspect every included crate.

The default runtime target is linked-development mode for fast local testing.
It is never a release artifact. Bare interpreter command names such as
`python3` are resolved through `PATH` before the runtime output is replaced, so
the Makefile fallback works without creating broken repository-relative links.

Browser-driven checks are intentionally excluded from this execution and from
the default CI workflow by product-owner direction. Static output validation,
route manifest validation, and non-browser regressions remain required. The
browser parity gates in the execution plan remain open until that direction is
revisited.

## Dependency and Lock Updates

Dependency changes are deliberate release-engineering work, never app-startup
behavior:

1. Update frontend or shell declarations, regenerate the matching npm/Cargo
   lockfile, review the complete lock diff, run both npm audits, then run
   `desktop-verify-nonbrowser`.
2. Update Python packages only alongside reviewed target-specific fully hashed
   locks and offline wheelhouses. Rebuild each target runtime, review its
   manifest/SBOM/license diff, then run its full runtime smoke.
3. Reject unexplained lock, hash, SBOM, license, or provenance drift. The shipped
   application never installs or updates packages.
4. Run `audit_dependencies.py` against each completed target-native runtime.
   Python findings fail unless they exactly match the reviewed policy. Cargo
   vulnerabilities fail; transitive maintenance warnings remain visible and
   require release review.

Python runtime audits derive a complete exact-version requirements inventory
from the target interpreter, disable dependency resolution, and reject scanner
output that does not cover every installed package. PEP 440 local versions fail
closed unless a narrow reviewed policy matches both the package/local version
and the runtime manifest provenance. The current policy permits only official
`torch==...+cpu` wheels from unlocked Windows/Linux characterization runtimes,
auditing them under the corresponding public PyTorch version while preserving
the installed local version in the report.

The current policy records one narrow pip-audit scanner mismatch:
`CVE-2025-3000` is accepted only for `torch` versions strictly newer than
`2.6.0`, because the linked OSV record identifies `2.6.0` as the last affected
version. No other package, version, or vulnerability is ignored.

The repository pins Rust 1.88.0. The lock uses `time` 0.3.47, which resolves
RUSTSEC-2026-0009. Cargo Audit currently reports zero vulnerabilities and 19
visible transitive informational warnings: 17 unmaintained and two unsound
dependencies inherited through the Tauri/WebKit stack. These warnings are not
silently ignored; every release must review their current dependency paths and
available replacements.

Every release build runs `cargo audit bin` against the completed shell and fails
if `cargo-auditable` metadata is missing or incomplete. The current macOS arm64
shell scan recovers all 291 included crates, reports zero vulnerabilities, and
keeps seven applicable transitive maintenance/unsoundness warnings visible.
This binary scan complements the 538-crate lock scan; neither warning set is
silently suppressed.

## Target-Native Release Runtime

On each target OS/architecture, use a reviewed Python 3.12 redistributable and
locked wheelhouse:

```bash
python scripts/desktop/build_runtime.py \
  --portable \
  --install-dependencies \
  --lock build/desktop/locks/<target>.txt \
  --wheelhouse build/desktop/wheelhouse \
  --runtime-version "$VERSION"
python scripts/desktop/smoke_runtime.py \
  build/desktop/runtime/<target> \
  --performance-report build/desktop/release/performance-characterization-<target>.json
python scripts/desktop/stage_runtime.py build/desktop/runtime/<target>
npm ci --prefix apps/llm-studio/desktop
python scripts/desktop/build_shell.py
```

The builder emits `manifest.json`, `VERSION`, CycloneDX-style `sbom.json`,
`licenses.json`, provenance, required-file declarations, and SHA-256 hashes. The
runtime is immutable after signing and never installs packages at app launch.
It also records deterministic payload file/byte counts and the threshold used
from `scripts/desktop/runtime-size-policy.json`. Linked and unlocked
characterization builds use conservative development guardrails to catch
accidental caches, test trees, duplicate runtimes, and other unexplained growth.
Release-portable construction fails before replacing any existing output unless
the exact target has a reviewed `release_thresholds` entry. Release staging
recomputes the payload measurement and rejects missing, mismatched, exceeded, or
development-only thresholds.
The portable builder uses `pip` only to install and validate dependencies, then
uninstalls it and verifies the package manager is absent before inventorying and
hashing the runtime. It also removes build-only console/activation scripts,
dependency tests/caches/bytecode, and venv command/executable provenance. All
post-sanitization Python probes disable bytecode writes.
Only an exact, fully SHA-256-hashed lock plus an offline reviewed wheelhouse can
produce a manifest with `build_mode=portable`. Explicit unlocked network builds
are marked `portable-unlocked-development` and are rejected by staging.
Windows and Linux unlocked CI characterization builds additionally pass
`--development-cpu-torch`, which installs PyTorch from the official CPU-only
wheel channel and constrains the remaining open-ended requirements to that
selected wheel. Because that channel can supply a stale transitive `setuptools`
wheel, the subsequent PyPI resolution explicitly requires
`setuptools>=78.1.1,<82`: the lower bound excludes CVE-2025-47273 while the upper
bound preserves PyTorch's declared compatibility contract. The option and this
dynamic characterization-only resolution are rejected for reviewed release
builds, and the channel is recorded in runtime provenance. This keeps
characterization aligned with the v1 Windows/Linux CPU support promise without
weakening the requirement for reviewed target locks and wheelhouses.
`stage_runtime.py` rejects linked-development runtimes, release symlinks, target
mismatches, incompatible manifest/API/data schema contracts, unsafe manifest
paths, missing files, and checksum mismatches. Runtime smoke also verifies that
the backend readiness contract reports the same API, data, runtime, and manifest
schema versions recorded in the built runtime manifest.

The current macOS arm64 unlocked characterization build proves this sanitizer
and the full smoke/audit flow, but it is not releasable: the extracted payload
audit still identifies 80 upstream wheel/example home-path strings. A reviewed
locked wheelhouse must resolve or explicitly disposition those findings before
staging; broad path or vulnerability ignores are prohibited.

## CI and Promotion

`.github/workflows/llm-studio-desktop.yml` runs target-native macOS arm64,
Windows x64, and Linux x64 quality/runtime smoke checks. Default RunPod checks
must use fakes and must never create billable resources. It also compiles the
target-native release shell without bundling an unsafe characterization runtime.
The Windows shell owns its kill-on-close Job Object through Rust's
`OwnedHandle`, so the handle is safe in Tauri's shared supervisor state and
closes on every drop path, including unexpected backend removal.
Runtime manifest paths use normalized portable forward-slash relative syntax;
rooted, drive-relative, UNC, backslash, dot, parent, empty, and repeated
separator forms fail closed consistently on every host.
Each target-native characterization runtime is scanned with pip-audit after its
build-only package manager is removed, and the Rust lock is scanned with
Cargo Audit. The target-native release shell is built with pinned
`cargo-auditable` metadata and scanned as a completed binary. CI uploads each
target's non-browser performance-characterization report as retained evidence;
the reports do not establish release thresholds without review.
The deterministic tokenizer/training/inference smoke workload explicitly runs
on CPU so transient accelerator availability cannot invalidate the baseline
workflow. Readiness and retained characterization reports still record CPU,
MPS, and CUDA capabilities; accelerator workload qualification remains a
separate target-specific release gate.

Channels promote in order: `alpha` -> `beta` -> `stable`. Promotion requires:

1. Matching source revision, lockfiles, web output, runtime, SBOM, and provenance.
2. Target-native runtime smoke and installer clean-machine smoke.
3. No critical/high security findings.
4. Signed artifacts and published SHA-256 checksums.
5. macOS hardened runtime/notarization and Windows signing/timestamping.
6. Upgrade, rollback, uninstall, and user-data retention evidence.
7. Architecture, security, release, QA, documentation, and product approvals.

The supported upgrade path is previous stable to current stable. Newer data
schemas fail closed on downgrade. Until a first stable release and predecessor
exist, upgrade/rollback evidence remains an external release gate.

Data schema 3 preserves schema-2 typed data-root-relative SQLite locations. Its
upgrade first backs up both SQLite databases, transactionally rewrites only
managed paths proven to be under the configured data root, then strips legacy
Hugging Face dataset tokens and redacts provider credentials from active,
retained legacy, and migration-backup databases before checkpointing/removing
WAL/SHM sidecars and physically compacting them. It also sanitizes managed job
inputs/logs and removes transient remote bundles that could retain historical
embedded credentials. External paths and
RunPod remote workspace paths are preserved. Focused tests cover relocation,
unsafe typed-location rejection, rollback, backup isolation, physical secret
and sidecar removal, and idempotent re-run; released-predecessor restore/upgrade
evidence remains a release gate.

Runtime credential regressions also use arbitrary non-prefixed token values.
They prove job-scoped process-memory redaction for active errors/logs and
terminal scrubbing of managed text outputs before artifact ZIP creation.

Before schema/store initialization, an independent one-time migration copies
the historical `tokenizer_studio.db` and `training_studio.db` defaults into
`llm_studio_tokenizer.db` and `llm_studio_training.db` through SQLite's backup
API, verifies integrity, atomically promotes the copy, and retains the old
sources for rollback evidence. Existing current databases and custom paths are
never overwritten. Frontend startup similarly migrates historical Tokenizer
Studio form/job/theme localStorage keys into active LLM Studio keys using
copy-before-remove semantics.

After installers are built and signed, `release_manifest.py` generates a
path-redacted `release-manifest.json` plus `SHA256SUMS`. It rejects symlinks,
duplicate artifact names, generated-name collisions, and output outside
`build/desktop`. The manifest records artifact size/hash, source revision/tree
state, shell/web versions, target, and the external signing/notarization/
clean-machine gates. Signing systems must sign the final artifacts and publish
their signatures separately.

## Signing Gates

Signing and notarization are intentionally not simulated in source or normal CI.
Release jobs must use protected environments and short-lived credentials:

- macOS: Developer ID Application signing, hardened runtime, entitlements,
  notarization, and stapling.
- Windows: Authenticode signing and trusted timestamping.
- Linux: package/repository signatures for selected package targets.

The signed Tauri updater remains disabled until signed update artifacts and
rollback are proven on clean machines.

## Release Artifact Audit

Before publication, scan installers and extracted payloads for:

- credentials, API/runtime tokens, and private keys;
- developer home/repository paths;
- caches, test output, datasets, and local artifacts;
- unexpected binaries or licenses;
- mismatched manifest hashes or target metadata.

`audit_release_payload.py` performs the automated extracted-payload portion. It
fails on secrets, private keys, developer home paths, test/cache/dependency
trees, bytecode, symlinks, unsafe runtime-manifest paths, missing runtime files,
or runtime hash drift. Its secret patterns distinguish source-code identifier
usage from concrete assigned tokens/keys and reject localStorage/sessionStorage
writes to secret-bearing keys. Console output is bounded while a requested JSON
report retains every finding. Installer-format-specific extraction and
signature verification remain target-native release steps.

Uninstall tests must prove binaries are removed and user data is preserved by
default.
