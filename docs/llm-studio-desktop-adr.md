# ADR: LLM Studio Desktop Product and Platform Contract

- Date: 2026-06-12
- Status: Implemented baseline; product/security/release approvals pending
- Decision owners: desktop shell, web, API/runtime, release engineering

## Product Identity

| Item | Decision |
| --- | --- |
| Product name | LLM Studio |
| Bundle identifier | `com.llmbuilder.studio` |
| Rust executable/package | `llm_studio_desktop` |
| App-data directory name | `LLMStudio` |
| Main window label | `main` |
| Process instances | Single instance; a second launch focuses the existing main window |
| Deep-link scheme | Deferred until a reviewed external URL/deep-link design exists |
| Native notifications | Deferred; long-job status remains in-app for v1 |

## Release Channel

The first supported release is a full/offline installer containing the static web
bundle, Tauri shell, and target-native Python/ML runtime. A thin installer or
downloaded runtime is deferred until signed runtime manifests, atomic staging,
rollback, proxy support, and update-recovery tests are complete.

## Platform Matrix

| Target | Initial status | Minimum | Local compute |
| --- | --- | --- | --- |
| macOS arm64 | Required GA target | macOS 13 | CPU and PyTorch MPS where reported available |
| Windows x64 | Required GA target | Windows 10 22H2 with WebView2 | CPU; CUDA intentionally excluded from v1 promise |
| Linux x64 | Beta | Ubuntu 22.04 / glibc 2.35 baseline | CPU |

Runtimes must be built and smoked on the target OS/architecture. Cross-built ML
runtimes do not satisfy release gates.

## Runtime Baselines

- Python: 3.12 target-native runtime.
- Rust: 1.88 repository baseline.
- API compatibility: the sanitized FastAPI OpenAPI request/response catalog is
  frozen in `docs/llm-studio-desktop-api-contract-fixtures.json`; unreviewed
  operation or schema drift fails local verification and CI.
- Runtime size: checked-in development guardrails are enforced during
  characterization. Every release target requires a separately reviewed size
  threshold, and release staging independently recomputes it.
- Node: 24 for builds only; Node is never shipped or required at runtime.
- PyTorch: target-native package resolved by the reviewed desktop lock/wheel
  manifest. The current development environment is characterization evidence,
  not a release lock.
- RunPod workflows remain network-dependent and use fake providers in default CI.

## Data and Uninstall

Mutable projects, databases, uploads, artifacts, logs, and caches live under
OS-managed application data/cache directories. Install, update, and uninstall
must not write mutable data under signed resources. Uninstall preserves user
data by default and documentation explains manual removal.

## Consequences

- The stable Tauri asset origin preserves current localStorage behavior.
- The backend is a loopback-only authenticated sidecar, never the main UI origin.
- Duplicate launches focus the existing app and do not forward unreviewed arguments.
- Full installers are large because they include target-native ML dependencies.
- Signing, notarization, clean-machine validation, and approvals remain mandatory
  release gates outside source implementation.
