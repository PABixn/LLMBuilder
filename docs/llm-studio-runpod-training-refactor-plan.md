# LLM Studio RunPod Training Refactor Plan

## Purpose

Refactor the `llm-studio` external training module so RunPod training remains working while the codebase becomes easier to understand, test, debug, and extend.

This is an execution plan, not a design sketch. The executor must move in small behavior-preserving steps, keep tests green after each phase, and update every completed checklist item.

## Completion Protocol

- Every executable step starts with `[ ]`.
- When a step is finished, replace `[ ]` with `[*]`.
- Do not mark a parent step complete until every child step under it is complete.
- If a step is intentionally skipped, leave it unchecked and add a short `Skipped:` note with the reason.
- If a step discovers a bug or missing prerequisite, leave it unchecked and add a short `Blocked:` note with the concrete blocker.
- Keep this file current during the refactor. It is the handoff artifact for future executors.

## Scope

In scope:

- Backend training API under `apps/llm-studio/api/app`.
- RunPod executor code under `apps/llm-studio/api/app/training_executors`.
- Training persistence under `apps/llm-studio/api/app/training_storage.py`.
- Remote pod agent under `apps/llm-studio/remote_agent`.
- Training web page under `apps/llm-studio/web/app/training`.
- Training web client under `apps/llm-studio/web/lib/trainingApi.ts`.
- Training styles under `apps/llm-studio/web/app/styles/training.css`.
- RunPod training docs under `docs/`.
- Existing regression tests under `apps/llm-studio/api/tests` and `apps/llm-studio/web/app/**/*.test.ts`.

Out of scope unless a step explicitly adds it:

- Replacing RunPod Pods with Serverless.
- Requiring RunPod S3 keys for the normal path.
- Rewriting the actual model trainer in `training/runner.py`.
- Major visual redesign unrelated to RunPod training stability.
- Real paid RunPod integration tests in default CI.

## Current State Snapshot

Observed current hotspots:

- `apps/llm-studio/api/app/training_jobs.py` is about 1,344 lines and mixes preflight, asset loading, job creation, polling, response mapping, artifact reads, log reads, local path validation, and executor coordination.
- `apps/llm-studio/api/app/training_recommendations.py` is about 1,196 lines and contains recommendation domain logic plus many formatting helpers.
- `apps/llm-studio/api/app/training_executors/runpod_pod.py` is about 859 lines and mixes RunPod provisioning, agent health checks, bundle upload, refresh, output sync, cleanup, URL resolution, and lifecycle logging.
- `apps/llm-studio/api/app/training_executors/remote_sync.py` mixes pod-agent HTTP transport, bundle building, local dataset file rewriting, checksum helpers, and path sanitization.
- `apps/llm-studio/api/app/main.py` still owns many `/api/v1/training/*` route definitions directly.
- `apps/llm-studio/web/app/training/components/TrainingPageContent.tsx` is about 3,462 lines and mixes page state, polling, RunPod settings, asset picker, dataset editor, preflight UI, active-run monitor, recent runs, forms, recommendations, and dialogs.
- `apps/llm-studio/web/lib/trainingApi.ts` is about 643 lines and mixes generated-like API types, HTTP helpers, job methods, RunPod provider methods, and inference streaming helpers.
- `apps/llm-studio/web/app/styles/training.css` is about 2,773 lines and is a single style island for many unrelated page regions.
- Existing tests already cover many RunPod regressions, including old SQLite migration, agent URL resolution, Cloudflare 1010 handling, stale image messaging, terminal cleanup, optional file sync, remote-agent auth, diagnostics, and local dataset bundle rewrite.

Known fragile or incomplete areas that the refactor must protect or clarify:

- UI-pasted RunPod API keys are intentionally in-memory only. Do not persist them.
- Pod-agent bearer tokens are intentionally raw-in-memory only. The DB stores only a token hash.
- API restart cannot currently reauthenticate to a running pod-agent because the raw token is gone.
- `list_incomplete_runpod_jobs()` exists but restart recovery is not fully implemented.
- RunPod cleanup currently acts on Pods; network volume cleanup is exposed in UI/contracts but not clearly backed by a created network volume ID.
- `runpod_network_volume_id` and `runpod_cost_per_hr` are persisted but not reliably populated.
- Remote output sync currently downloads append-only logs/metrics/samples and selected JSON files. Full checkpoint/artifact file sync needs a clearer contract and tests.
- `_verify_agent_compatibility()` exists and is tested, but the launch flow must explicitly decide when it is called and how legacy images continue.
- RunPod direct TCP vs HTTP proxy behavior is a high-risk path because it has already caused Cloudflare and port-mapping bugs.
- Training jobs are refreshed from both local runtime files and executor snapshots; terminal states must never regress back to provisioning/running.

## Target Shape

### Backend Target File Map

The final backend should use a package that avoids name collisions with the root `training` package. Use `training_runs` for the API-side application module.

```text
apps/llm-studio/api/app/training_runs/
  __init__.py
  routes.py
  manager.py
  schemas.py
  store.py
  migrations.py
  responses.py
  artifacts.py
  runtime_files.py
  identifiers.py
  preflight/
    __init__.py
    service.py
    assets.py
    config_validation.py
    compatibility.py
    local_files.py
    scheduler_fixes.py
    runtime_summary.py
    recommendations.py
  executors/
    __init__.py
    base.py
    local.py
    runpod/
      __init__.py
      executor.py
      client.py
      config.py
      state.py
      lifecycle_log.py
      cleanup.py
      ports.py
      agent_client.py
      bundle.py
      dataset_files.py
      sync.py
      errors.py
      tokens.py
```

Compatibility shims should remain temporarily:

```text
apps/llm-studio/api/app/training_jobs.py
apps/llm-studio/api/app/training_models.py
apps/llm-studio/api/app/training_storage.py
apps/llm-studio/api/app/training_recommendations.py
apps/llm-studio/api/app/training_executors/*.py
```

Each shim should import and re-export the new implementation until all imports are migrated. Delete shims only in the final cleanup phase.

### Remote Agent Target File Map

```text
apps/llm-studio/remote_agent/
  app.py
  routes.py
  auth.py
  paths.py
  bundle.py
  files.py
  runner.py
  diagnostics.py
  sync_manifest.py
  schemas.py
```

`app.py` should become a small FastAPI app factory/bootstrap file. Route handlers should move to `routes.py`, with file range serving in `files.py` and workspace/job path helpers in `paths.py`.

### Frontend Target File Map

```text
apps/llm-studio/web/lib/training/
  client.ts
  errors.ts
  types.ts
  jobs.ts
  providers.ts
  artifacts.ts
  generation.ts

apps/llm-studio/web/app/training/
  page.tsx
  types.ts
  constants.ts
  hooks/
    useTrainingWorkspace.ts
    useTrainingSelection.ts
    useTrainingPreflight.ts
    useTrainingPolling.ts
    useRunPodSettings.ts
    useDatasetSettings.ts
    usePromptSettings.ts
    useAssetPicker.ts
    useTrainingToasts.ts
  components/
    TrainingPageContent.tsx
    TrainingHeroSection.tsx
    TrainingWorkflowSection.tsx
    ActiveRunPanel.tsx
    ActiveRunSummaryCards.tsx
    RunPodLifecyclePanel.tsx
    MetricsPanel.tsx
    SamplesPanel.tsx
    CheckpointsPanel.tsx
    LogsPanel.tsx
    RunDetailsPanel.tsx
    RecentRunsPanel.tsx
    PreflightPanel.tsx
    BatchLrAdvisor.tsx
    ExecutionTargetPanel.tsx
    RunPodSettingsPanel.tsx
    TrainingPlanPanel.tsx
    DatasetSettingsPanel.tsx
    StreamingDatasetEditor.tsx
    LocalFilesDatasetEditor.tsx
    SamplingPromptsPanel.tsx
    AdvancedRuntimePanel.tsx
    GeneratedConfigPanel.tsx
    AssetPickerDialog.tsx
```

Target style split:

```text
apps/llm-studio/web/app/styles/training.css
apps/llm-studio/web/app/styles/training/layout.css
apps/llm-studio/web/app/styles/training/hero.css
apps/llm-studio/web/app/styles/training/workflow.css
apps/llm-studio/web/app/styles/training/active-run.css
apps/llm-studio/web/app/styles/training/runpod.css
apps/llm-studio/web/app/styles/training/preflight.css
apps/llm-studio/web/app/styles/training/settings.css
apps/llm-studio/web/app/styles/training/dataset.css
apps/llm-studio/web/app/styles/training/prompts.css
apps/llm-studio/web/app/styles/training/charts.css
apps/llm-studio/web/app/styles/training/asset-picker.css
apps/llm-studio/web/app/styles/training/responsive.css
```

`training.css` should become an import-only aggregator after styles are split.

## Non-Negotiable Refactor Rules

- [ ] Do not change the RunPod launch behavior and refactor structure in the same step unless a characterization test covers the behavior.
- [ ] Do not persist raw RunPod API keys or raw pod-agent tokens.
- [ ] Do not delete or rewrite existing user worktree changes unless the user explicitly asks.
- [ ] Do not remove current troubleshooting docs until equivalent updated docs exist.
- [ ] Keep old import paths working until all direct imports are migrated.
- [ ] Keep local training behavior working throughout the refactor.
- [ ] Keep existing API response shapes stable unless the plan explicitly calls for a versioned or backward-compatible addition.
- [ ] Prefer fake RunPod clients and fake pod-agent servers in tests. Do not require paid RunPod resources in normal tests.
- [ ] Keep every phase small enough that failures can be isolated to one boundary.

## Phase 0 - Baseline And Safety Harness

Goal: capture the current behavior before moving code.

- [*] Record the current dirty worktree with `git status --short` before making refactor edits.

  Baseline `git status --short` on 2026-04-29 before refactor edits:

  ```text
   M apps/llm-studio/api/app/training_recommendations.py
   M apps/llm-studio/api/tests/test_runpod_training.py
   M apps/llm-studio/remote_agent/app.py
   M apps/llm-studio/remote_agent/diagnostics.py
   M docker/training/Dockerfile
   M docs/runpod-training-troubleshooting.md
   D llm_builder/__init__.py
   D llm_builder/local_text_data.py
   M tokenizer/dataloader.py
   M training/dataloader.py
  ?? docs/llm-studio-runpod-training-refactor-plan.md
  ?? training/local_text_data.py
  ```

- [*] Identify user-owned modified files and avoid overwriting them.

  Treating the dirty files listed above as pre-existing user-owned work. Refactor edits must preserve their current intent and avoid reverting them.

- [*] Run or document the current backend regression baseline:
  - [*] `cd apps/llm-studio/api && pytest tests/test_runpod_training.py tests/test_training_regressions.py`
  - [*] If the command fails for environment reasons, record the exact failure in this file.

  Baseline result:

  - Exact plan command failed because `pytest` was not installed on the active `PATH`: `zsh:1: command not found: pytest`.
  - `python -m pytest tests/test_runpod_training.py tests/test_training_regressions.py` also failed in the active environment: `/Users/pabi/Documents/GitHub/nanochat/.venv/bin/python: No module named pytest`.
  - Created isolated test env at `/tmp/llmbuilder-api-venv` and installed `apps/llm-studio/api/requirements-dev.txt`.
  - Running the same tests from `apps/llm-studio/api` with only that env reached collection but failed with `ModuleNotFoundError: No module named 'model'` because `pytest.ini` only adds `.` to `pythonpath`.
  - Verified current behavior with explicit repo-root path:

    ```text
    PYTHONPATH=/Users/pabi/Documents/GitHub/LLMBuilder:/Users/pabi/Documents/GitHub/LLMBuilder/apps/llm-studio/api /tmp/llmbuilder-api-venv/bin/python -m pytest tests/test_runpod_training.py tests/test_training_regressions.py
    52 passed in 2.14s
    ```

- [*] Run or document the current frontend baseline:
  - [*] `cd apps/llm-studio/web && npm run typecheck`
  - [*] `cd apps/llm-studio/web && npm run lint`
  - [*] `cd apps/llm-studio/web && npm run test:regression`
  - [*] `cd apps/llm-studio/web && npm run build`
  - [*] If any command fails for pre-existing reasons, record the exact failure in this file.

  Baseline result: all four frontend commands passed.

- [*] Add a short baseline note listing current file sizes for:
  - [*] `training_jobs.py`
  - [*] `training_recommendations.py`
  - [*] `runpod_pod.py`
  - [*] `remote_sync.py`
  - [*] `TrainingPageContent.tsx`
  - [*] `trainingApi.ts`
  - [*] `training.css`

  Baseline line counts:

  ```text
  1344 apps/llm-studio/api/app/training_jobs.py
  1196 apps/llm-studio/api/app/training_recommendations.py
   859 apps/llm-studio/api/app/training_executors/runpod_pod.py
   399 apps/llm-studio/api/app/training_executors/remote_sync.py
  3462 apps/llm-studio/web/app/training/components/TrainingPageContent.tsx
   643 apps/llm-studio/web/lib/trainingApi.ts
  2773 apps/llm-studio/web/app/styles/training.css
  ```

- [*] Confirm the current public API routes under `/api/v1/training/*` before moving route handlers.

  Confirmed current training routes:

  ```text
  GET    /api/v1/training/health
  GET    /api/v1/training/config/templates
  GET    /api/v1/training/config/schemas
  POST   /api/v1/training/validate/dataloader
  POST   /api/v1/training/validate/training-config
  POST   /api/v1/training/validate/preflight
  GET    /api/v1/training/providers/runpod/defaults
  GET    /api/v1/training/providers/runpod/status
  POST   /api/v1/training/providers/runpod/validate-key
  GET    /api/v1/training/providers/runpod/pods
  GET    /api/v1/training/providers/runpod/network-volumes
  POST   /api/v1/training/jobs
  GET    /api/v1/training/jobs
  GET    /api/v1/training/jobs/{job_id}
  DELETE /api/v1/training/jobs/{job_id}
  GET    /api/v1/training/jobs/{job_id}/metrics
  GET    /api/v1/training/jobs/{job_id}/samples
  GET    /api/v1/training/jobs/{job_id}/logs
  GET    /api/v1/training/jobs/{job_id}/data-preview
  GET    /api/v1/training/jobs/{job_id}/checkpoints
  POST   /api/v1/training/jobs/{job_id}/generate
  POST   /api/v1/training/jobs/{job_id}/generate/stream
  POST   /api/v1/training/jobs/{job_id}/stop
  POST   /api/v1/training/jobs/{job_id}/remote/resync
  POST   /api/v1/training/jobs/{job_id}/remote/cleanup
  POST   /api/v1/training/jobs/{job_id}/remote/reattach
  GET    /api/v1/training/jobs/{job_id}/artifact
  ```

- [*] Confirm no real RunPod API key is required for default regression tests.

  Confirmed by running the backend regression tests with fake clients and no real RunPod key configured.

Exit criteria:

- [*] Existing behavior has a written baseline.
- [*] Known failing commands, if any, are documented before refactor work starts.
- [*] No production code has been moved yet.

## Phase 1 - Add Characterization Tests Before Moving Code

Goal: lock down current behavior around fragile RunPod paths.

### Backend API And Manager Tests

- [*] Add tests for creating a local training job using a fake local executor so route/manager behavior is covered without running the trainer.
- [*] Add tests for creating a RunPod job using a fake RunPod executor and verifying:
  - [*] Job row is created before remote submission.
  - [*] Initial state becomes `running` + `preflight` + `executor_status=provisioning`.
  - [*] Remote submission failures mark the job `failed` and preserve the local job directory for diagnostics.
  - [*] `remote_error` is populated on launch failure.
- [*] Add tests for `_state_updates_from_runtime()` terminal precedence:
  - [*] A completed runtime file sets completed state.
  - [*] A failed runtime file sets failed state and error.
  - [*] A terminal DB row is not reverted by a stale runtime file or executor snapshot.
- [*] Add tests for `get_logs()` on RunPod jobs:
  - [*] Lifecycle/startup/agent/runner logs are prepended to stdout.
  - [*] `lines` limiting happens after merged logs.
- [*] Add tests for `delete_job()` preserving the "stop before deleting running job" rule.
- [*] Add tests for `cleanup_remote_job()` and `reattach_remote_job()` route behavior.

### RunPod Executor Tests

- [*] Add tests that `RunPodPodExecutor.submit()` calls compatibility probing at the intended point, or explicitly test that it does not and document why.
- [*] Add tests for pod creation request defaults:
  - [*] `imageName`
  - [*] `gpuTypeIds`
  - [*] `gpuCount`
  - [*] `cloudType`
  - [*] `ports` with `8021/tcp` by default
  - [*] required remote-agent env vars
  - [*] Hugging Face cache env vars
- [ ] Add tests for API key handling:
  - [*] UI-provided key is used for launch.
  - [ ] Env key is used when UI key is absent.
  - [ ] Raw key is not placed in job metadata, lifecycle logs, or remote env.
    Partial: lifecycle logs and remote env are covered.
- [ ] Add tests for pod-agent token handling:
  - [ ] Raw token is stored only in the executor process memory.
  - [ ] Hash is persisted.
  - [ ] Refresh after missing token reports a clear `remote_error`.
- [ ] Add tests for cleanup policy behavior:
  - [ ] Completed + delete policy deletes pod.
  - [ ] Failed + delete policy stops pod for inspection.
  - [ ] Keep policy does nothing.
  - [ ] Cleanup failure reports `remote_error` but does not alter terminal job status.
- [ ] Add tests for port resolution with representative RunPod payloads:
  - [ ] Direct TCP mapping dict.
  - [ ] Direct TCP mapping list under `runtime.ports`.
  - [ ] HTTP proxy mapping.
  - [ ] Explicit URL mapping.
  - [ ] Private IP plus public IP fallback.
- [ ] Add tests for retry policy:
  - [*] Capacity errors retry.
  - [*] Auth errors do not retry.
  - [ ] Cloudflare 1010 does not retry.

### Remote Sync Tests

- [ ] Add tests for bundle manifest contents:
  - [ ] All input files are listed with size and SHA-256.
  - [ ] `resolved_preflight.json` includes `remote_local_files`.
  - [ ] Local dataset files are copied exactly once.
  - [ ] Globbed local dataset files produce stable remote paths.
  - [ ] Remote URLs in `data_files` are preserved.
- [ ] Add tests for path resolution semantics and document whether paths are repo-root-relative, job-dir-relative, or absolute.
- [ ] Add tests for append-only output sync idempotency:
  - [ ] First sync writes full file.
  - [ ] Second sync only appends new bytes.
  - [ ] Missing optional files do not mark the run failed.
- [ ] Add tests for final artifact manifest verification behavior before adding full artifact sync.

### Remote Agent Tests

- [*] Add tests for `safe_join()` blocking path traversal for file downloads.
- [*] Add tests for bundle extraction rejecting unsafe tar members.
- [*] Add tests for `/v1/system` response shape and runner compatibility payload.
- [*] Add tests for range serving headers:
  - [*] `X-File-Size`
  - [*] `X-Start-Offset`
  - [*] empty response when offset is at EOF.
- [*] Add tests for cancel behavior when no process exists.
- [ ] Add tests for cancel behavior sending SIGTERM then SIGKILL after timeout using fakes.

### Frontend Tests

- [*] Add unit tests for `shouldPollTrainingRun()` and terminal status handling if not already complete.
- [ ] Add tests for RunPod launch payload construction after extraction to a helper.
- [ ] Add tests for RunPod readiness:
  - [ ] Local execution does not require RunPod key.
  - [ ] RunPod execution requires env-configured key or typed key.
  - [ ] Validate-key state updates provider status.
- [ ] Add tests for active-run polling behavior after extraction:
  - [ ] 404 clears active run.
  - [ ] terminal jobs stop polling.
  - [ ] running jobs continue polling.
- [ ] Add tests for dataset UI config roundtrip:
  - [ ] local files to dataloader config
  - [ ] streaming datasets to dataloader config
  - [ ] hydrated config back to UI state

Exit criteria:

- [ ] New tests fail only for missing new behavior intentionally scheduled later, or pass against current behavior.
- [ ] Existing regression tests still pass or documented pre-existing failures are unchanged.

## Phase 2 - Extract Training Routes From `main.py`

Goal: make API routing readable without changing route URLs or response shapes.

- [*] Create `apps/llm-studio/api/app/training_runs/routes.py`.
- [*] Move all `/api/v1/training/*` route handlers from `main.py` into `training_runs/routes.py`.
- [*] Keep the same `training_api = APIRouter(prefix="/api/v1/training", tags=["training-workspace"])` behavior.
- [*] Add a `register_training_routes(app: FastAPI) -> APIRouter` or equivalent function that wires access to `app.state.training_jobs` and `app.state.runpod_api_key_override`.
- [*] Keep provider helpers such as RunPod defaults/API-key lookup inside the new route module or a small `providers.py`, not in `main.py`.
- [*] Update `main.py` so it only imports and includes the training router.
- [*] Verify route URLs remain unchanged:
  - [*] `/api/v1/training/health`
  - [*] `/api/v1/training/config/templates`
  - [*] `/api/v1/training/config/schemas`
  - [*] `/api/v1/training/validate/dataloader`
  - [*] `/api/v1/training/validate/training-config`
  - [*] `/api/v1/training/validate/preflight`
  - [*] `/api/v1/training/providers/runpod/*`
  - [*] `/api/v1/training/jobs*`
- [*] Add route-level tests that import the app and assert the above routes still exist.
- [*] Run backend tests.

Exit criteria:

- [*] `main.py` no longer contains training route bodies.
- [*] No frontend API path changes are required.

## Phase 3 - Introduce `training_runs` Package With Compatibility Shims

Goal: create the new module layout while keeping old imports working.

- [*] Create `apps/llm-studio/api/app/training_runs/__init__.py`.
- [*] Move `training_models.py` into `training_runs/schemas.py`.
- [*] Replace `training_models.py` with a compatibility shim that re-exports from `training_runs.schemas`.
- [*] Move `training_storage.py` into `training_runs/store.py`.
- [*] Replace `training_storage.py` with a compatibility shim that re-exports from `training_runs.store`.
- [*] Move schema migration logic out of the store and into `training_runs/migrations.py`.
- [*] Keep `TrainingStudioStore.initialize()` behavior identical after extraction.
- [*] Move identifier regexes and identifier validation to `training_runs/identifiers.py`.
- [*] Move runtime file helpers to `training_runs/runtime_files.py`:
  - [*] `read_jsonl`
  - [*] `tail_lines`
  - [*] `directory_size`
  - [*] `load_optional_json`
  - [*] datetime parsing helpers
- [*] Move artifact bundle creation to `training_runs/artifacts.py`.
- [ ] Update imports in tests and production code gradually, preferring new imports in new code.
- [*] Run backend tests.

Exit criteria:

- [*] Old imports still work.
- [*] New code can import from `app.training_runs.*`.
- [ ] No behavior changes.

## Phase 4 - Split Preflight From `TrainingRunManager`

Goal: make preflight independently testable and remove domain validation from the job manager.

- [*] Create `training_runs/preflight/service.py` with a `TrainingPreflightService`.
- [*] Move `ResolvedPreflightContext` to the preflight package.
- [*] Move project asset loading to `training_runs/preflight/assets.py`.
- [*] Move tokenizer asset loading to `training_runs/preflight/assets.py`.
- [*] Move Pydantic validation error conversion to `training_runs/preflight/config_validation.py`.
- [*] Move model/tokenizer compatibility checks to `training_runs/preflight/compatibility.py`.
- [*] Move local dataset file validation to `training_runs/preflight/local_files.py`.
- [*] Move scheduler suggested fixes to `training_runs/preflight/scheduler_fixes.py`.
- [*] Move runtime memory/batch summary building to `training_runs/preflight/runtime_summary.py`.
- [*] Move `build_batch_and_lr_recommendation()` and related helpers into `training_runs/preflight/recommendations.py`.
- [*] Preserve existing recommendation response fields and wording unless tests are updated intentionally.
- [*] Update `TrainingRunManager.build_preflight()` to delegate to `TrainingPreflightService`.
- [*] Update `TrainingRunManager.create_job()` to use the preflight service result.
- [*] Add focused tests for each extracted preflight module.
- [*] Run backend tests.

Exit criteria:

- [*] `TrainingRunManager` no longer imports model runtime objects except what it needs for response/job creation.
- [*] Preflight can be tested without constructing a full manager where practical.

## Phase 5 - Simplify `TrainingRunManager`

Goal: reduce the manager to orchestration only.

- [*] Move job directory creation and input file writing to `training_runs/artifacts.py`.
- [*] Create a `PreparedTrainingJob` dataclass containing:
  - [*] `StoredTrainingJob`
  - [*] `TrainingJobBundle`
  - [*] job directory paths
  - [*] preflight payload
- [*] Move `StoredTrainingJob` construction into a factory function.
- [*] Move `TrainingJobResponse` conversion to `training_runs/responses.py`.
- [*] Move active output readers to `training_runs/runtime_files.py` or `training_runs/artifacts.py`.
- [*] Move executor refresh throttling into a small helper object.
- [*] Keep the manager methods as the public API:
  - [*] `build_preflight`
  - [*] `create_job`
  - [*] `list_jobs`
  - [*] `get_job`
  - [*] `delete_job`
  - [*] `stop_job`
  - [*] `get_metrics`
  - [*] `get_samples`
  - [*] `get_logs`
  - [*] `get_data_preview`
  - [*] `get_checkpoints`
  - [*] `build_artifact_bundle`
  - [*] `resync_remote_job`
  - [*] `cleanup_remote_job`
  - [*] `reattach_remote_job`
- [*] Make `_submit_remote_job()` update only state fields and delegate all RunPod details to the executor.
- [*] Ensure every manager write goes through `TrainingStudioStore`.
- [*] Run backend tests after each meaningful extraction.

Exit criteria:

- [*] `manager.py` is primarily orchestration and contains little validation or filesystem parsing.
- [*] Existing manager public methods are preserved.

## Phase 6 - Rework Persistence And Restart Semantics

Goal: make persisted state explicit and honest.

- [*] Move SQLite migration statements to `training_runs/migrations.py`.
- [*] Add a schema version table or documented migration registry for future migrations.
- [*] Add tests for migration idempotency.
- [*] Add tests for non-SQLite behavior if the code intentionally skips migrations for other databases.
- [*] Rename or document `mark_incomplete_jobs_failed()` and `mark_incomplete_local_jobs_failed()` so startup behavior is unambiguous.
- [*] Decide and implement startup behavior for incomplete RunPod jobs:
  - [*] Option A: leave them running with `remote_error` explaining token/API-key recovery limits.
  - [ ] Option B: mark them failed if raw token is absent.
  - [ ] Option C: support reattach with a user-provided fresh token.
- [*] Add a startup recovery function specifically for RunPod jobs.
- [*] Make `reattach_remote_job()` either truly reattach or rename/copy so it does not imply impossible recovery.
- [*] Audit nullable/default fields:
  - [*] `runpod_network_volume_id`
  - [*] `runpod_cost_per_hr`
  - [*] `runpod_agent_base_url`
  - [*] `runpod_cleanup_policy`
  - [*] `remote_workspace_path`
- [*] Add tests that old DB rows still hydrate with safe defaults.
- [*] Run backend tests.

Exit criteria:

- [*] API restart behavior for local and RunPod jobs is documented in code and tests.
- [*] Persisted RunPod fields match real behavior.

## Phase 7 - Split Executor Base And Local Executor

Goal: make executor contracts explicit before deeper RunPod work.

- [*] Move `training_executors/base.py` to `training_runs/executors/base.py`.
- [*] Move `training_executors/local.py` to `training_runs/executors/local.py`.
- [*] Keep old `training_executors` modules as compatibility shims.
- [*] Add docstrings to the executor protocol explaining:
  - [*] `submit()` ownership.
  - [*] `refresh()` idempotency.
  - [*] `stop()` expected terminal snapshot.
  - [*] `cleanup()` resource side effects.
- [*] Add tests for local executor command construction without spawning a real long-running trainer.
- [*] Add tests for local executor process exit mapping:
  - [*] exit code `0` -> completed
  - [*] exit code `2` -> cancelled
  - [*] other exit -> failed
- [*] Ensure local executor still uses the same `python -m training.runner` command and job-scoped paths.
- [*] Run backend tests.

Exit criteria:

- [*] Executor boundary is stable and documented.
- [*] Local training behavior is unchanged.

## Phase 8 - Split RunPod Executor Into Focused Modules

Goal: remove the high-risk monolith in `runpod_pod.py`.

### RunPod Configuration And State

- [*] Create `training_runs/executors/runpod/config.py`.
- [*] Move target/default resolution into a `ResolvedRunPodTarget` dataclass.
- [*] Validate RunPod fields in one place:
  - [*] API key source
  - [*] GPU type
  - [*] GPU count
  - [*] cloud type
  - [*] datacenter
  - [*] volume size
  - [*] cleanup policy
  - [*] agent port/protocol
- [*] Create `training_runs/executors/runpod/state.py`.
- [*] Define explicit executor statuses:
  - [*] `queued`
  - [*] `provisioning`
  - [*] `booting`
  - [*] `checking_agent`
  - [*] `building_bundle`
  - [*] `uploading`
  - [*] `starting`
  - [*] `running`
  - [*] `syncing`
  - [*] `cleaning_up`
  - [*] `completed`
  - [*] `failed`
  - [*] `cancelled`
  - [*] `cleaned_up`
- [*] Add tests that state labels map to user-visible stages consistently.

### RunPod Client

- [*] Move `runpod_client.py` to `training_runs/executors/runpod/client.py`.
- [*] Keep payload shape tests.
- [*] Add response-shape tests for list/object extraction.
- [*] Add typed methods or tracked follow-up records for resource endpoints that UI exposes:
  - [*] GPU availability if supported.
  - [*] network volumes if truly used.
  - [*] cost fields if available.
  Follow-up record: GPU availability and cost fields are not currently exposed as live capability endpoints; network volumes are intentionally not created by this flow.
- [*] Ensure all RunPod HTTP errors include status and sanitized detail.

### Port Resolution

- [*] Move `build_agent_base_url()` and helpers to `training_runs/executors/runpod/ports.py`.
- [*] Keep all current port mapping tests.
- [ ] Add fixtures for actual RunPod payloads captured from logs if available.
- [ ] Document direct TCP as the default and HTTP proxy as opt-in.

### Lifecycle Logging

- [*] Move `log_lifecycle()` and sanitizers to `training_runs/executors/runpod/lifecycle_log.py`.
- [*] Ensure lifecycle logs never include:
  - [*] raw RunPod API keys
  - [*] raw pod-agent tokens
  - [*] Authorization headers
  - [*] HF tokens
- [*] Add tests for recursive redaction in nested dicts/lists.
- [ ] Add a small helper for structured event names so they do not drift.

### Token Registry

- [*] Create `training_runs/executors/runpod/tokens.py`.
- [*] Move in-memory pod-agent token and API-key storage into a small registry object.
- [*] Add tests for:
  - [*] insert
  - [*] lookup
  - [*] remove
  - [*] missing token error text
  - [*] token hash calculation
- [*] Keep raw token/API-key fields process-local only.

### Cleanup

- [*] Move cleanup policy helpers to `training_runs/executors/runpod/cleanup.py`.
- [*] Clarify network volume behavior:
  - [*] If no separate network volume is created, rename UI/API wording away from "network volume cleanup" or mark it unsupported.
  - [*] If a network volume is created, persist its ID and implement delete behavior.
- [*] Add tests for cleanup behavior with and without `runpod_network_volume_id`.
- [*] Add tests that cleanup failure never masks a completed training result.

### Executor Orchestration

- [*] Move the class to `training_runs/executors/runpod/executor.py`.
- [*] Keep `submit()` as a readable sequence of high-level calls:
  - [*] resolve target
  - [*] create pod
  - [*] wait for port
  - [*] wait for agent
  - [*] verify compatibility
  - [*] build bundle
  - [*] upload bundle
  - [*] start remote process
  - [*] return handle updates
- [*] Ensure each high-level call emits lifecycle logs.
- [*] Ensure submit failure cleanup remains stop-by-default for failed launches.
- [*] Run backend tests after each submodule extraction.

Exit criteria:

- [*] No single RunPod executor file owns unrelated lifecycle, HTTP, sync, path, and logging concerns.
- [*] RunPod launch behavior remains covered by tests.

## Phase 9 - Split Remote Agent Client, Bundle, Dataset, And Sync

Goal: make local-to-pod transfer and pod-to-local sync reliable and testable.

### Agent Client

- [*] Move `RemoteAgentClient` to `training_runs/executors/runpod/agent_client.py`.
- [*] Move `RemoteAgentError` and HTTP error formatting to `errors.py`.
- [*] Keep certifi SSL context behavior.
- [*] Keep browser-like user agent behavior for HTTP proxy compatibility.
- [*] Add request timeout constants in one place.
- [*] Add tests for:
  - [*] auth headers
  - [*] job headers
  - [*] unauthenticated health
  - [*] query `job_id` on `/v1/system`
  - [*] optional file behavior
  - [*] Cloudflare 1010 formatting and retry flag

### Bundle Building

- [*] Move `build_remote_bundle()` to `training_runs/executors/runpod/bundle.py`.
- [*] Move dataset file rewriting to `training_runs/executors/runpod/dataset_files.py`.
- [*] Define a `RemoteBundleManifest` shape or TypedDict.
- [*] Make bundle temp/staging paths explicit and easy to clean.
- [*] Verify bundle format stays `llm-studio-training-bundle-v1`.
- [*] Add a cleanup step for stale `.remote_bundle` directories after successful bundle creation if safe.
- [*] Add tests for `.tar.zst` and `.tar.gz` fallback if `zstandard` is unavailable.

### Output Sync

- [*] Move `sync_small_outputs()` to `training_runs/executors/runpod/sync.py`.
- [*] Rename it to reflect its real scope, for example `sync_incremental_outputs()`.
- [*] Split sync tasks:
  - [*] append-only logs
  - [*] append-only metrics
  - [*] append-only samples
  - [*] latest JSON files
  - [*] checkpoint listing
  - [*] checkpoint file download
  - [*] final artifact manifest verification
- [*] Implement full checkpoint sync or document why checkpoints remain remote-only.
- [*] If full checkpoint sync is implemented:
  - [*] download checkpoint files listed by remote `/checkpoints`
  - [*] write into local `artifact_dir/checkpoints/<step>/`
  - [*] avoid re-downloading unchanged files
  - [*] verify file sizes/checksums if available
  - [*] update `checkpoint_count` from local synced files
- [*] Verify `get_checkpoints()` reports remote RunPod checkpoints only after they are locally available, or add a separate remote-visible state.
- [*] Verify final `artifact_manifest.json` is downloaded and checked before deleting the pod when policy is `delete_after_sync`.
- [*] Add tests that delete-after-sync does not run until required final sync succeeds.

Exit criteria:

- [*] Remote sync behavior is explicit enough to trust cleanup.
- [*] The UI copy about synced artifacts matches real behavior.

## Phase 10 - Refactor Remote Agent Server

Goal: make the pod image agent small, auditable, and compatible.

- [*] Create `remote_agent/paths.py` for:
  - [*] `workspace_root()`
  - [*] `job_root()`
  - [*] `outputs_dir()`
  - [ ] log paths if appropriate
- [*] Create `remote_agent/files.py` for:
  - [*] ranged file response
  - [*] JSON reads
  - [*] optional file response behavior
- [*] Create `remote_agent/routes.py` for FastAPI route registration.
- [*] Keep `remote_agent/app.py` as a small app bootstrap that registers middleware and routes.
- [*] Add `remote_agent/schemas.py` for response payload TypedDicts or Pydantic models if useful.
  Not added: current route response payloads are small inline dictionaries and adding a schema module would be ceremony without a caller benefit.
- [*] Keep auth behavior unchanged.
- [*] Keep `/health` unauthenticated.
- [*] Keep `/v1/system` authenticated.
- [*] Keep legacy compatibility for `/v1/system?job_id=...` while accepting the header.
- [*] Add an explicit remote-agent protocol version to `/v1/system`, for example:
  - [*] `agent_protocol_version`
  - [*] `bundle_format_versions`
  - [*] `supports_optional_files`
  - [*] `supports_checkpoint_manifest`
- [*] Update local compatibility probing to use protocol fields when present and legacy fallback when absent.
- [*] Keep diagnostics logging behavior.
- [*] Run remote-agent tests.

Exit criteria:

- [*] Remote agent routes are easy to scan.
- [*] The local API can reason about agent compatibility without relying on vague image freshness.

## Phase 11 - Clarify RunPod Product Contract

Goal: align UI, backend, docs, and actual RunPod behavior.

- [*] Decide whether this module uses RunPod ephemeral pod volume only or creates a RunPod network volume resource.
- [*] If using only pod volume:
  - [*] Rename `network_volume_size_gb` in new UI labels to "Pod volume size" while keeping old API field backward compatible.
  - [*] Hide or remove "Volume cleanup" from the UI, or label it as not applicable.
  - [*] Update docs to avoid implying a separately managed network volume.
- [ ] If using network volume resources:
  - [ ] Implement `CreateNetworkVolumeRequest` usage.
  - [ ] Persist `runpod_network_volume_id`.
  - [ ] Attach the volume to the pod.
  - [ ] Delete or keep the volume according to cleanup policy.
  - [ ] Add tests for volume create/attach/delete failures.
- [*] Decide whether cost estimation is supported.
- [*] If cost is unsupported:
  - [*] Remove or hide `runpod_cost_per_hr` UI promises.
  - [*] Keep nullable field for future use.
- [ ] If cost is supported:
  - [ ] Populate `runpod_cost_per_hr` from a real RunPod response or capability endpoint.
  - [ ] Add tests for missing and present cost fields.
- [*] Update `docs/runpod-training-user-guide.md`.
- [*] Update `docs/runpod-training-security.md`.
- [*] Update `docs/runpod-training-troubleshooting.md`.

Exit criteria:

- [*] User-facing language matches actual resource behavior.
- [*] Cleanup policy does exactly what the UI says it does.

## Phase 12 - Split Frontend API Client

Goal: make web API contracts maintainable without changing consumers all at once.

- [*] Create `apps/llm-studio/web/lib/training/types.ts`.
- [*] Move all exported TypeScript interfaces/types from `trainingApi.ts` into `training/types.ts`.
- [*] Create `apps/llm-studio/web/lib/training/errors.ts` for `TrainingApiError`.
- [*] Create `apps/llm-studio/web/lib/training/client.ts` for base URL, runtime headers, `request()`, and error parsing.
- [*] Create `apps/llm-studio/web/lib/training/jobs.ts` for job/preflight/metrics/logs/checkpoints methods.
- [*] Create `apps/llm-studio/web/lib/training/providers.ts` for RunPod provider methods.
- [*] Create `apps/llm-studio/web/lib/training/artifacts.ts` for artifact download URLs.
- [*] Create `apps/llm-studio/web/lib/training/generation.ts` for checkpoint generation/streaming.
- [*] Replace `web/lib/trainingApi.ts` with a compatibility barrel that re-exports from the new files.
- [*] Update training page imports to use the new modules after the barrel is in place.
- [*] Run `npm run typecheck`.
- [*] Run `npm run lint`.
- [*] Run `npm run test:regression`.

Exit criteria:

- [*] `trainingApi.ts` is no longer a large mixed module.
- [*] Existing import paths still work until migrated.

## Phase 13 - Extract Frontend Hooks From `TrainingPageContent`

Goal: make page state testable and reduce the page component safely.

- [*] Create `useTrainingToasts()` and move toast state/timing.
- [*] Create `useTrainingSelection()` for selected project/tokenizer/run IDs and local storage.
- [*] Create `useRunPodSettings()` for:
  - [*] provider status load
  - [*] defaults hydration
  - [*] API key field state
  - [*] validation call
  - [*] launch target construction
  - [*] confirmation prompts
- [*] Extract launch payload construction to a pure helper and test it.
- [*] Create `useTrainingPreflight()` for debounced preflight validation.
- [*] Create `useTrainingPolling()` for active run, metrics, samples, logs, preview, checkpoints, and recent run updates.
- [*] Create `useDatasetSettings()` for local file and streaming dataset state.
- [*] Create `usePromptSettings()` for sample prompts.
- [*] Create `useAssetPicker()` for project/tokenizer picker state.
- [*] Keep `TrainingPageContent` rendering behavior unchanged after each hook extraction.
- [*] Run frontend tests after each major hook extraction.

Exit criteria:

- [*] `TrainingPageContent` no longer owns most effects directly.
- [*] Polling, RunPod settings, dataset state, and picker state each have focused tests or pure helper tests.

## Phase 14 - Extract Frontend Components

Goal: turn the training page into composable UI pieces.

- [*] Extract `ActiveRunPanel`.
- [*] Extract `ActiveRunSummaryCards`.
- [*] Extract `RunPodLifecyclePanel`.
- [*] Extract `MetricsPanel`.
- [*] Extract `SamplesPanel`.
- [*] Extract `CheckpointsPanel`.
- [*] Extract `LogsPanel`.
- [*] Extract `RunDetailsPanel`.
- [*] Extract `RecentRunsPanel`.
- [*] Extract `PreflightPanel`.
- [*] Extract `BatchLrAdvisor`.
- [*] Extract `ExecutionTargetPanel`.
- [*] Extract `RunPodSettingsPanel`.
- [*] Extract `TrainingPlanPanel`.
- [*] Extract `DatasetSettingsPanel`.
- [*] Extract `StreamingDatasetEditor`.
- [*] Extract `LocalFilesDatasetEditor`.
- [*] Extract `SamplingPromptsPanel`.
- [*] Extract `AdvancedRuntimePanel`.
- [*] Extract `GeneratedConfigPanel`.
- [*] Extract `AssetPickerDialog`.
- [*] Keep props narrow and domain-specific.
- [*] Avoid passing one giant controller object unless a hook returns a well-named view model.
- [*] Run frontend typecheck and lint after each batch.

Exit criteria:

- [*] `TrainingPageContent.tsx` is a high-level composition component.
- [*] No extracted component exceeds a reasonable size without a clear reason.

## Phase 15 - Improve RunPod Training UX Without Changing Backend Semantics

Goal: make the existing RunPod flow clearer and safer for users.

- [*] Show RunPod setup state in one focused panel:
  - [*] key source
  - [*] validation status
  - [*] selected GPU
  - [*] cloud type
  - [*] datacenter
  - [*] pod volume size
  - [*] port protocol
  - [*] cleanup policy
- [*] Make dangerous choices visually explicit:
  - [*] interruptible capacity can preempt runs
  - [*] keep pod can continue billing
  - [*] delete volume/cache removes remote cache if implemented
- [*] Add a visible "what happens next" lifecycle list for RunPod launches:
  - [*] create pod
  - [*] wait for port
  - [*] start pod-agent
  - [*] upload bundle
  - [*] start trainer
  - [*] sync outputs
  - [*] cleanup
- [*] On active RunPod jobs, show:
  - [*] pod ID with copy action if a copy utility exists locally
  - [*] agent URL only if safe and useful
  - [*] last heartbeat
  - [*] last sync
  - [*] remote error
  - [*] cleanup status
- [*] Add explicit recovery actions only if backend supports them:
  - [*] resync
  - [*] cleanup
  - [*] reattach or "reattach unavailable after restart"
- [*] Ensure button text matches the actual action.
- [*] Keep local training UI unaffected.

Exit criteria:

- [*] Users can tell whether a RunPod run is provisioning, training, syncing, cleaning up, or blocked.
- [*] UI no longer overpromises unsupported remote recovery behavior.

## Phase 16 - Split Training Styles

Goal: reduce CSS risk and make future UI changes local.

- [*] Create `apps/llm-studio/web/app/styles/training/`.
- [*] Split layout and grid styles into `layout.css`.
- [*] Split hero styles into `hero.css`.
- [*] Split workflow styles into `workflow.css`.
- [*] Split active-run and monitor styles into `active-run.css`.
- [*] Split RunPod-specific styles into `runpod.css`.
- [*] Split preflight styles into `preflight.css`.
- [*] Split settings form styles into `settings.css`.
- [*] Split dataset editor styles into `dataset.css`.
- [*] Split prompt editor styles into `prompts.css`.
- [*] Split chart styles into `charts.css`.
- [*] Split asset picker styles into `asset-picker.css`.
- [*] Split responsive overrides into `responsive.css`.
- [*] Make `training.css` import the split files in a deliberate order.
- [*] Remove duplicate selectors while preserving computed visual output.
- [*] Check mobile and desktop layouts after the split.

Exit criteria:

- [*] Styling for one training page section can be edited without scanning the whole stylesheet.
- [*] Visual behavior is unchanged except for intentional fixes.

## Phase 17 - Add Stronger Observability And Diagnostics

Goal: make future RunPod bugs faster to diagnose.

- [*] Standardize lifecycle event names and status transitions.
- [*] Include a correlation ID/job ID in every local lifecycle log line.
- [*] Include sanitized RunPod payload summaries for create-pod and port-mapping failures.
- [*] Include remote-agent protocol version in lifecycle logs when available.
- [*] Add a local `runpod_lifecycle.jsonl` parser helper if useful for tests or debugging.
- [*] Improve user-visible error categories:
  - [*] invalid API key
  - [*] no capacity
  - [*] pod created but no port
  - [*] agent unreachable
  - [*] stale image
  - [*] bundle upload rejected
  - [*] trainer import failure
  - [*] trainer runtime failure
  - [*] sync failure
  - [*] cleanup failure
- [*] Ensure troubleshooting docs map these categories to actions.

Exit criteria:

- [*] A user report with logs can be traced to a phase of the RunPod lifecycle quickly.

## Phase 18 - Documentation Update

Goal: keep operator docs aligned with the refactored architecture.

- [*] Update `docs/runpod-training-user-guide.md` with the final launch flow.
- [*] Update `docs/runpod-training-security.md` with final API-key/token handling.
- [*] Update `docs/runpod-training-troubleshooting.md` with final lifecycle states and recovery actions.
- [*] Add or update an architecture doc showing:
  - [*] web page
  - [*] local FastAPI training API
  - [*] local DB and job dir
  - [*] RunPod REST API
  - [*] pod-agent
  - [*] `training.runner`
  - [*] sync back to local artifacts
- [*] Add a developer note for how to run fake RunPod tests.
- [*] Add a developer note for how to safely do a real RunPod smoke test without leaving billable resources running.

Exit criteria:

- [*] Docs describe the code that exists after the refactor, not the earlier implementation plan.

## Phase 19 - Final Compatibility Cleanup

Goal: remove temporary shims only after the app has fully migrated.

- [*] Search for imports from old modules:
  - [*] `app.training_jobs`
  - [*] `app.training_models`
  - [*] `app.training_storage`
  - [*] `app.training_recommendations`
  - [*] `app.training_executors`
  - [*] `../../../lib/trainingApi`
- [*] Update remaining imports to new module paths.
- [*] Decide whether to keep public compatibility barrels for external callers.
- [*] If safe, delete old backend compatibility shims.
  Kept intentionally for compatibility and documented in `docs/runpod-training-architecture.md`.
- [*] If safe, delete old frontend `trainingApi.ts` barrel, or keep it intentionally documented.
  Kept intentionally for compatibility and documented in `docs/runpod-training-architecture.md`.
- [*] Remove stale tests that only verify old shim imports, if any.
- [*] Remove dead code found during extraction.
- [*] Remove obsolete docs that conflict with final docs.
- [*] Run full verification.

Exit criteria:

- [*] No dead compatibility layer remains unless intentionally documented.

## Phase 20 - Full Verification Checklist

Goal: prove the refactor is complete enough to hand off.

### Backend

- [*] `cd apps/llm-studio/api && pytest`
- [*] `cd apps/llm-studio/api && pytest tests/test_runpod_training.py`
- [*] `cd apps/llm-studio/api && pytest tests/test_training_regressions.py`
- [*] Start the API locally and verify:
  - [*] `/api/v1/training/health`
  - [*] `/api/v1/training/config/templates`
  - [*] `/api/v1/training/providers/runpod/status`
  - [*] `/api/v1/training/jobs`

### Frontend

- [*] `cd apps/llm-studio/web && npm run typecheck`
- [*] `cd apps/llm-studio/web && npm run lint`
- [*] `cd apps/llm-studio/web && npm run test:regression`
- [*] `cd apps/llm-studio/web && npm run build`
- [ ] Manual browser smoke test for `/training`:
  - [*] page loads
  - [ ] model picker opens
  - [ ] tokenizer picker opens
  - [ ] local/RunPod execution target toggle works
  - [ ] RunPod settings defaults load
  - [ ] preflight runs after selecting assets
  - [ ] recent runs render
  - [ ] active run monitor renders

### Fake RunPod End-To-End

- [*] Add or run a fake RunPod client/pod-agent integration test that covers:
  - [*] create job
  - [*] provision fake pod
  - [*] resolve fake port
  - [*] upload bundle
  - [*] start fake trainer
  - [*] sync metrics/logs/samples
  - [*] mark completed
  - [*] cleanup policy

### Real RunPod Smoke Test

Only run this when intentionally approved because it can create billable resources.

Not run in this pass; it requires an intentional real RunPod launch with billing risk.

- [ ] Use a low-risk training config with very small `max_steps`.
- [ ] Use cleanup policy `delete_after_sync` unless debugging requires keeping the pod.
- [ ] Validate the RunPod key.
- [ ] Launch one RunPod run.
- [ ] Confirm pod ID appears in UI.
- [ ] Confirm lifecycle logs show create, agent health, upload, start, sync, cleanup.
- [ ] Confirm metrics and stdout/stderr appear locally.
- [ ] Confirm final status is terminal.
- [ ] Confirm cleanup policy was applied in RunPod.
- [ ] Confirm no raw API key/token appears in logs or job metadata.

Exit criteria:

- [*] All required automated checks pass or have documented pre-existing failures.
- [ ] Manual smoke checks pass.
- [*] Real RunPod smoke test is either passed or explicitly not run with reason.
  Not run: requires explicit approval because it can create billable RunPod resources.

## Suggested Implementation Order

- [ ] Phase 0
- [ ] Phase 1
- [ ] Phase 2
- [ ] Phase 3
- [*] Phase 4
- [*] Phase 5
- [*] Phase 7
- [ ] Phase 8
- [*] Phase 9
- [*] Phase 10
- [*] Phase 6
- [*] Phase 11
- [*] Phase 12
- [*] Phase 13
- [*] Phase 14
- [*] Phase 15
- [*] Phase 16
- [*] Phase 17
- [*] Phase 18
- [*] Phase 19
- [ ] Phase 20

Reason for this order:

- Backend tests and route extraction come first because RunPod training bugs are expensive and hard to reproduce.
- Preflight and manager extraction should happen before deep executor surgery because job creation is the shared local/remote path.
- Executor and sync refactors should happen before frontend UX changes so the UI describes a stable backend contract.
- Restart semantics are intentionally after the core split because they require clear executor/token boundaries.
- Style splitting comes after component extraction because selectors become easier to group once the markup is modular.

## Refactor Completion Definition

The refactor is complete when all of these are true:

- [*] RunPod training can still launch, report telemetry, finish, sync outputs, and clean up.
  Verified with fake RunPod/pod-agent coverage; real billable smoke was intentionally not run.
- [*] Local training can still launch and report telemetry.
- [*] No raw RunPod API key or pod-agent token is persisted or logged.
- [*] Backend training code is organized into focused modules with compatibility shims removed or documented.
- [*] Remote-agent code has small route/path/file/process boundaries.
- [*] Frontend training page is split into hooks and components that can be tested independently.
- [*] Training API client is split into type/client/job/provider modules.
- [*] Training CSS is split by page region.
- [*] Restart behavior for incomplete RunPod jobs is explicit and tested.
- [*] Network volume and cleanup UI copy matches actual backend behavior.
- [*] Automated backend and frontend checks pass, or remaining failures are documented as unrelated pre-existing issues.
- [*] RunPod docs are updated to match the final implementation.
