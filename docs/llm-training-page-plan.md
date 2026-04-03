# LLM Training Page Plan

## Goal
Add a first-class `LLM Training` page to `apps/llm-studio/web` that lets a user:

1. Pick a saved model config artifact and a completed tokenizer artifact from the home-page asset manager.
2. Configure dataset + runtime settings with sane defaults and advanced controls hidden by default.
3. Validate the full run before launch.
4. Start a real training job from the unified app.
5. Watch the run clearly: status, progress, loss, LR, grad norm, throughput, ETA, samples, checkpoints, logs, errors.
6. Keep all outputs and metadata in the workspace in a way that feels consistent with existing model and tokenizer flows.

This plan is intentionally grounded in the current codebase, not an imagined architecture.

---

## Current Project Analysis

### What already exists
- `apps/llm-studio/web/app/page.tsx`
  - Home page with `WorkspaceAssetManager`.
  - Uses `useWorkspaceAssetInventory()` from `apps/llm-studio/web/lib/workspaceAssets.ts`.
  - Today it knows only two asset types: `"model"` and `"tokenizer"`.
- `apps/llm-studio/web/app/studio/*`
  - Model designer page.
  - Persists saved model configs through `/api/v1/projects`.
  - Supports deep-linking with `?project=<id>`.
- `apps/llm-studio/web/app/tokenizer/page.tsx`
  - The strongest UI reference for the new page.
  - Already has the exact interaction pattern we want to reuse:
    - sticky top nav
    - step-based workflow card
    - automatic validation
    - active job panel
    - recent jobs panel
    - periodic polling
    - advanced settings collapsed behind `details`
    - local persistence + query-param deep links
- `apps/llm-studio/api/app/main.py`
  - Model config validation, analysis, and saved-project CRUD already exist.
  - Tokenizer config validation, train-file upload, tokenizer jobs, artifact download, and preview already exist.
- `training/*`
  - Real model-training runtime already exists.
  - Important files:
    - `training/train.py`
    - `training/training_config.py`
    - `training/dataloader_config.py`
    - `training/memory_estimator.py`
    - `training/logger.py`
    - `training/checkpoint_manager.py`
- `docs/training-dataloader.md`
  - Clear documentation for the training dataloader and its schema.

### What does not exist yet
- No `/api/v1/training/*` namespace.
- No persisted model-training job store.
- No model-training page route in Next.js.
- No training asset type in `useWorkspaceAssetInventory()`.
- No home-page flow for pairing one model config + one tokenizer and launching training.
- No live training telemetry endpoints.
- No job-safe runtime wrapper around `training/train.py`.

### Important technical constraints discovered in the code
- `training/train.py` is not API-ready yet.
  - It hardcodes:
    - `training/training_config.json`
    - `training/dataloader_config.json`
    - `trained_tokenizer.json`
    - `model/gpt2_config.json`
    - `stats.jsonl`
    - `samples.jsonl`
    - `checkpoints/`
- `training/logger.py` writes JSONL files and stdout, but it does not emit structured callbacks to an API job manager.
- `training/checkpoint_manager.py` writes to a fixed directory, not a job-specific workspace folder.
- `training/train.py` asserts `model_config.vocab_size == tokenizer.get_vocab_size()`.
  - The new UI must surface this explicitly before a run.
- `training/training_config.py` requires:
  - `sum(lr_scheduler.schedulers[*].steps) == max_steps`
- `training/memory_estimator.py` already exposes enough data to build an excellent preflight panel and runtime summary.
- The tokenizer page currently uses polling, not sockets or SSE.
  - Matching that pattern for v1 is safer and more consistent.
- The tokenizer page has its own route-local CSS (`app/tokenizer/globals.css`) while the rest of the app uses `app/styles/*`.
  - Creating a third isolated style island for training would be the wrong direction.

### Architectural conclusion
The new training page should not call `training/train.py` directly as a shared in-process function from FastAPI threads. Model training is long-running, stateful, GPU-heavy, and failure-prone. It should run in a dedicated subprocess with job-owned directories, structured metadata, and pollable outputs.

---

## Product Principles

### UX principles
- Default-simple, advanced-powerful.
- The first screen should answer three questions immediately:
  - What model config am I training?
  - What tokenizer am I using?
  - Is this run safe to start?
- Nothing should require raw JSON until the user explicitly opens advanced sections.
- Every blocking validation should explain how to fix it.
- The run monitor should make it impossible to wonder whether training is alive.

### Engineering principles
- Reuse proven patterns from the tokenizer page instead of inventing a totally different flow.
- Keep all training-run data job-scoped and self-contained on disk.
- Prefer polling JSON endpoints for v1.
- Structure the backend so SSE can be added later without rewriting the job model.
- Avoid new heavyweight frontend dependencies unless they materially improve clarity.
- Do not add another large monolithic page without extracting some shared pieces first.

---

## Recommended UX

### Route and naming
- Route: `/training`
- Page title: `LLM Training`
- Nav label: `Training`
- Query params:
  - `project=<project_id>`
  - `tokenizerJob=<job_id>`
  - optionally later: `run=<training_job_id>`

### Home-page launch flow
Enhance the asset manager instead of forcing users to manually remember IDs.

Recommended behavior:
- Keep existing card click behavior:
  - model card click -> `/studio?project=...`
  - tokenizer card click -> `/tokenizer?job=...`
- Add a lightweight `Training Launchpad` strip above the asset grid.
- Each asset card gets a small action:
  - model card: `Use as model`
  - tokenizer card: `Use as tokenizer`
- Launchpad shows:
  - selected model config
  - selected tokenizer
  - `Open Training Page` button
  - `Clear` button
- When both selections are present, route to:
  - `/training?project=<id>&tokenizerJob=<id>`

This is the cleanest solution because it keeps the asset manager as the source of truth without breaking the existing direct-open behavior.

### Training page layout
Mirror the tokenizer page structure because it already fits this product well.

Recommended top-to-bottom layout:

1. Sticky top nav
2. Hero / run composer card
3. Workflow steps card
4. Two-column results area
   - left: active run monitor
   - right: recent runs
5. Configuration studio
6. Generated JSON / artifacts / debug outputs
7. Toast notifications

### Workflow steps
Use a five-step deck similar to tokenizer training:

1. Choose saved model config
2. Choose completed tokenizer artifact
3. Configure dataset + run settings
4. Validate compatibility + estimate memory/runtime
5. Start training

Each step should have:
- readiness state
- one-sentence explanation
- action button that jumps to the relevant settings area

### Configuration philosophy
Show only high-value fields first.

Default-visible settings:
- Dataset source list
- Local file vs HF dataset entries
- `seq_len`
- `max_steps`
- `total_batch_size`
- `learning rate`
- `weight decay`
- `save_every`
- `sample_every`
- prompt list for sampling

Collapsed advanced settings:
- BOS/EOS/PAD controls
- token dtype
- pretokenize batch size
- cache dir
- shuffle + mixing
- optimizer betas / eps
- scheduler composition details
- node split / distributed overrides
- raw generated JSON

### Live run monitor
The active run panel should show:

- Run ID
- Status badge
- Human-readable stage
- Progress bar
- Model config name
- Tokenizer name
- Device/runtime summary
- Created / started / finished timestamps
- Elapsed time
- ETA when derivable

Status cards under the header:
- Current step / max steps
- Latest loss
- Latest grad norm
- Latest LR
- Latest tokens/sec
- Checkpoint count
- Sample count
- Last log update age
- Memory estimate / fit status

Tabs or stacked subpanels:
- `Metrics`
  - loss chart
  - lr chart
  - grad norm chart
  - throughput chart
- `Samples`
  - grouped by step and prompt index
- `Checkpoints`
  - saved steps + download buttons
- `Logs`
  - raw stdout/stderr tail
- `Run Details`
  - resolved configs and derived runtime values

### Preflight quality checks
Before the start button unlocks, validate:

- model config loads
- tokenizer job exists and is completed
- tokenizer artifact file exists
- tokenizer vocab size matches model `vocab_size`
- training `seq_len <= model.context_length`
- scheduler total steps equals `max_steps`
- dataset config validates
- training config validates
- BOS/EOS/PAD tokens exist in the tokenizer if configured
- memory estimation produces a valid positive batch size
- derived grad accumulation is valid
- local dataset files exist if local files are used

Where possible, include one-click fixes:
- `Sync model vocab_size to tokenizer vocab`
- `Set seq_len to model context_length`
- `Load starter optimizer defaults`
- `Load starter scheduler template`

---

## Recommended Backend Architecture

### New API namespace
Add a new namespace under `apps/llm-studio/api/app/main.py`:

- `/api/v1/training/health`
- `/api/v1/training/config/templates`
- `/api/v1/training/config/schemas`
- `/api/v1/training/validate/dataloader`
- `/api/v1/training/validate/training-config`
- `/api/v1/training/validate/preflight`
- `/api/v1/training/jobs`
- `/api/v1/training/jobs/{job_id}`
- `/api/v1/training/jobs/{job_id}/metrics`
- `/api/v1/training/jobs/{job_id}/samples`
- `/api/v1/training/jobs/{job_id}/logs`
- `/api/v1/training/jobs/{job_id}/checkpoints`
- `/api/v1/training/jobs/{job_id}/artifact`
- `/api/v1/training/jobs/{job_id}/stop`
- optionally later: `/resume`

### Job execution model
Use a subprocess-based manager, not a thread-only manager.

Why:
- isolates PyTorch runtime and CUDA state
- easier cancellation
- easier log capture
- easier crash containment
- better future compatibility with distributed launches

Recommended components:
- `TrainingRunManager`
- `TrainingRunStore`
- `training_runner.py` or a refactored `training/train.py` entrypoint that accepts arguments

### Job directory layout
Under the llm-studio data directory, create a dedicated model-training area:

`<data_dir>/training/jobs/<job_id>/`

Recommended contents:
- `metadata.json`
- `model_config.json`
- `tokenizer.json` or `tokenizer_artifact.json`
- `training_config.json`
- `dataloader_config.json`
- `resolved_preflight.json`
- `stdout.log`
- `stderr.log`
- `stats.jsonl`
- `samples.jsonl`
- `checkpoints/<step>/model-<step>.pt`
- `checkpoints/<step>/optimizer-<step>.pt`
- `checkpoints/<step>/meta-<step>.pt`
- `artifact_manifest.json`

This makes each run portable, debuggable, and easy to expose through the API.

### Persistent job model
Create a dedicated job record similar to tokenizer jobs, but with training-specific fields.

Suggested stored fields:
- `id`
- `status`
- `state`
- `stage`
- `progress`
- `created_at`
- `started_at`
- `finished_at`
- `project_id`
- `project_name`
- `tokenizer_job_id`
- `tokenizer_name`
- `model_config`
- `training_config`
- `dataloader_config`
- `resolved_runtime`
- `artifact_dir`
- `stats_path`
- `samples_path`
- `stdout_path`
- `stderr_path`
- `last_step`
- `max_steps`
- `latest_loss`
- `latest_grad_norm`
- `latest_lr`
- `latest_tokens_per_sec`
- `checkpoint_count`
- `sample_count`
- `error`
- `process_id`

Recommended states:
- `queued`
- `preflight`
- `estimating_memory`
- `initializing_model`
- `building_dataloader`
- `training`
- `checkpointing`
- `sampling`
- `finalizing`
- `completed`
- `failed`
- `cancelled`

### Training runtime refactor
The current training runtime needs a thin job wrapper before it can power the web app cleanly.

Refactor goals:
- accept explicit file paths or in-memory config payloads
- accept output directory
- accept logger hooks / callback hooks
- stop depending on the process CWD
- write structured metrics and samples to job-scoped files
- surface stage transitions
- capture derived runtime data:
  - device type
  - chosen batch size
  - max allowed batch size
  - grad accumulation steps
  - memory estimate summary

Recommended change set in `training/*`:
- turn `training/train.py` into a callable job runner:
  - `run_training_job(args: TrainingRunArgs, reporter: TrainingReporter | None = None)`
- make `Logger` callback-aware
- make `CheckpointManager` take an explicit root directory
- emit structured step events in addition to JSONL writes

### Polling model
For v1, poll every `1.5s` to `2s`, just like tokenizer jobs.

Expose separate lightweight endpoints:
- run summary
- metrics tail
- samples tail
- checkpoint list
- logs tail

That keeps the page responsive without over-fetching one large payload.

---

## Recommended Frontend Architecture

### New web API client
Add `apps/llm-studio/web/lib/trainingApi.ts`.

It should own:
- types
- request helper reuse
- config template/schema fetchers
- preflight validation
- training job CRUD
- metrics / samples / logs / checkpoints fetchers
- stop-job action

### New route
Add:
- `apps/llm-studio/web/app/training/page.tsx`
- optionally `apps/llm-studio/web/app/training/layout.tsx`

### Styling strategy
Do not create a third giant route-local style system like `tokenizer/globals.css`.

Recommended approach:
- extract the shared visual shell from tokenizer page into reusable app-level styles
- keep only training-specific pieces route-local

Good candidates to extract:
- sticky nav shell
- panel card shell
- workflow step tiles
- progress track
- job badge styles
- recent-job list styles
- toast styles
- meta list / stats card styles

Possible file additions:
- `apps/llm-studio/web/app/styles/training-shell.css`
- `apps/llm-studio/web/app/styles/panels/training-monitor.css`
- `apps/llm-studio/web/app/styles/panels/training-charts.css`

### Suggested component breakdown
Keep the page composable from the start.

Suggested components:
- `TrainingTopNav`
- `TrainingHero`
- `TrainingLaunchSummary`
- `TrainingWorkflowSteps`
- `TrainingAssetPairingPanel`
- `TrainingPreflightPanel`
- `TrainingRunMonitor`
- `TrainingMetricCards`
- `TrainingChartsPanel`
- `TrainingSamplesPanel`
- `TrainingCheckpointsPanel`
- `TrainingLogsPanel`
- `RecentTrainingRunsCard`
- `TrainingSettingsStudio`
- `TrainingConfigJsonPanel`

Suggested hooks:
- `useTrainingPageController`
- `useTrainingAssetSelection`
- `useTrainingConfigState`
- `useTrainingPreflight`
- `useTrainingJobs`
- `useActiveTrainingRunPolling`

### Home-page integration changes
Modify:
- `apps/llm-studio/web/app/components/WorkspaceAssetManager.tsx`
- `apps/llm-studio/web/app/page.tsx`
- `apps/llm-studio/web/app/workspace-home.module.css`

Add:
- selection state for one model + one tokenizer
- launchpad strip
- `Open Training Page` CTA
- optional later: training-run cards in the same manager

### Inventory hook evolution
Eventually extend `apps/llm-studio/web/lib/workspaceAssets.ts` to support:
- `type: "training_run"`

This can come after the core training page is working, but the plan should reserve for it now so we do not paint ourselves into a corner.

---

## Step-by-Step Implementation Plan

### Phase 1: Add training templates, schemas, and settings plumbing
Goal: make the backend able to serve training config defaults and schemas exactly like the tokenizer workspace does today.

Files to touch:
- `apps/llm-studio/api/app/config.py`
- `apps/llm-studio/api/app/main.py`
- `apps/llm-studio/api/templates/` or explicit root-path loading from `training/`

Tasks:
1. Add constants for:
   - training config template path
   - training config schema path
   - training dataloader template path
   - training dataloader schema path
2. Decide whether to:
   - load templates directly from root `training/*.json`, or
   - copy stable versions into `apps/llm-studio/api/templates/`
3. Add `/api/v1/training/config/templates`.
4. Add `/api/v1/training/config/schemas`.
5. Add a minimal `/api/v1/training/health`.

Acceptance:
- frontend can fetch starter configs and schemas without knowing filesystem paths

### Phase 2: Refactor the training runtime into a job-safe runner
Goal: make `training/*` callable by a job manager without hardcoded paths or shared output locations.

Files to touch:
- `training/train.py`
- `training/logger.py`
- `training/checkpoint_manager.py`
- optionally new `training/runner.py`

Tasks:
1. Replace hardcoded config reads with explicit inputs.
2. Replace fixed output files with job-scoped paths.
3. Add a small runtime arg model, for example:
   - model config path
   - tokenizer artifact path
   - training config path
   - dataloader config path
   - output dir
4. Make logger write both:
   - stdout summaries
   - JSONL files
   - optional callback hooks for live updates
5. Make checkpoint manager save into `<job_dir>/checkpoints`.
6. Capture derived runtime values and return them or write them to `metadata.json`.
7. Ensure failures raise structured exceptions with useful messages.

Acceptance:
- one isolated run can execute fully with custom paths without writing to repo root

### Phase 3: Add model-training job models and persistence
Goal: make training jobs first-class persisted workspace entities.

Files to touch:
- `apps/llm-studio/api/app/training_models.py` new
- `apps/llm-studio/api/app/training_storage.py` new
- `apps/llm-studio/api/app/config.py`

Tasks:
1. Create request/response models:
   - create job
   - validate preflight
   - job summary
   - metrics list
   - samples list
   - checkpoints list
   - log tail
2. Create a SQLAlchemy store modeled after tokenizer storage.
3. Create runtime directories for training jobs.
4. Decide if training and tokenizer can share a database file or use separate tables in one DB.
5. Add cleanup behavior for interrupted jobs on API restart.

Acceptance:
- training jobs survive page reloads and API restarts

### Phase 4: Build the training job manager
Goal: launch, track, and stop real model-training runs.

Files to touch:
- `apps/llm-studio/api/app/training_jobs.py` new
- `apps/llm-studio/api/app/main.py`

Tasks:
1. Implement `TrainingRunManager`.
2. Launch each run in a subprocess.
3. Store and update:
   - status
   - stage
   - progress
   - pid
   - latest metric summary
4. Parse `stats.jsonl`, `samples.jsonl`, and checkpoint directories into API responses.
5. Capture stdout/stderr to files.
6. Add stop/cancel support.
7. Expose download endpoint for the run artifact bundle or manifest.

Acceptance:
- create, inspect, poll, and stop a training job through API endpoints only

### Phase 5: Add training-specific validation and preflight
Goal: prevent bad runs before they start.

Files to touch:
- `apps/llm-studio/api/app/main.py`
- `apps/llm-studio/api/app/training_models.py`
- optionally helper modules under `apps/llm-studio/api/app/`

Tasks:
1. Validate training dataloader config with `TrainingDataloaderConfig`.
2. Validate training loop config with `TrainingConfig`.
3. Load the selected model project and tokenizer artifact.
4. Compute tokenizer vocab size from the artifact.
5. Run compatibility checks:
   - vocab size
   - seq length
   - special tokens
   - scheduler math
   - dataset file existence
6. Run memory estimation using the real model config and selected training config.
7. Return a rich preflight response:
   - pass/fail
   - warnings/errors
   - derived runtime values
   - memory estimate summary
   - recommended fixes

Acceptance:
- the frontend can unlock `Start Training` only when preflight passes

### Phase 6: Add the frontend training API layer and controller hooks
Goal: make the web app capable of talking to the new training backend cleanly.

Files to touch:
- `apps/llm-studio/web/lib/trainingApi.ts` new
- `apps/llm-studio/web/app/training/*` new

Tasks:
1. Create typed client methods for all new training endpoints.
2. Create query-param hydration for `project` and `tokenizerJob`.
3. Add local persistence for:
   - selected assets
   - dataset settings
   - training settings
   - active run id
4. Add polling hooks for:
   - active run summary
   - metrics tail
   - samples tail
   - logs tail
   - checkpoints

Acceptance:
- page can restore itself after refresh with the same active run and settings

### Phase 7: Add home-page asset-manager launch flow
Goal: let users choose artifacts from the home page exactly as requested.

Files to touch:
- `apps/llm-studio/web/app/components/WorkspaceAssetManager.tsx`
- `apps/llm-studio/web/app/page.tsx`
- `apps/llm-studio/web/app/workspace-home.module.css`

Tasks:
1. Add `Use as model` and `Use as tokenizer` actions on asset cards.
2. Add a `Training Launchpad` strip.
3. Keep current open-on-card-click behavior unchanged.
4. Route to `/training?project=...&tokenizerJob=...`.
5. Prevent invalid launches:
   - no incomplete tokenizer jobs
   - missing pair selection

Acceptance:
- a user can launch training from the home-page asset manager without manual copying

### Phase 8: Build the training page UI shell
Goal: land the route and make the page feel native to the current app.

Files to touch:
- `apps/llm-studio/web/app/training/page.tsx`
- shared styles under `apps/llm-studio/web/app/styles/*`
- optionally `apps/llm-studio/web/app/training/layout.tsx`

Tasks:
1. Add nav entry for `Training` on:
   - home
   - studio
   - tokenizer
   - training
2. Build hero card with selected artifact summary.
3. Build workflow step deck.
4. Build settings studio with collapsed advanced sections.
5. Reuse existing tone, spacing, card, and badge patterns.
6. Avoid exposing raw JSON first.

Acceptance:
- the page looks like it belongs to the app already

### Phase 9: Build the preflight and start-run UX
Goal: make starting a run feel safe and obvious.

Files to touch:
- `apps/llm-studio/web/app/training/*`

Tasks:
1. Run preflight automatically after debounced setting changes.
2. Surface blocking issues inline with clear language.
3. Add one-click corrective actions where possible.
4. Show derived runtime preview:
   - device
   - estimated memory
   - derived micro-batch
   - grad accumulation
   - expected checkpoints/sample cadence
5. Unlock `Start Training` only when the run is valid.

Acceptance:
- the user understands exactly what will happen before launching

### Phase 10: Build the live monitor
Goal: make the run fully observable.

Files to touch:
- `apps/llm-studio/web/app/training/*`

Tasks:
1. Build active-run header and progress bar.
2. Add metric cards.
3. Add charts for:
   - loss
   - lr
   - grad norm
   - tokens/sec
4. Add sample viewer.
5. Add checkpoint browser.
6. Add logs tail.
7. Add `Stop Training` button for running jobs.

Implementation note:
- Use simple custom SVG or CSS-based charts before adding a charting dependency.
- The current app is intentionally lean; keep it that way unless charts become hard to read.

Acceptance:
- a user can tell if training is healthy without opening terminal output

### Phase 11: Add recent runs and deep-linking
Goal: make training jobs navigable like tokenizer jobs.

Files to touch:
- `apps/llm-studio/web/app/training/*`

Tasks:
1. Add recent-runs list with badges and last-known progress.
2. Persist hidden/removed recent jobs locally if that interaction is desired.
3. Deep-link selected run with `?run=<job_id>` later if useful.
4. Default to the newest visible run when there is no explicit selection.

Acceptance:
- users can jump between past and current runs quickly

### Phase 12: Integrate outputs into workspace assets
Goal: make model training feel like part of the same workspace, not a sidecar tool.

Files to touch:
- `apps/llm-studio/web/lib/workspaceAssets.ts`
- `apps/llm-studio/web/app/components/WorkspaceAssetManager.tsx`
- API listing endpoints for training runs

Tasks:
1. Introduce a new asset type, likely `training_run`.
2. Show:
   - run name
   - source model config
   - source tokenizer
   - status
   - created time
   - output size if meaningful
3. Add download action for run bundle / artifact manifest.
4. Update home-page summary cards to include model training activity.

Acceptance:
- completed and running LLM training jobs are visible from the home workspace

### Phase 13: Tests and quality gates
Goal: land the feature without regressing the existing app.

Backend tests:
- config template/schema endpoints
- validation endpoints
- preflight mismatch handling
- create/list/get/stop training jobs
- metric/log/checkpoint parsing
- interrupted-job recovery

Frontend tests or verification tasks:
- query-param hydration
- disabled/enabled start button logic
- asset launchpad behavior
- recent-runs selection behavior
- polling lifecycle
- dark/light theme correctness

Manual QA scenarios:
- valid local-file run
- valid streaming HF run
- tokenizer/model vocab mismatch
- seq_len too large
- missing special token
- failed training subprocess
- cancelled run
- completed run with samples and checkpoints

---

## File Touch Map

### Definitely new or changed backend files
- `apps/llm-studio/api/app/main.py`
- `apps/llm-studio/api/app/config.py`
- `apps/llm-studio/api/app/training_models.py`
- `apps/llm-studio/api/app/training_storage.py`
- `apps/llm-studio/api/app/training_jobs.py`
- `apps/llm-studio/api/tests/test_main.py`
- optionally new test files for training API coverage

### Definitely new or changed frontend files
- `apps/llm-studio/web/app/page.tsx`
- `apps/llm-studio/web/app/components/WorkspaceAssetManager.tsx`
- `apps/llm-studio/web/app/workspace-home.module.css`
- `apps/llm-studio/web/lib/workspaceAssets.ts`
- `apps/llm-studio/web/lib/trainingApi.ts`
- `apps/llm-studio/web/app/training/page.tsx`
- `apps/llm-studio/web/app/training/layout.tsx`
- `apps/llm-studio/web/app/styles/*` for shared training/studio shell extraction
- nav components in studio and tokenizer routes

### Definitely new or changed runtime files
- `training/train.py`
- `training/logger.py`
- `training/checkpoint_manager.py`
- optionally `training/runner.py`

---

## Recommended v1 Scope Boundary

### In scope for the first serious implementation
- home-page asset selection into training page
- training page route
- training config UI
- preflight validation
- run launch
- live polling monitor
- metrics, logs, samples, checkpoints
- stop/cancel
- recent runs

### Best treated as follow-up once v1 is solid
- resume from checkpoint
- side-by-side run comparison
- tensorboard export
- websocket/SSE streaming
- multi-run queues and scheduling policies
- hyperparameter sweep UI

---

## Final Acceptance Checklist

- A user can open home page, choose one saved model config and one completed tokenizer artifact, and launch the training page with both preselected.
- The training page feels visually consistent with home, studio, and tokenizer pages.
- The page is simple on first load and still exposes deep control through advanced sections.
- Preflight catches invalid runs before launch.
- Training jobs run through the app, not through manual terminal steps.
- Live monitoring clearly shows progress, loss, lr, grad norm, throughput, logs, samples, and checkpoints.
- The page is usable in both light and dark themes.
- Outputs are stored in a stable workspace location and are recoverable after refresh or API restart.
- The implementation adds tests for the new backend and does not regress current project or tokenizer flows.
