# LLM Studio Simple Mode Plan

## Executor Instructions

This document is the execution plan for adding Simple Mode to `apps/llm-studio/web`.

Use this checklist as the source of truth while implementing. Mark a task as `[*]` only after the described code, tests, and verification for that task are complete. Leave unfinished work as `[ ]`. If implementation reveals a better approach, update this file first so the plan stays accurate.

Legend:
- `[ ]` not started
- `[*]` completed

## Goal

Add a navbar switch that lets users move between the existing expert interface and a new Simple Mode.

Simple Mode must give non-expert users one guided flow:

1. Create a model architecture from a strong template.
2. Train a tokenizer with matching defaults.
3. Train the model with recommended training settings.
4. Run inference against the trained model.

The existing expert pages should remain available, but Simple Mode should be the primary path for users who do not already know LLM architecture, tokenization, training schedules, batch sizes, learning rates, or inference sampling settings.

## Current App Reality

The app already has the primitives needed for this flow:

- `app/studio` builds and saves model configs through `/api/v1/projects`.
- `app/tokenizer` trains tokenizer jobs through `/api/v1/tokenizer/jobs`.
- `app/training` selects a saved model and completed tokenizer, runs preflight, applies recommendations, starts model training, and tracks jobs.
- `app/inference` selects completed training runs and checkpoints, then streams generation.
- `lib/workspaceAssets.ts` already unifies models, tokenizers, and training runs for the home workspace.

The main UX problem is not missing capability. The problem is that users must understand the full expert surface and manually connect separate pages. Simple Mode should orchestrate the same backend capabilities into one constrained path.

## Product Principles

- Simple Mode is a guided product flow, not a watered-down copy of every expert page.
- Expert Mode must remain intact for advanced users.
- A beginner should never need to edit raw JSON.
- A beginner should never need to know learning-rate scheduler internals, optimizer betas, micro-batch size, GQA constraints, or tokenizer decoder strategy.
- Defaults should be recommended by the app whenever the backend can derive a recommendation.
- The user should always know the current step, what is blocking them, and the next action.
- Simple Mode must create real workspace artifacts that remain compatible with Expert Mode.
- Any template named after a known model family must be honest about support. If the runtime cannot exactly reproduce LLaMA, Mistral, GPT-2, or Pythia, label the template as `family-inspired` or `size-style`, not as an exact clone.

## Recommended UX Shape

### Navigation

Add a shared top navigation component used by home, model studio, tokenizer, training, inference, and the new simple route.

In Expert Mode:
- Brand: `LLM Builder`
- Links: `Home`, `Model Studio`, `Tokenizer`, `Training`, `Inference`
- Right controls: `Simple Mode` switch, theme toggle

In Simple Mode:
- Brand: `LLM Builder`
- Links: `Guide`, `Workspace`
- Right controls: `Expert Mode` switch, theme toggle
- The switch should persist mode and navigate to `/simple`.

Do not keep five separate nav implementations after this work. The current duplicated files are:
- `app/home/components/HomeNavigation.tsx`
- `app/studio/components/page/StudioTopNav.tsx`
- `app/tokenizer/components/TokenizerStudioNav.tsx`
- `app/training/components/TrainingStudioNav.tsx`
- `app/inference/components/InferenceTopNav.tsx`

### Simple Route

Add a new route:

- `apps/llm-studio/web/app/simple/page.tsx`

Use route-local structure:

- `app/simple/components/`
- `app/simple/hooks/`
- `app/simple/lib/`
- `app/simple/constants.ts`
- `app/simple/types.ts`

The route should show one guided page with a stepper and focused panels. It should not send beginners across four separate expert pages unless they explicitly choose an expert escape hatch.

### Stepper

Use a persistent stepper at the top of `/simple`:

1. `Architecture`
2. `Tokenizer`
3. `Training`
4. `Inference`

Each step must show:
- Ready, blocked, running, failed, or completed state
- One concise blocker message
- The next primary action
- The artifact created by that step when available

### Page Density

Only the active step should be fully expanded by default. Completed steps collapse into summary rows with an edit button. Future blocked steps show why they are blocked.

Do not place expert controls in the main simple flow. If a setting needs to exist for debugging, place it under a collapsed `Advanced details` disclosure or link to the matching expert page.

### Mode And Routing Rules

Simple Mode should behave predictably from every route:

- Clicking `Simple Mode` from any expert page writes `llm-studio-ui-mode-v1 = "simple"` and navigates to `/simple`.
- Clicking `Expert Mode` from `/simple` writes `llm-studio-ui-mode-v1 = "expert"` and navigates to the most relevant expert route for the current step:
  - Architecture step -> `/studio`
  - Tokenizer step -> `/tokenizer`
  - Training step -> `/training`
  - Inference step -> `/inference`
- If no current step is known, Expert Mode should navigate to `/`.
- Direct visits to expert routes must still work. Do not hard-block bookmarked expert URLs.
- In Simple Mode, expert routes may show an unobtrusive `Back to Simple Guide` affordance, but the route content should not be hidden from a direct URL.
- Query params may override initial mode for shareable links:
  - `?mode=simple` sets Simple Mode and navigates or renders accordingly.
  - `?mode=expert` sets Expert Mode.
- Do not add query params to every internal link. Persisted mode should handle normal navigation.

### Simple Mode Copy Rules

Simple Mode copy should be concise and action-oriented:

- Prefer `Choose a template`, `Train tokenizer`, `Run preflight`, and `Generate`.
- Avoid expert jargon in primary UI. If a term is necessary, put the explanation in a tooltip or collapsed detail.
- Do not say a model is "GPT-2" or "LLaMA" unless the backend exactly implements that architecture.
- Use status copy to explain blockers, not long instructional paragraphs.
- Primary buttons should always move the flow forward or perform the current step's main action.

## Simple Flow Details

### Step 1: Create Model Architecture

The user should choose a template, name the model, and create the architecture.

Visible controls:
- Template cards
- Model name
- Optional target size selector only when a template has safe variants
- Target vocabulary size if no tokenizer exists yet
- Target context length if the template supports safe variants
- Primary action: `Create architecture`

Hidden defaults:
- Block layout
- Norm placement
- Attention head counts
- Key/value head counts
- MLP sequence
- MLP multiplier
- Activation
- Weight tying

Behavior:
- Generate a `ModelConfig` from the selected preset.
- Validate locally and with the backend.
- Analyze parameter count with `/api/v1/analyze/model`.
- Save the config as a project with `createProject`.
- Store the returned `project_id` in Simple Mode state.
- If the tokenizer later trains with a different actual vocabulary size, update the saved project so `model_config.vocab_size` matches the tokenizer before model training.

Template cards should show:
- Best use
- Expected relative size
- Context length
- Head layout
- Norm and activation family
- Estimated parameter count from backend analysis, not hard-coded marketing numbers
- Hardware warning when the preset is unlikely to fit locally

### Step 2: Train Tokenizer

The user should provide text data and start tokenizer training.

Visible controls:
- Dataset source: `Upload text files` or `Use starter dataset`
- Optional `Use streaming dataset template` for users who want public data
- Tokenizer name
- Primary action sequence: `Validate tokenizer` -> `Train tokenizer`

Hidden defaults:
- `tokenizer_type`: `bpe`
- `byte_fallback`: `true`
- `min_frequency`: `2`
- `special_tokens`: `<|endoftext|>`, `<|pad|>`
- `pre_tokenizer`: `byte_level`
- `decoder`: `byte_level`
- Budget unit: `chars`
- Budget behavior: `truncate`
- Evaluation thresholds
- HF filters unless streaming template is explicitly selected

Behavior:
- Set tokenizer `vocab_size` from the model architecture target.
- Reuse existing upload and tokenizer-job APIs.
- Auto-validate when config and dataset are complete.
- Start a real tokenizer job.
- Poll the tokenizer job until completed, failed, or dismissed.
- On completion, store `tokenizer_job_id` in Simple Mode state.
- Preview tokenization with a small default text, but keep it as confirmation rather than a required step.
- Update the saved model project vocabulary to the completed tokenizer vocab if necessary.

### Step 3: Train Model

The user should confirm a recommended training plan and start training.

Visible controls:
- Selected model summary
- Selected tokenizer summary
- Dataset source: default to the same local files used for tokenizer training when available
- Training profile: `Quick check`, `Balanced`, `Longer run`
- Execution target: `Local machine` by default, `RunPod` only when selected or when local preflight cannot fit
- Run name
- Primary action sequence: `Run preflight` -> `Start training`

Hidden defaults:
- Scheduler composition
- Optimizer betas and epsilon
- Weight decay unless profile needs to surface it
- Micro batch size
- Gradient accumulation
- Shuffle buffer
- Token dtype
- BOS/EOS/PAD controls
- Node split and distributed controls
- Raw generated JSON

Training defaults:
- Start from backend `training_config_template` and `dataloader_config_template`.
- Clamp `seq_len` to the selected model context length.
- Use backend preflight to validate compatibility, memory, local files, scheduler, and tokenizer tokens.
- If preflight returns `batch_and_lr_recommendation`, Simple Mode should auto-select the recommended option.
- For `Quick check`, use a shorter safe run derived from the recommendation.
- For `Balanced`, use the backend recommended option directly.
- For `Longer run`, use the recommended batch and learning rate but extend max steps only within a conservative cap.
- If preflight provides recommended fixes, Simple Mode should apply safe deterministic fixes automatically after explaining them in the step status.

Training profile mapping:
- `Quick check`
  - Use recommended total batch size and learning rate when available.
  - Set max steps to `min(recommended_max_steps, 100)` for tiny local experiments, with a lower bound of `20`.
  - Set `sample_every` to `max(10, floor(max_steps / 4))`.
  - Set `save_every` to `max_steps` unless the backend requires an earlier checkpoint.
- `Balanced`
  - Use the backend recommended option exactly.
  - Refit scheduler phases to max steps with the existing scheduler helper.
  - Keep `sample_every` and `save_every` from the template unless they exceed max steps, then clamp them.
- `Longer run`
  - Use recommended total batch size and learning rate.
  - Set max steps to `min(recommended_max_steps * 2, 2000)` unless the dataset is tiny.
  - For tiny local datasets, cap at a conservative number of passes based on tokenizer/job stats or preflight signals.
  - Make the UI copy clear that longer runs are not guaranteed to improve output on small datasets.

Cloud guardrail:
- Never start a paid RunPod job as an invisible default.
- If local preflight cannot fit and RunPod is available, Simple Mode may recommend RunPod, but the user must explicitly confirm cloud execution and cleanup policy.

Behavior:
- Preselect the model project and tokenizer job created in previous steps.
- Build dataloader config from the selected dataset source.
- Run preflight whenever model, tokenizer, dataset, profile, or execution target changes.
- Unlock training only after preflight is valid.
- Start a real training job.
- Poll run status, metrics, samples, checkpoints, and logs using existing training hooks or factored shared helpers.
- Store `training_job_id` in Simple Mode state.

### Step 4: Inference

The user should generate from the latest checkpoint with minimal sampling controls.

Visible controls:
- Selected completed training run
- Checkpoint selector defaulting to `Latest checkpoint`
- Prompt text
- Generation length: `Short`, `Medium`, `Long`
- Creativity: `Precise`, `Balanced`, `Creative`
- Primary action: `Generate`

Hidden defaults:
- Numeric temperature
- Top K
- Seed
- Repetition penalty
- Raw checkpoint paths

Behavior:
- Auto-select the completed training run from Simple Mode state.
- Auto-select latest checkpoint.
- Map length options to `max_tokens`.
- Map creativity options to temperature, top_k, and repetition_penalty.
- Stream generation using the existing inference generation client.
- Show the prompt and continuation.
- Provide an expert link to open `/inference` with the selected run for advanced sampling.

Inference preset mapping:

| Simple option | Internal settings |
| --- | --- |
| Length `Short` | `max_tokens = 32` |
| Length `Medium` | `max_tokens = 64` |
| Length `Long` | `max_tokens = 128` |
| Creativity `Precise` | `temperature = 0.2`, `top_k = 20`, `repetition_penalty = 1.1` |
| Creativity `Balanced` | `temperature = 0.8`, `top_k = 50`, `repetition_penalty = 1.0` |
| Creativity `Creative` | `temperature = 1.0`, `top_k = 100`, `repetition_penalty = 1.0` |

Use `seed = 42` by default and hide it. If the user regenerates, either keep the seed for reproducibility or add a separate `Try another` action that increments the seed internally.

## Template Strategy

Create a typed preset registry in `apps/llm-studio/web/app/simple/lib/modelPresets.ts` or `apps/llm-studio/web/lib/modelPresets.ts` if shared by Expert Mode later.

Do not encode presets only as loose JSON. Use builders that enforce the same constraints the validator requires:

- `n_embd % n_head === 0`
- `(n_embd / n_head) % 2 === 0`
- `n_kv_head <= n_head`
- `n_head % n_kv_head === 0`
- At least one attention and MLP component per block
- MLP starts and ends with linear steps when multiplier is not `1`
- Model vocab size is synchronized to tokenizer vocab before model training

Recommended initial templates:

| Preset ID | User-facing name | Intent | Defaults | Honesty note |
| --- | --- | --- | --- | --- |
| `nano-gpt-quick` | NanoGPT-style quick model | Fast local smoke tests and tiny datasets | 4 blocks, 256 embed, 4 heads, 4 KV heads, 512 context, GELU, LayerNorm, MLP x4 | Inspired by small decoder-only GPT examples, not a pretrained nanoGPT checkpoint |
| `gpt2-small-style` | GPT-2 Small size baseline | Familiar dense transformer baseline | 12 blocks, 768 embed, 12 heads, 12 KV heads, 1024 context, GELU, LayerNorm, MLP x4 | Size/layout style only; runtime uses this app's RoPE and input/output norms |
| `pythia-160m-style` | Pythia/GPT-NeoX-style baseline | Efficient research baseline with modern positional behavior | 12 blocks, 768 embed, 12 heads, 12 KV heads, 2048 context, GELU, LayerNorm or RMSNorm after validation comparison, MLP x4 | Family-style preset, not exact Pythia architecture |
| `llama-tiny-gqa` | LLaMA-family tiny GQA model | Efficient local GPU run with GQA | 16 blocks, 1024 embed, 16 heads, 4 KV heads, 2048 context, RMSNorm, SiLU, MLP x4 | LLaMA-like because runtime has RoPE, RMSNorm, GQA, SiLU; not exact LLaMA until gated SwiGLU is supported |
| `gqa-balanced` | Efficient GQA balanced model | Default recommendation for users with a GPU | 12 blocks, 768 embed, 12 heads, 4 KV heads, 1024 or 2048 context, RMSNorm, SiLU, MLP x4 | App-native efficient template rather than a named external architecture |

Default preset:
- Use `nano-gpt-quick` if no GPU is detected or memory estimate is unavailable.
- Use `gqa-balanced` if preflight/memory analysis suggests the device can fit it.
- Do not guess hardware from the browser. Use backend analysis/preflight when possible.

Tokenizer defaults by preset:

| Preset group | Target vocab default | Tokenizer budget default |
| --- | ---: | ---: |
| Tiny/local quick presets | `1000` for starter dataset, `8000` for uploaded user data | `250000` chars for starter data, `2000000` chars for uploaded data |
| GPT-2/Pythia-style presets | `32000` unless user chooses starter dataset | `10000000` chars |
| LLaMA/GQA balanced presets | `32000` unless user chooses starter dataset | `10000000` chars |

Rules:
- If using the built-in starter dataset, keep vocab small enough to avoid many unused tokens.
- If the user uploads meaningful data, use the preset target vocab.
- If tokenizer training reports a different final vocab size, model sync wins.
- Show `Vocabulary will be matched automatically` instead of making beginners reason about embeddings.

## State And Persistence

Add a small app-mode state layer:

- Storage key: `llm-studio-ui-mode-v1`
- Values: `"simple"` or `"expert"`
- Hook: `useUiMode`
- Event: `llm-studio:ui-mode-change`

Add Simple Mode flow state:

- Storage key: `llm-studio-simple-flow-v1`
- Versioned payload
- Fields:
  - `version`
  - `presetId`
  - `modelName`
  - `targetVocabSize`
  - `projectId`
  - `tokenizerJobId`
  - `trainingJobId`
  - `datasetSource`
  - `localTrainFiles`
  - `trainingProfile`
  - `executionKind`
  - `checkpointValue`
  - `lastCompletedStep`

Persistence rules:
- Never persist secrets such as RunPod API keys or HF tokens in Simple Mode state.
- Keep artifact IDs and local file references only.
- On page load, rehydrate state and refresh live artifact status from the backend.
- If an artifact was deleted, mark the dependent step blocked and preserve enough state for the user to recover.

Step readiness model:

| Step | Ready condition | Blocked by | Running condition | Completed condition |
| --- | --- | --- | --- | --- |
| Architecture | Valid preset and model name | Invalid preset config, backend validation failure | Project create/update in flight | Saved project exists and validates |
| Tokenizer | Dataset available and tokenizer config valid | No dataset, invalid config, missing project | Tokenizer job pending/running | Tokenizer job completed with artifact |
| Training | Project and tokenizer complete, preflight valid | Missing project, incomplete tokenizer, invalid preflight, unconfirmed cloud target | Training job pending/running | Training job completed with checkpoint |
| Inference | Completed training run with checkpoint | No completed run, checkpoint load failure | Generation in flight | Last generation succeeded |

The UI should derive these states from refreshed backend data plus local flow state. Do not rely only on `lastCompletedStep`.

## Architecture Plan

### Shared Navigation

Create:

- `app/shared/components/AppTopNav.tsx`
- `app/shared/hooks/useUiMode.ts`
- `app/shared/lib/navigation.ts` if route metadata becomes noisy

Replace the five current nav components with `AppTopNav`.

Requirements:
- Preserve existing theme behavior from `lib/theme.ts`.
- Preserve `aria-current`.
- Use accessible switch semantics for Simple/Expert mode.
- Keep mobile behavior usable; nav links should wrap or collapse without overlapping.
- Do not make theme and mode controls visually ambiguous.

### Simple Controller

Create a controller hook:

- `app/simple/hooks/useSimpleModeController.ts`

It should compose smaller hooks:

- `useSimpleFlowPersistence`
- `useSimpleModelStep`
- `useSimpleTokenizerStep`
- `useSimpleTrainingStep`
- `useSimpleInferenceStep`

Keep network calls inside hooks or existing API clients, not inside presentational components.

### Simple Components

Recommended components:

- `SimpleModePageView`
- `SimpleStepper`
- `SimpleStepShell`
- `ArchitectureStep`
- `ModelPresetCard`
- `TokenizerStep`
- `SimpleDatasetPicker`
- `TrainingStep`
- `TrainingProfilePicker`
- `SimplePreflightSummary`
- `SimpleRunMonitor`
- `InferenceStep`
- `SimpleArtifactSummary`
- `SimpleBlockedState`

### Reuse Existing Code Carefully

Reuse pure helpers and API clients:

- `lib/api.ts`
- `lib/tokenizerLegacyApi.ts`
- `lib/training/jobs.ts`
- `lib/training/generation.ts`
- `app/training/lib/object.ts`
- `app/training/lib/display.ts`
- `app/training/lib/run.ts`
- `app/tokenizer/lib/dataset.ts`
- `app/inference/lib/formatters.ts`

Do not import large expert presentational panels directly into Simple Mode unless they are small and already shaped for reuse. Simple Mode needs a different information architecture.

### Performance Rules

- Keep preset data static and hoisted.
- Use direct imports instead of broad barrel files.
- Split expensive derived state by step.
- Avoid polling every endpoint on every step. Poll only active or relevant jobs.
- Use `startTransition` for non-urgent artifact list and polling updates where it improves responsiveness.
- Version and minimize localStorage payloads.

### Expected File Map

Likely new files:

- `app/simple/page.tsx`
- `app/simple/components/SimpleModePageView.tsx`
- `app/simple/components/SimpleStepper.tsx`
- `app/simple/components/ArchitectureStep.tsx`
- `app/simple/components/TokenizerStep.tsx`
- `app/simple/components/TrainingStep.tsx`
- `app/simple/components/InferenceStep.tsx`
- `app/simple/components/SimpleRunMonitor.tsx`
- `app/simple/hooks/useSimpleModeController.ts`
- `app/simple/hooks/useSimpleFlowPersistence.ts`
- `app/simple/hooks/useSimpleModelStep.ts`
- `app/simple/hooks/useSimpleTokenizerStep.ts`
- `app/simple/hooks/useSimpleTrainingStep.ts`
- `app/simple/hooks/useSimpleInferenceStep.ts`
- `app/simple/lib/modelPresets.ts`
- `app/simple/lib/tokenizerDefaults.ts`
- `app/simple/lib/trainingProfiles.ts`
- `app/simple/lib/inferencePresets.ts`
- `app/simple/lib/stepStatus.ts`
- `app/simple/constants.ts`
- `app/simple/types.ts`
- `app/shared/components/AppTopNav.tsx`
- `app/shared/hooks/useUiMode.ts`

Likely changed files:

- `app/home/components/WorkspaceHomePageView.tsx`
- `app/studio/components/StudioPageView.tsx`
- `app/tokenizer/components/TokenizerPageContent.tsx`
- `app/training/components/TrainingPageContent.tsx`
- `app/inference/components/InferencePageView.tsx`
- `app/styles/*.css` or a new route-specific simple CSS file imported by the route

Potentially deleted files after migration:

- `app/home/components/HomeNavigation.tsx`
- `app/studio/components/page/StudioTopNav.tsx`
- `app/tokenizer/components/TokenizerStudioNav.tsx`
- `app/training/components/TrainingStudioNav.tsx`
- `app/inference/components/InferenceTopNav.tsx`

## Implementation Checklist

[ ] 1. Confirm and preserve the current expert baseline.
Record the current expert routes, nav behavior, mode-independent theme behavior, localStorage keys, and core API flows before changing navigation.

[ ] 2. Add this plan to the working checklist.
Keep this file updated during implementation. Do not mark later tasks `[*]` unless their acceptance checks pass.

[ ] 3. Create `useUiMode`.
Implement storage, same-tab custom event sync, storage-event sync, SSR-safe defaulting, and a setter that accepts either a value or updater function.

[ ] 4. Create shared `AppTopNav`.
Support active route, simple/expert mode switch, theme toggle, route labels, accessible names, and mobile-safe layout.

[ ] 5. Replace duplicated nav components.
Update home, studio, tokenizer, training, and inference pages to use `AppTopNav`. Remove old nav components only after verifying no imports remain.

[ ] 6. Add `/simple` route shell.
Create the route-local folders, `page.tsx`, `SimpleModePageView`, controller hook placeholder, and route CSS imports that fit the existing app style system.

[ ] 7. Define Simple Mode state types and persistence.
Create versioned state parsing, migration-safe defaults, artifact ID persistence, and deletion recovery behavior.

[ ] 8. Define model preset registry.
Create typed preset definitions, repeated block builders, validation helper assertions, display metadata, and tests for every preset's structural constraints.

[ ] 9. Implement backend-backed preset analysis.
Use existing model validation and analysis APIs to show parameter estimates and errors. Do not rely on stale hard-coded parameter counts.

[ ] 10. Build the stepper.
Implement ready, blocked, running, failed, and completed states. The stepper must derive status from real artifacts, not only local UI state.

[ ] 11. Implement Architecture step.
Add template cards, model naming, target vocab/context controls where appropriate, create/update project calls, validation messages, and summary collapse after success.

[ ] 12. Add model/tokenizer vocabulary synchronization.
After tokenizer completion, compare tokenizer vocab to saved model `vocab_size`. If they differ, update the project before enabling model training.

[ ] 13. Implement Simple dataset picker.
Support starter dataset, uploaded local files, and streaming template. Reuse existing upload/stat helpers and make duplicate file handling visible.

[ ] 14. Implement Tokenizer step config generation.
Build BPE byte-level config from defaults plus selected model target vocab. Keep advanced tokenizer fields hidden but available in an `Advanced details` disclosure for inspection.

[ ] 15. Implement Tokenizer validation and training.
Reuse existing validation and job creation APIs. Poll active tokenizer status and block Step 3 until the job is completed.

[ ] 16. Implement Training profile mapping.
Map `Quick check`, `Balanced`, and `Longer run` to training config changes. Base them on backend templates and preflight recommendations, not arbitrary fixed values.

[ ] 17. Implement Simple preflight orchestration.
Run preflight after model, tokenizer, dataset, training profile, or execution target changes. Summarize only the important blockers and warnings in Simple Mode.

[ ] 18. Implement safe automatic fixes.
Apply deterministic preflight fixes when safe, such as scheduler step alignment or model vocab sync. Log the fix in the UI status so the user knows what changed.

[ ] 19. Implement Training step launch.
Start training only when preflight is valid and execution target is confirmed. Store the returned training job ID and move the flow into running state.

[ ] 20. Implement Simple training monitor.
Show stage, progress, elapsed/ETA when available, loss, learning rate, tokens/sec, sample count, checkpoint count, and the latest error if failed.

[ ] 21. Implement Inference step.
Auto-select the completed training run and latest checkpoint. Provide prompt, length preset, creativity preset, generate action, streaming output, and expert link.

[ ] 22. Add expert escape hatches.
From each Simple Mode step, add a small expert link to the corresponding route with the relevant artifact selected where possible.

[ ] 23. Add home integration.
When Simple Mode is active, home should emphasize the guided flow and workspace status. The existing asset manager should remain usable.

[ ] 24. Handle empty workspace and recovery states.
Cover no backend, no projects, no tokenizer jobs, deleted project, deleted tokenizer, failed tokenizer, failed training, no checkpoint, and generation failure.

[ ] 25. Add focused unit tests.
Cover mode storage, Simple Mode state parsing, preset generation, vocabulary sync, training profile mapping, inference preset mapping, and step readiness derivation.

[ ] 26. Add route-level regression checks.
Verify `/`, `/simple`, `/studio`, `/tokenizer`, `/training`, and `/inference` render, preserve mode/theme, and keep expert route behavior intact.

[ ] 27. Run frontend quality commands.
From `apps/llm-studio/web`, run `npm run lint`, `npm run typecheck`, `npm run test:regression`, and `npm run build`.

[ ] 28. Run backend/API compatibility checks if backend contracts changed.
If any API changes were required, run the relevant `apps/llm-studio/api` tests and update API models/types together.

[ ] 29. Perform browser QA in both modes.
Check desktop and mobile viewports. Verify the nav switch, the complete Simple Mode flow with a tiny local dataset, and all expert pages after switching back.

[ ] 30. Final cleanup.
Remove dead nav files, unused imports, stale CSS selectors, temporary debug UI, and any duplicated helper code introduced during the implementation.

## Acceptance Criteria

Simple Mode is complete only when all of the following are true:

- The navbar has a clear Simple/Expert switch on every major route.
- Switching to Simple Mode takes the user to a guided `/simple` flow.
- Switching back restores normal expert navigation.
- A beginner can create a model architecture from a template without using the visual block editor.
- A beginner can train a tokenizer without choosing tokenizer internals.
- A beginner can train a model without choosing optimizer internals, scheduler internals, micro-batch size, or raw JSON.
- A beginner can run inference without choosing top-k, temperature, seed, or repetition penalty numerically.
- Model vocab size and tokenizer vocab size cannot silently diverge before model training.
- Training settings come from backend templates and recommendations wherever possible.
- Paid cloud execution is never launched without explicit user confirmation.
- Existing expert pages still work.
- Existing saved projects, tokenizer jobs, training jobs, and inference artifacts remain compatible.
- Lint, typecheck, regression tests, build, and browser QA pass.

## Out Of Scope For The First Pass

- Exact LLaMA or Mistral architecture parity if the backend still lacks gated/SwiGLU MLP support.
- Full natural-language tutoring or course content.
- Chat-style instruction tuning.
- Multi-user accounts or cloud workspace sync.
- Automatic paid GPU provisioning without user confirmation.
- Replacing Expert Mode.

## Risks And Mitigations

Risk: Simple Mode becomes another expert page with fewer fields.
Mitigation: Keep one active step expanded, hide raw config by default, and make every primary action flow into the next step.

Risk: Named templates overclaim architecture compatibility.
Mitigation: Use `style`, `family-inspired`, or `app-native` labels unless the runtime exactly matches the published architecture.

Risk: Training recommendations are stale or unavailable.
Mitigation: Fall back to conservative templates, block launch on invalid preflight, and state clearly when recommendations are unavailable.

Risk: The nav refactor changes expert behavior.
Mitigation: Replace navs first with parity checks before building Simple Mode.

Risk: LocalStorage state points at deleted artifacts.
Mitigation: Revalidate artifacts on load and mark dependent steps blocked with recovery actions.

Risk: Simple Mode and Expert Mode drift apart.
Mitigation: Simple Mode should use the same API clients and produce the same workspace artifacts as Expert Mode.
