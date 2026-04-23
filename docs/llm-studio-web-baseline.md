# LLM Studio Web Baseline

Date: 2026-04-23

## Command Baseline

Commands were run from `apps/llm-studio/web`.

### `npm run lint`

Current result: fails before linting application code.

Observed output:

```text
> llm-studio-web@0.1.0 lint
> next lint

Invalid project directory provided, no such directory: /Users/pabi/Documents/GitHub/LLMBuilder/apps/llm-studio/web/lint
```

Interpretation:

- The project is on Next.js 16.1.6.
- The current `lint` script still points at `next lint`.
- In the current toolchain, that script does not execute a usable lint pass.

### `npm run build`

Current result: passes.

Observed summary:

```text
▲ Next.js 16.1.6 (Turbopack)
✓ Compiled successfully
✓ Generating static pages

Route (app)
┌ ○ /
├ ○ /_not-found
├ ○ /inference
├ ○ /studio
├ ○ /tokenizer
└ ○ /training
```

### `npm run typecheck`

Current result before build artifacts exist: fails.

Observed output:

```text
error TS6053: File '/Users/pabi/Documents/GitHub/LLMBuilder/apps/llm-studio/web/.next/types/validator.ts' not found.
```

Current result after `npm run build`: passes.

Interpretation:

- `tsconfig.json` includes `.next/types/**/*.ts`.
- Fresh `typecheck` depends on generated Next artifacts.
- This is part of the baseline and will be cleaned up during the tooling pass.

## Route Inventory

- `/` -> `app/page.tsx` with workspace summary, navigation, theme toggle, and asset manager.
- `/inference` -> `app/inference/page.tsx` with completed-run picker, checkpoint picker, config controls, and completion results.
- `/tokenizer` -> `app/tokenizer/page.tsx` with persisted config forms, dataset management, local uploads, validation, preview, artifact actions, and job polling.
- `/training` -> `app/training/page.tsx` with model/tokenizer selection, dataset editor, training config, preflight, active-run monitoring, charts, logs, checkpoints, samples, and recent runs.
- `/studio` -> `app/studio/page.tsx` plus large builder support files under `app/studio/components` and `app/studio/hooks`.

## QA Checklist

Use this checklist before and after each route refactor.

### `/`

- Page loads without errors.
- Navigation links target the same routes.
- Theme toggle still flips between stored themes.
- Workspace inventory loads, refreshes, and error state still renders.
- Asset manager actions still work.

### `/inference`

- Page loads with existing defaults.
- Completed training runs still populate the picker.
- Job search still filters on the same fields.
- Checkpoint selection still loads from the selected run.
- Query and selection behavior remain stable.
- Generate action still uses the selected run, checkpoint, and settings.
- Result and error states still match the original flow.

### `/tokenizer`

- Existing localStorage state still hydrates.
- `?job=` deep links still load and repopulate forms.
- Config validation still uses the same payloads and errors.
- Local file upload and drag-and-drop still work.
- Streaming dataset editing, filter editing, and weight normalization still work.
- Preview generation still works and still token-highlights the same way.
- Active job polling and recent jobs still update correctly.
- Artifact download still uses the same file naming and URL behavior.

### `/training`

- Existing localStorage state still hydrates.
- `?project=`, `?tokenizerJob=`, and `?run=` deep links still work.
- Model picker and tokenizer picker still load and filter correctly.
- Dataset editor still supports local files and streaming datasets.
- Training config editing still writes the same structure.
- Preflight still runs on the same conditions and displays the same issues.
- Launch, stop, active run polling, logs, metrics, samples, checkpoints, and recent runs still work.
- Artifact download links and data preview still behave the same way.

### `/studio`

- Page still loads through the existing controller.
- Builder interactions remain stable.
- Import/export still works.
- Undo/redo shortcuts still work.
- Diagnostics, backend validation, and backend analysis still display the same way.

## Screenshot Baseline

The structural baseline is documented here first. Browser screenshot capture should be refreshed into `docs/llm-studio-web-screenshots/` once local unsandboxed browser capture is available.
