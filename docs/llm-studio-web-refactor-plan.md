# LLM Studio Web Refactor Plan

## Purpose

This plan covers a structural refactor of the `apps/llm-studio/web` frontend so the codebase becomes easier to maintain, safer to extend, and faster to reason about without changing anything for the user.

The refactor is explicitly **non-product** work:
- No UX changes
- No copy changes
- No route changes
- No API contract changes
- No localStorage key changes
- No query param changes
- No behavior changes in polling, uploads, downloads, forms, or charts unless the behavior is proven equivalent

## Current Hotspots

The current frontend is already partially componentized, but the largest route files are still too large to maintain safely:

- `apps/llm-studio/web/app/training/page.tsx`: `4851` lines
- `apps/llm-studio/web/app/tokenizer/page.tsx`: `3795` lines
- `apps/llm-studio/web/app/inference/page.tsx`: `1025` lines
- `apps/llm-studio/web/app/page.tsx`: `207` lines
- `apps/llm-studio/web/app/studio/page.tsx`: `22` lines, but several supporting files are still large

Largest follow-up hotspots already visible in `studio`:

- `apps/llm-studio/web/app/studio/components/builder/BuilderCanvasContent.tsx`: `870` lines
- `apps/llm-studio/web/app/studio/hooks/useStudioDocumentEditor.ts`: `610` lines
- `apps/llm-studio/web/app/studio/hooks/useStudioWorkspaceState.ts`: `597` lines
- `apps/llm-studio/web/app/studio/components/builder/BuilderPrefabEditorPopover.tsx`: `540` lines

## Target Architecture Rules

These rules should guide the entire refactor:

- Route entry files (`page.tsx`) should become orchestration layers, not implementation dumps.
- Route logic should be split into route-local `components`, `hooks`, `lib`, `constants`, and `types`.
- Shared code should be promoted only when at least two routes genuinely use the same abstraction.
- Inline component definitions should be removed from large route entry files.
- Pure formatting, parsing, storage, and transformation logic should live outside render functions.
- Polling, localStorage, query-param syncing, uploads, and async orchestration should move into focused hooks.
- Recharts-heavy and other expensive UI sections should be isolated so their render cost does not leak across the whole page.
- Existing CSS output and visual behavior should remain stable unless a change is intentionally proven equivalent.
- No file should remain “large because it is complicated.” Complexity should be split by responsibility.

Recommended file-size budgets:

- Route `page.tsx`: ideally under `200` lines, hard cap `300`
- Presentational components: ideally under `200` lines, hard cap `300`
- Hooks: ideally under `200` lines, hard cap `250`
- Pure utility files: split by domain once they stop feeling single-purpose

## Execution Checklist

Legend:
- `[ ]` not started
- `[*]` completed

[ ] 1. Freeze a behavior baseline before moving code.
Capture the current behavior of `/`, `/inference`, `/tokenizer`, `/training`, and `/studio` with a written QA checklist, route screenshots, and command baselines from `npm run lint`, `npm run typecheck`, and `npm run build`.

[ ] 2. Define and commit the frontend refactor conventions before touching large pages.
Create a short architecture note for route-local folder structure, naming rules, file-size budgets, component boundaries, hook boundaries, and the rule that `page.tsx` is only allowed to assemble sections and wire route state.

[ ] 3. Add a regression safety net for behavior-preserving work.
Introduce lightweight frontend regression coverage for the core flows that are easiest to break during extraction: route loading, deep links, local persistence, uploads, polling states, artifact downloads, and chart rendering.

[ ] 4. Create consistent route-local folders for every major frontend route.
Normalize `app/page.tsx`, `app/inference`, `app/tokenizer`, `app/training`, and `app/studio` so each route has an obvious place for `components`, `hooks`, `lib`, `constants`, and route-local `types`.

[ ] 5. Extract truly shared primitives first, but keep them narrow.
Move only the clearly repeated building blocks into shared locations: small form controls, toast primitives, status badges, card shells, number-input helpers, storage helpers, file/path formatters, and cross-route display formatters. Do not create generic “god components.”

[ ] 6. Refactor the home page into composable sections without changing its behavior.
Split `apps/llm-studio/web/app/page.tsx` into route sections such as shell, hero/introduction, workspace asset area, and any shared supporting UI so the route file becomes a thin assembler.

[ ] 7. Refactor the inference page by responsibility, not by arbitrary chunks.
Break `apps/llm-studio/web/app/inference/page.tsx` into route-local pieces such as run selection, checkpoint selection, prompt/composer area, inference settings, results/history panels, and async data hooks. Keep existing query-param behavior, loading states, and generation flow identical.

[ ] 8. Extract tokenizer page pure domain logic before touching most JSX.
Move tokenizer parsing, validation, hydration, serialization, weight normalization, dataset/filter utilities, preview token helpers, and storage helpers out of `apps/llm-studio/web/app/tokenizer/page.tsx` into route-local `lib`, `constants`, and `types` files first.

[ ] 9. Split tokenizer state management into focused hooks.
Create tokenizer-specific hooks for persisted form state, dataset management, job polling, preview generation, artifact actions, and notifications so the route no longer owns every state transition directly.

[ ] 10. Split the tokenizer UI into stable section components.
Extract route-local components for tokenizer configuration, dataset configuration, training controls, preview/output panels, active job display, recent jobs, dialogs, and small reusable field groups. Keep DOM structure and CSS behavior stable unless equivalent output is verified.

[ ] 11. Extract training page pure domain logic before moving major UI sections.
Move `apps/llm-studio/web/app/training/page.tsx` helpers into route-local modules for config parsing, dataset building, local file normalization, metric formatting, chart-domain calculation, storage access, workflow targeting, preflight issue formatting, and training-run derived state.

[ ] 12. Split training async orchestration into route-local hooks.
Create focused hooks for configuration persistence, dataset editor state, training run polling, recent run polling, metric/sample/checkpoint/log retrieval, artifact selection, preflight validation, and toast/event management so the route file stops coordinating all async work inline.

[ ] 13. Split the training UI into domain components with explicit ownership.
Extract training UI sections into components such as page shell/navigation, workflow summary, asset selection, dataset editor, model settings, tokenizer settings, training settings, preflight panel, active run monitor, charts, samples, checkpoints, logs, and recent runs. `LearningRateSchedulePlanner.tsx` should also be reviewed and decomposed if it remains oversized after the surrounding page is cleaned up.

[ ] 14. Use the same refactor pass to clean remaining oversized `studio` support files.
`app/studio/page.tsx` is already small, but the builder still contains large files. Break down `BuilderCanvasContent.tsx`, `BuilderPrefabEditorPopover.tsx`, `useStudioDocumentEditor.ts`, and `useStudioWorkspaceState.ts` into smaller builder-specific components and hooks so the same quality bar applies across the whole frontend.

[ ] 15. Rationalize styling without changing the rendered UI.
Audit route-level CSS and shared style files so each component owns the smallest sensible style surface. Remove dead selectors, reduce accidental duplication, and keep route-specific styling close to the route unless it is genuinely shared.

[ ] 16. Run a zero-behavior-change performance pass after structural extraction.
Apply safe improvements only where they are measurable and behavior-neutral: hoist static data, isolate expensive charts, avoid unnecessary re-renders, split hooks with unrelated dependencies, defer non-urgent updates with `startTransition` where already appropriate, use `useDeferredValue` for expensive filtering/search surfaces where behavior stays identical, and reduce bundle leakage from oversized route files.

[ ] 17. Verify route parity after every route-level refactor, not only at the end.
For each route, compare the refactored result against the baseline for visual output, interaction flow, local persistence, polling cadence, API calls, loading states, error states, and downloaded/uploaded artifacts before moving to the next route.

[ ] 18. Finish with a hard cleanup pass.
Remove dead code, delete now-unused helpers, collapse duplicate abstractions that appeared during the extraction, fix naming drift, and ensure no new shared abstraction exists without a real second caller.

[ ] 19. Enforce the final quality bar before considering the refactor done.
The refactor is complete only when `npm run lint`, `npm run typecheck`, and `npm run build` pass, the route checklist passes, no major route page is monolithic anymore, and the resulting code is easier to navigate than the current version.

## Route-Specific Target Shape

These are the intended end states for the largest routes.

### `app/inference`

- `page.tsx` should load route state and assemble the page
- `components/` should contain panels and form sections
- `hooks/` should own async queries, selection state, and generation flow
- `lib/` should own formatting and request-shaping helpers

### `app/tokenizer`

- `page.tsx` should be a thin container
- `components/` should contain config sections, preview UI, job panels, dialogs, and field groups
- `hooks/` should own persistence, preview actions, training jobs, and notifications
- `lib/` should own parsing, normalization, hydration, serialization, and display helpers
- `constants.ts` and route-local `types.ts` should remove low-signal noise from the route entry file

### `app/training`

- `page.tsx` should become a route shell only
- `components/` should contain workflow, editors, panels, charts, and run-monitor subtrees
- `hooks/` should isolate polling, persistence, preflight, dataset editing, and artifact selection
- `lib/` should own metric helpers, config helpers, dataset helpers, storage helpers, and workflow mapping
- `constants.ts` and route-local `types.ts` should keep the route shell readable

### `app/studio`

- Keep the existing route-level shape
- Continue reducing oversized builder files until the builder follows the same maintainability standard as the rest of the app

## Guardrails

Do not compromise the refactor by making these mistakes:

- Do not replace one monolith with a “components” folder full of new monoliths.
- Do not invent broad shared abstractions too early.
- Do not mix structural refactor work with feature work.
- Do not silently change storage keys, API payloads, or query-param names.
- Do not move expensive logic into render paths if it can be derived once in helpers or hooks.
- Do not optimize with `useMemo` or `useCallback` by reflex; only do it where render cost or prop stability actually matters.
- Do not keep chart logic, polling logic, and form-editing logic coupled in the same file.

## Definition Of Done

The refactor should be considered successful only if all of the following are true:

- Every major frontend route is split into intentional components and hooks.
- `training/page.tsx` and `tokenizer/page.tsx` are no longer monolithic.
- `inference/page.tsx` is reduced to route assembly.
- The most oversized `studio` support files are materially reduced.
- Shared helpers are easier to discover and have clear ownership.
- Build, typecheck, lint, and regression verification all pass.
- A user cannot tell that the frontend was refactored, but a developer can.
