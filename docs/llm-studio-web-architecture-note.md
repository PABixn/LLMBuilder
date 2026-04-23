# LLM Studio Web Refactor Conventions

## Goal

Keep frontend behavior stable while making route code easier to navigate, safer to change, and cheaper to review.

## Route-Local Structure

Every major route should prefer this shape:

- `page.tsx`: route assembly only
- `components/`: presentational sections and small route-owned controls
- `hooks/`: focused async orchestration, persistence, and interaction state
- `lib/`: pure parsing, formatting, normalization, and transformation helpers
- `constants.ts`: route-owned constants and storage keys
- `types.ts`: route-owned types that are too noisy to keep inline

The root route uses `app/home/` as its route-local folder because `/` does not have a segment directory of its own.

## File Boundaries

- Route `page.tsx`: target under 200 lines, hard cap 300
- Presentational components: target under 200 lines, hard cap 300
- Hooks: target under 200 lines, hard cap 250
- Utility files: split once they stop feeling single-purpose

## Promotion Rules

- Keep code route-local unless a real second caller exists.
- Shared code must stay narrow and specific.
- Do not create generic shells or “universal” abstractions without repeated use.

## Hook Rules

- Move polling, localStorage syncing, uploads, downloads, query-param hydration, and async effects into hooks.
- Split hooks by responsibility, not by arbitrary line count.
- Prefer controller hooks that compose smaller hooks over route files that own every effect directly.
- Keep unrelated effect dependencies separated.

## Component Rules

- No inline component definitions inside page components.
- Components should receive already-shaped data whenever possible.
- Keep expensive charts and other expensive surfaces isolated from unrelated rerenders.
- Preserve DOM structure and CSS hooks unless an equivalent output is verified.

## Utility Rules

- Parsing, validation, serialization, formatting, and normalization belong in `lib/`.
- Storage keys, hash maps, and route constants belong in `constants.ts`.
- Repeated file/path formatting and numeric input helpers may be shared if at least two routes use them.

## Shared Primitive Rules

Allowed shared primitives:

- narrow number input controls
- narrow toast helpers and toast view components
- status badges
- simple card shells
- browser storage helpers
- file/path display helpers
- cross-route display formatters with multiple callers

Disallowed shared primitives:

- generic “studio panel” wrappers with unclear ownership
- route-agnostic config editors
- shared hooks that mix unrelated route behavior

## Refactor Guardrails

- No route changes
- No copy changes unless correcting an accidental regression from the refactor itself
- No API contract changes
- No localStorage key changes
- No query param changes
- No behavior drift in uploads, downloads, polling, forms, or chart output
- No reflexive `useMemo` or `useCallback`
- No new monoliths hidden in `components/` or `hooks/`
