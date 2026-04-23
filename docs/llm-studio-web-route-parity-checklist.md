# LLM Studio Web Route Parity Checklist

Use this checklist after each route-level refactor before moving on.

## Shared

- `npm run build`
- `npm run typecheck`
- Route still renders
- No route path changes
- No storage key changes
- No query param changes
- No unexpected console or network error paths introduced

## `/`

- Theme toggle still persists
- Inventory refresh state still renders
- Inventory error state still renders
- Navigation links unchanged
- Workspace asset manager unchanged

## `/inference`

- Completed run picker loads
- Run search works
- Checkpoint list refreshes when run changes
- Checkpoint search works
- Generate request still succeeds
- Result panel still renders completion output
- Error panel still renders request failures

## `/tokenizer`

- Local persisted draft still hydrates
- `job` query param still loads job data
- Validate action still works
- Start training action still works
- Local file upload and drag/drop still works
- Local file stats still fill in
- Streaming dataset and filter editing still works
- Preview still works
- Recent jobs hide/remove behavior still works
- Artifact download still works

## `/training`

- Persisted selections and configs still hydrate
- `project`, `tokenizerJob`, and `run` query params still hydrate
- Project picker works
- Tokenizer picker works
- Dataset editing still works
- Preflight still runs automatically
- Recommendation selection still works
- Launch training still works
- Stop training still works
- Active run metrics/logs/checkpoints/samples still refresh
- Recent runs still refresh

## `/studio`

- Builder loads
- Import/export still works
- Undo/redo shortcuts still work
- Builder drag interactions still work
- Diagnostics still render
- Backend analysis still runs
