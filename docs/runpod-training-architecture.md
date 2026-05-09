# RunPod Training Architecture

## Runtime Path

1. The Training page collects model, tokenizer, training, dataset, prompt, and RunPod launch settings.
2. The local FastAPI training API validates preflight, persists the job in the local SQLite database, and writes the normal local job directory.
3. The RunPod executor resolves the launch target, creates a RunPod Pod through the RunPod REST API, waits for the configured pod-agent port, and verifies pod-agent protocol compatibility.
4. The executor builds a remote bundle containing configs, resolved preflight data, tokenizer artifact, and local dataset files, then uploads it to the pod-agent.
5. The pod-agent extracts the bundle under the remote job workspace and starts `python -m training.runner`.
6. `training.runner` writes logs, metrics, samples, checkpoints, runtime state, data preview, and `artifact_manifest.json` under the remote outputs directory.
7. The local API polls the pod-agent, syncs outputs back into the local job artifact directory, verifies checkpoint files and final manifest data, then applies the selected cleanup policy.

## Boundaries

- Web page: React components and hooks under `apps/llm-studio/web/app/training`.
- API routes: `apps/llm-studio/api/app/training_runs/routes.py`.
- Local DB and job directory: `training_runs/store.py`, `training_runs/runtime_files.py`, and `training_runs/artifacts.py`.
- RunPod REST client: `training_runs/executors/runpod/client.py`.
- RunPod executor orchestration: `training_runs/executors/runpod/executor.py`.
- Pod-agent server: `apps/llm-studio/remote_agent`.
- Remote trainer: `training/runner.py`.
- Sync back to local artifacts: `training_runs/executors/runpod/sync.py`.

## Developer Notes

Compatibility barrels such as `app.training_models`, `app.training_storage`, `app.training_executors.*`, and `web/lib/trainingApi.ts` remain for older internal callers and tests. New code should import from `training_runs/*` on the API side and `web/lib/training/*` on the frontend side.

Run fake RunPod tests from `apps/llm-studio/api`:

```sh
PYTHONPATH=/path/to/LLMBuilder:/path/to/LLMBuilder/apps/llm-studio/api python -m pytest tests/test_runpod_executor_modules.py tests/test_runpod_training.py
```

For a real smoke test, use a tiny `max_steps` value, prefer `Delete after sync`, validate the key before launch, keep the RunPod console open to the pod ID shown in the UI, and confirm the pod is deleted or stopped when the run reaches a terminal state. Do not use `Keep running` unless you are actively debugging and prepared to stop the pod manually.
