# RunPod Pod Training Implementation Plan

## Goal

Implement fully managed LLM training on RunPod Pods from `llm-studio`.

The target user flow is:

1. User opens `llm-studio`.
2. User enters a RunPod API key.
3. User selects a saved model config, tokenizer artifact, dataset, and training settings.
4. `llm-studio` creates and manages the RunPod infrastructure.
5. Training runs to completion on a rented GPU Pod.
6. Metrics, logs, samples, checkpoints, and final artifacts appear in the existing `llm-studio` training UI.
7. `llm-studio` safely syncs outputs locally and stops or deletes the Pod according to the selected cleanup policy.

No manual RunPod console work, SSH, Jupyter, S3 key, or file copy should be required for the standard path.

## Current Repo Baseline

The existing training architecture is already close to the right shape:

- `apps/llm-studio/api/app/training_jobs.py` owns preflight validation, job directory creation, metadata persistence, job polling, stop/delete, metrics, logs, checkpoints, and artifact bundling.
- `TrainingRunManager.create_job()` currently writes job inputs into a local job directory and launches a local subprocess.
- `_spawn_process()` currently runs `python -m training.runner` with explicit file paths.
- `training/runner.py` already accepts job-scoped paths and writes runtime state, metrics, samples, logs, checkpoints, and artifact manifests.
- `apps/llm-studio/api/app/training_storage.py` already persists enough training metadata to extend with RunPod lifecycle fields.
- `apps/llm-studio/web/lib/trainingApi.ts` and `/training` already poll job status, metrics, logs, checkpoints, and samples through the local API.

The right implementation is not a separate RunPod-only training path. It is a new executor behind the existing training job model.

## Key Product Constraint: RunPod API Key Only

RunPod's normal REST API key can create and manage Pods, templates, and network volumes. The RunPod REST API docs state that all API requests use bearer authentication, and expose Pods and network volumes as managed resources:

- RunPod REST API overview: https://docs.runpod.io/api-reference/overview
- Create Pod: https://docs.runpod.io/api-reference/pods/POST/pods
- Find Pod by ID: https://docs.runpod.io/api-reference/pods/GET/pods/podId
- Stop Pod: https://docs.runpod.io/api-reference/pods/POST/pods/podId/stop
- Delete Pod: https://docs.runpod.io/api-reference/pods/DELETE/pods/podId
- Create network volume: https://docs.runpod.io/api-reference/network-volumes/POST/networkvolumes

However, RunPod's S3-compatible API for network volumes requires a separate S3 API key. That violates the "just give it the RunPod API key" requirement:

- RunPod S3-compatible API: https://docs.runpod.io/storage/s3-api

Therefore, the standard implementation must not require RunPod S3 credentials. It should transfer inputs and outputs through a pod-resident training agent over the Pod's exposed HTTP port. Network volumes can still be created and attached by `llm-studio` using only the RunPod API key, but they are workspace/cache storage for the Pod, not the primary local-to-cloud transfer mechanism.

## Architecture

```text
llm-studio web
  -> local FastAPI training API
    -> TrainingRunManager
      -> TrainingExecutor protocol
        -> LocalSubprocessExecutor
        -> RunPodPodExecutor
          -> RunPod REST API
          -> managed GPU Pod
            -> public llm-studio training image
            -> pod HTTP agent
              -> receives job bundle
              -> runs python -m training.runner
              -> serves runtime state, metrics, logs, checkpoints, artifacts
```

### Standard Data Flow

```text
local job dir
  -> bundle inputs as tar/zstd
  -> upload bundle to Pod agent over authenticated HTTP
  -> Pod agent extracts to /workspace/llm-studio/jobs/<job_id>
  -> Pod agent runs training.runner on GPU
  -> local API polls Pod agent
  -> local API incrementally downloads metrics/logs/samples/checkpoints
  -> local API verifies final manifest
  -> local API stops/deletes Pod
```

### Source Of Truth

The local `llm-studio` training database remains the source of truth for user-visible jobs.

The Pod is a remote executor. It can be recreated, stopped, or deleted without changing the local job identity. Remote files are not considered durable until synced back into the local job directory.

## Implementation Phases

## Phase 1: Define The Executor Boundary

Goal: make local and RunPod execution interchangeable.

### Backend Changes

Add a new module:

```text
apps/llm-studio/api/app/training_executors/
  __init__.py
  base.py
  local.py
  runpod_client.py
  runpod_pod.py
  remote_sync.py
```

Define the executor protocol in `base.py`:

```python
class TrainingExecutor(Protocol):
    kind: str

    def submit(self, job: StoredTrainingJob, bundle: TrainingJobBundle) -> ExecutionHandle:
        ...

    def refresh(self, job: StoredTrainingJob) -> ExecutionSnapshot:
        ...

    def stop(self, job: StoredTrainingJob) -> ExecutionSnapshot:
        ...

    def cleanup(self, job: StoredTrainingJob, policy: CleanupPolicy) -> None:
        ...
```

Use these concepts:

- `TrainingJobBundle`: paths to `model_config.json`, `tokenizer_artifact.json`, `training_config.json`, `dataloader_config.json`, `resolved_preflight.json`, plus manifest metadata.
- `ExecutionHandle`: executor kind, remote IDs, agent URL, start timestamps, process ID if local.
- `ExecutionSnapshot`: status, state, stage, progress, remote runtime state, metrics/log availability, error, cost fields.
- `CleanupPolicy`: `delete_pod`, `stop_pod`, `keep_pod`, `delete_volume`, `keep_volume`.

### Refactor Current Local Execution

Move the current `_spawn_process()` implementation from `TrainingRunManager` into `LocalSubprocessExecutor`.

Keep behavior identical:

- same `python -m training.runner` command
- same output directory layout
- same local polling behavior
- same cancel behavior

This phase should produce zero UI behavior changes.

### Acceptance Criteria

- Local training still works.
- Existing `/api/v1/training/jobs` behavior is unchanged.
- Existing tests pass.
- `TrainingRunManager` no longer knows subprocess details directly.

## Phase 2: Extend Training Job Persistence

Goal: persist enough remote lifecycle state to survive API restarts.

Add fields to `TrainingRunRow` and `StoredTrainingJob`:

```text
executor_kind                 local | runpod_pod
executor_status               provisioning | booting | uploading | running | syncing | cleaning_up | completed | failed | cancelled
runpod_pod_id                 nullable string
runpod_pod_name               nullable string
runpod_network_volume_id      nullable string
runpod_data_center_id         nullable string
runpod_gpu_type_id            nullable string
runpod_gpu_count              integer
runpod_cloud_type             SECURE | COMMUNITY
runpod_interruptible          boolean
runpod_cost_per_hr            nullable float
runpod_public_ip              nullable string
runpod_port_mappings          json
runpod_agent_base_url         nullable string
runpod_agent_token_hash       nullable string
runpod_last_heartbeat_at      nullable datetime
runpod_last_sync_at           nullable datetime
runpod_cleanup_policy         json
remote_workspace_path         nullable string
remote_error                  nullable text
```

Do not store the raw RunPod API key in the job row.

### Migration Strategy

This project currently creates tables directly with SQLAlchemy metadata. Add a lightweight schema migration helper before introducing these fields:

1. On startup, inspect `llm_training_jobs`.
2. Add missing columns with SQLite-safe `ALTER TABLE`.
3. Keep `Base.metadata.create_all()` for fresh installs.
4. Add a regression test that starts from an old SQLite schema and verifies startup migration.

### Acceptance Criteria

- Old local training databases load.
- New RunPod fields are present after startup.
- Incomplete RunPod jobs can be reattached after API restart.

## Phase 3: Add RunPod Configuration

Goal: let the user provide only a RunPod API key and sensible defaults.

### Backend Settings

Add environment settings:

```text
LLM_STUDIO_RUNPOD_API_KEY
LLM_STUDIO_RUNPOD_DEFAULT_GPU_TYPE=NVIDIA GeForce RTX 4090
LLM_STUDIO_RUNPOD_DEFAULT_GPU_COUNT=1
LLM_STUDIO_RUNPOD_DEFAULT_CLOUD_TYPE=SECURE
LLM_STUDIO_RUNPOD_DEFAULT_DATA_CENTER_ID=
LLM_STUDIO_RUNPOD_DEFAULT_VOLUME_SIZE_GB=100
LLM_STUDIO_RUNPOD_CONTAINER_DISK_GB=50
LLM_STUDIO_RUNPOD_VOLUME_MOUNT_PATH=/workspace
LLM_STUDIO_RUNPOD_TRAINING_IMAGE=ghcr.io/<owner>/llm-builder-training:<version>
LLM_STUDIO_RUNPOD_AGENT_PORT=8021
LLM_STUDIO_RUNPOD_POD_TTL_MINUTES=0
LLM_STUDIO_RUNPOD_AUTO_DELETE_POD=1
LLM_STUDIO_RUNPOD_AUTO_DELETE_VOLUME=0
```

### UI Settings

Add a small RunPod settings panel to the training page or app settings:

- API key input.
- Validate key button.
- GPU type selector.
- Secure Cloud vs Community Cloud selector.
- Datacenter selector.
- Network volume size.
- Interruptible toggle.
- Cleanup policy.
- Estimated hourly cost once a Pod is selected.

The API key can be provided in either:

- `LLM_STUDIO_RUNPOD_API_KEY` environment variable.
- UI session input stored in memory for the current API process.
- Optional later: OS keychain integration.

For v1, prefer environment variable plus temporary in-memory UI override. Do not write the key to plaintext JSON by default.

### RunPod Capability Endpoints

Add local API endpoints:

```text
GET  /api/v1/training/providers/runpod/status
POST /api/v1/training/providers/runpod/validate-key
GET  /api/v1/training/providers/runpod/pods
GET  /api/v1/training/providers/runpod/network-volumes
GET  /api/v1/training/providers/runpod/defaults
```

If GPU availability endpoints are not convenient in the public REST API, use the published OpenAPI schema to generate or validate the client. RunPod exposes its OpenAPI document at `https://rest.runpod.io/v1/openapi.json`.

### Acceptance Criteria

- A user can paste a RunPod API key and see a valid/invalid result.
- The API never logs the key.
- The UI can choose `local` or `runpod_pod` as training target.
- The RunPod target is disabled until key validation succeeds.

## Phase 4: Build The Training Pod Image

Goal: provide a public image so the user does not need to build or push Docker images.

Add:

```text
docker/training/Dockerfile
docker/training/entrypoint.sh
apps/llm-studio/remote_agent/
  __init__.py
  app.py
  auth.py
  bundle.py
  runner.py
  sync_manifest.py
```

### Image Responsibilities

The image must include:

- Python runtime compatible with the trainer.
- PyTorch CUDA build.
- Repo code needed by `model`, `tokenizer`, and `training`.
- Python dependencies from the API/training runtime.
- `zstd` or Python zstandard support for bundles.
- A FastAPI or Starlette pod agent.
- A startup command that runs the pod agent.

The image must not include:

- RunPod API key.
- User datasets.
- User model/tokenizer artifacts.
- Local `llm-studio` database.

### Pod Agent API

Expose the following endpoints on the Pod:

```text
GET  /health
GET  /v1/system
POST /v1/jobs/{job_id}/bundle
POST /v1/jobs/{job_id}/start
GET  /v1/jobs/{job_id}
GET  /v1/jobs/{job_id}/runtime-state
GET  /v1/jobs/{job_id}/metrics?offset=<bytes>
GET  /v1/jobs/{job_id}/samples?offset=<bytes>
GET  /v1/jobs/{job_id}/logs/stdout?offset=<bytes>
GET  /v1/jobs/{job_id}/logs/stderr?offset=<bytes>
GET  /v1/jobs/{job_id}/checkpoints
GET  /v1/jobs/{job_id}/files?path=<relative_path>
POST /v1/jobs/{job_id}/cancel
POST /v1/jobs/{job_id}/shutdown
```

Use a short-lived per-job bearer token generated locally and injected into the Pod as an environment variable:

```text
LLM_STUDIO_REMOTE_AGENT_TOKEN
LLM_STUDIO_REMOTE_JOB_ID
LLM_STUDIO_REMOTE_WORKSPACE=/workspace/llm-studio
```

Every pod-agent request must require:

```text
Authorization: Bearer <agent_token>
X-LLM-Studio-Job-Id: <job_id>
```

### Agent Runtime Behavior

On `/bundle`:

1. Validate job ID and token.
2. Store upload under `/workspace/llm-studio/incoming/<job_id>.tar.zst`.
3. Extract to `/workspace/llm-studio/jobs/<job_id>/inputs`.
4. Verify manifest checksums.
5. Prepare output directory `/workspace/llm-studio/jobs/<job_id>/outputs`.

On `/start`:

1. Build the `training.runner` command with explicit paths.
2. Start it as a subprocess.
3. Redirect stdout/stderr to output logs.
4. Record process ID.
5. Return immediately.

On polling endpoints:

1. Read files produced by `training.runner`.
2. Return byte-range-friendly tails.
3. Return checkpoint manifests with file sizes and hashes.

On `/cancel`:

1. Send SIGTERM to training process group.
2. Wait a short grace period.
3. Send SIGKILL if needed.
4. Mark runtime state cancelled.

### Build And Publish

Add CI to build and publish the image to a public registry:

```text
ghcr.io/<owner>/llm-builder-training:<git_sha>
ghcr.io/<owner>/llm-builder-training:latest
```

End users should not need registry credentials for the standard image.

### Acceptance Criteria

- Image starts on RunPod PyTorch-compatible GPU environment.
- `/health` responds.
- A local integration test can upload a tiny bundle and run a 1 to 2 step CPU/GPU smoke training job.
- The same `training.runner` code path is used locally and remotely.

## Phase 5: Implement RunPod REST Client

Goal: manage Pod and network volume lifecycle through the normal RunPod API key.

Add `RunPodClient` with:

```python
class RunPodClient:
    def validate_key(self) -> RunPodAccountSummary: ...
    def list_pods(self, filters: PodFilters | None = None) -> list[RunPodPod]: ...
    def get_pod(self, pod_id: str) -> RunPodPod: ...
    def create_pod(self, request: CreatePodRequest) -> RunPodPod: ...
    def stop_pod(self, pod_id: str) -> None: ...
    def start_pod(self, pod_id: str) -> None: ...
    def delete_pod(self, pod_id: str) -> None: ...
    def list_network_volumes(self) -> list[RunPodNetworkVolume]: ...
    def create_network_volume(self, request: CreateNetworkVolumeRequest) -> RunPodNetworkVolume: ...
```

Use:

```text
Authorization: Bearer <RUNPOD_API_KEY>
Content-Type: application/json
```

### Pod Creation Request

Use the RunPod create Pod API fields:

```json
{
  "name": "llm-studio-<job_id>",
  "cloudType": "SECURE",
  "computeType": "GPU",
  "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
  "gpuCount": 1,
  "gpuTypePriority": "availability",
  "dataCenterIds": ["US-KS-2"],
  "dataCenterPriority": "availability",
  "containerDiskInGb": 50,
  "volumeInGb": 100,
  "networkVolumeId": "<optional-managed-volume-id>",
  "volumeMountPath": "/workspace",
  "ports": ["8021/http"],
  "supportPublicIp": true,
  "imageName": "ghcr.io/<owner>/llm-builder-training:<version>",
  "dockerEntrypoint": [],
  "dockerStartCmd": [],
  "env": {
    "LLM_STUDIO_REMOTE_AGENT_TOKEN": "<short-lived-token>",
    "LLM_STUDIO_REMOTE_JOB_ID": "<job_id>",
    "LLM_STUDIO_REMOTE_WORKSPACE": "/workspace/llm-studio",
    "PYTHONUNBUFFERED": "1"
  }
}
```

Confirm the exact image field name against the generated OpenAPI client. The RunPod docs show `image` in responses and Pod creation supports Docker image/template fields; the implementation must be tested against the current OpenAPI schema before release.

### Network Volume Policy

Default policy:

- Use Secure Cloud.
- Create one managed network volume per datacenter and reuse it for jobs.
- Name format: `llm-studio-<machine-user-or-install-id>-<data_center_id>`.
- Default size: 100 GB.
- Do not delete the volume after every run unless the user explicitly chooses `delete_volume`.

Why:

- Network volumes are persistent and mount at `/workspace` for Pods.
- They can cache datasets and dependencies between jobs.
- They reduce loss risk if the Pod restarts.
- They are creatable with only the RunPod REST API key.

Important limitation:

- The local API cannot read the volume after Pod deletion unless it starts another Pod or the user provides separate S3 API credentials.
- Therefore, final artifact sync must finish before deleting the last Pod that can access the volume.

### Acceptance Criteria

- The API can create a network volume if configured.
- The API can create a Pod with the training image.
- The API can poll until `publicIp` and `portMappings` are available.
- The API can stop and delete the Pod.
- Failures are reported with actionable messages.

## Phase 6: Implement RunPodPodExecutor

Goal: automate the full remote training lifecycle.

### Submit Lifecycle

`RunPodPodExecutor.submit()` should:

1. Ensure RunPod API key is available.
2. Create the local job directory exactly as local training does.
3. Generate a per-job agent token.
4. Create or select a network volume if configured.
5. Create a RunPod Pod.
6. Persist `runpod_pod_id`, selected GPU, datacenter, cleanup policy, and agent token hash.
7. Poll `GET /pods/{podId}` until Pod is running and `portMappings` are available.
8. Build `agent_base_url` from RunPod pod IP/proxy details.
9. Poll `GET /health` on the pod agent.
10. Build and upload the input bundle.
11. Call `POST /v1/jobs/{job_id}/start`.
12. Mark local job as running.

### Bundle Format

Create:

```text
bundle.tar.zst
  manifest.json
  inputs/model_config.json
  inputs/tokenizer_artifact.json
  inputs/training_config.json
  inputs/dataloader_config.json
  inputs/resolved_preflight.json
  inputs/local_files/<sanitized files if needed>
```

`manifest.json`:

```json
{
  "format": "llm-studio-training-bundle-v1",
  "job_id": "<job_id>",
  "created_at": "2026-04-24T00:00:00Z",
  "files": [
    {
      "path": "inputs/model_config.json",
      "sha256": "...",
      "size_bytes": 1234
    }
  ],
  "runner": {
    "module": "training.runner",
    "args": {
      "model_config_path": "inputs/model_config.json",
      "tokenizer_path": "inputs/tokenizer_artifact.json",
      "training_config_path": "inputs/training_config.json",
      "dataloader_config_path": "inputs/dataloader_config.json",
      "output_dir": "outputs"
    }
  }
}
```

### Local Dataset Handling

The remote Pod cannot access local file paths. Convert local dataset references before bundling.

For every dataloader dataset entry using local files:

1. Resolve local paths during preflight.
2. Copy those files into `inputs/local_files/<dataset_id>/`.
3. Rewrite `data_files` in the bundled `dataloader_config.json` to remote relative paths.
4. Store original local path metadata in `resolved_preflight.json`.

For Hugging Face or other remote datasets:

1. Keep the remote dataset spec.
2. Set `HF_HOME` and `HF_DATASETS_CACHE` inside the Pod to `/workspace/llm-studio/cache/huggingface`.
3. Add optional Hugging Face token support later if private datasets are required.

### Refresh Lifecycle

`RunPodPodExecutor.refresh()` should:

1. Fetch RunPod Pod state.
2. Fetch pod-agent runtime state.
3. Incrementally sync small files:
   - `runtime_state.json`
   - `metadata.json`
   - `artifact_manifest.json`
   - `training_data_preview.json`
4. Incrementally sync append-only files:
   - `stats.jsonl`
   - `samples.jsonl`
   - `stdout.log`
   - `stderr.log`
5. Detect new checkpoints and download them with checksum verification.
6. Update local `StoredTrainingJob` fields.
7. If remote training completed, run final artifact sync.
8. If final sync succeeds, apply cleanup policy.

### Checkpoint Sync

Checkpoint files can be large. Use resumable downloads:

- Agent endpoint returns file size, mtime, and sha256 if precomputed.
- Local sync writes to `*.partial`.
- Local sync resumes from byte offset if possible.
- Local sync renames to final path only after checksum passes.
- Local sync never deletes an existing verified checkpoint unless the user deletes the job.

### Completion Lifecycle

When remote state is completed:

1. Download all remaining checkpoints.
2. Download `artifact_manifest.json`.
3. Build or download a final artifact bundle.
4. Verify:
   - final checkpoint exists
   - manifest references valid files
   - local `output_size_bytes` is updated
5. Mark local job `completed`.
6. Stop/delete Pod according to policy.
7. Keep local artifacts available to inference and downloads.

### Cancellation Lifecycle

When user presses stop:

1. Mark local job `cancelling`.
2. Call pod-agent `/cancel`.
3. Poll a short grace period for remote cancellation.
4. Sync latest logs/state/checkpoints.
5. Stop Pod.
6. Delete Pod if policy says so.
7. Mark local job `cancelled`.

If pod-agent is unreachable:

1. Call RunPod stop.
2. Sync what is available if the Pod comes back.
3. Mark job cancelled with a clear warning that final outputs may be incomplete.

### Failure Lifecycle

Handle:

- API key invalid.
- RunPod insufficient funds.
- GPU unavailable.
- Pod stuck provisioning.
- Pod boots but agent never responds.
- Upload interrupted.
- Training process crashes.
- Pod is preempted or stopped.
- Local API restarts mid-run.
- Final artifact sync fails.

Every failure should have:

- local job status
- remote pod ID if one exists
- next action hint
- cleanup action availability

### Acceptance Criteria

- A full training run can be launched from the web UI with target `RunPod Pod`.
- No RunPod console interaction is required.
- Metrics/logs/checkpoints appear in the existing training page.
- Final trained checkpoint is available locally after Pod cleanup.
- Cancel stops remote spend.
- API restart can reattach to an active Pod and continue polling/syncing.

## Phase 7: Integrate With Existing API Routes

Goal: preserve the existing frontend contract.

Existing routes should continue to work:

```text
POST /api/v1/training/jobs
GET  /api/v1/training/jobs
GET  /api/v1/training/jobs/{job_id}
GET  /api/v1/training/jobs/{job_id}/metrics
GET  /api/v1/training/jobs/{job_id}/samples
GET  /api/v1/training/jobs/{job_id}/logs
GET  /api/v1/training/jobs/{job_id}/checkpoints
POST /api/v1/training/jobs/{job_id}/stop
GET  /api/v1/training/jobs/{job_id}/artifact
```

Extend `CreateTrainingJobRequest`:

```json
{
  "project_id": "...",
  "tokenizer_job_id": "...",
  "training_config": {},
  "dataloader_config": {},
  "name": "run name",
  "execution_target": {
    "kind": "runpod_pod",
    "gpu_type_id": "NVIDIA GeForce RTX 4090",
    "gpu_count": 1,
    "cloud_type": "SECURE",
    "data_center_id": "US-KS-2",
    "interruptible": false,
    "cleanup_policy": {
      "pod": "delete_after_sync",
      "network_volume": "keep"
    }
  }
}
```

Default to local execution unless the user explicitly chooses RunPod.

### API Behavior

`TrainingRunManager.create_job()` should:

1. Run the same preflight.
2. Create the same local job record.
3. Build a `TrainingJobBundle`.
4. Select executor based on `execution_target.kind`.
5. Submit to executor.
6. Return the normal `TrainingJobResponse`.

`TrainingRunManager._refresh_job()` should:

1. Dispatch to the job executor.
2. Merge executor snapshot into local job state.
3. Preserve existing status derivation for local jobs.

### Acceptance Criteria

- Existing frontend needs minimal changes.
- RunPod jobs appear in recent runs.
- Asset manager can list completed RunPod training runs exactly like local runs.
- Inference can load completed RunPod checkpoints because artifacts are local after sync.

## Phase 8: Frontend UX

Goal: make RunPod training understandable and safe.

### Training Page Changes

Add an execution target panel:

- Local machine.
- RunPod Pod.

When RunPod Pod is selected, show:

- API key state.
- GPU type.
- GPU count.
- Cloud type.
- Datacenter.
- Network volume policy.
- Interruptible warning.
- Cleanup policy.
- Estimated hourly cost if available.

### Run Monitor Changes

Add remote infrastructure status:

- Pod lifecycle: provisioning, booting, agent ready, uploading, training, syncing, cleaning up.
- Pod ID.
- GPU.
- Datacenter.
- Cost per hour.
- Last heartbeat.
- Last sync time.
- Cleanup policy.

Keep existing training status prominent:

- step / max steps
- loss
- grad norm
- LR
- tokens/sec
- checkpoints
- samples
- logs

### Safety Prompts

Require explicit confirmation when:

- Starting an interruptible Pod.
- Keeping a Pod alive after training.
- Deleting a network volume.
- Retrying a job that may create a second billable Pod.

### Acceptance Criteria

- User can start RunPod training without reading JSON.
- User can see whether spend is active.
- User can stop spend from the UI.
- User can recover from failed sync or cleanup.

## Phase 9: Security Model

Goal: protect user secrets and prevent public Pod misuse.

### API Key Handling

- Never send the RunPod API key to the frontend after submission.
- Never send the RunPod API key to the Pod.
- Never write the RunPod API key to job metadata.
- Redact all `Authorization` values in logs.
- Prefer environment variable or in-memory session storage for v1.

### Pod Agent Authentication

- Generate one random token per job.
- Inject only that token into the Pod.
- Store only token hash locally.
- Require bearer auth on every pod-agent endpoint.
- Include job ID in request headers.
- Reject path traversal in file endpoints.
- Serve files only from the job workspace.

### Network Exposure

- Expose only the pod-agent HTTP port.
- Do not expose SSH by default.
- Do not expose Jupyter by default.
- Add optional debug mode later that exposes SSH/Jupyter with clear warnings.

### Artifact Safety

- Validate checksums before using downloaded files.
- Treat pod-agent responses as untrusted.
- Do not execute any downloaded code.
- Only load checkpoints from completed/synced jobs.

### Acceptance Criteria

- No raw RunPod API key appears in database, logs, pod env, or UI responses.
- Pod-agent file API cannot read outside the job directory.
- A different job token cannot access another job.

## Phase 10: Reliability And Recovery

Goal: make cloud failure normal, not catastrophic.

### API Restart Recovery

On API startup:

1. Load jobs where `executor_kind = runpod_pod` and status is `pending` or `running`.
2. For each job with `runpod_pod_id`, call `GET /pods/{podId}`.
3. If Pod exists and agent responds, resume polling and sync.
4. If Pod exists but agent does not respond, mark `remote_reconnect_failed` but allow retry.
5. If Pod does not exist, mark failed unless final artifacts were already synced.

### Sync Retry

Add explicit job actions:

```text
POST /api/v1/training/jobs/{job_id}/remote/resync
POST /api/v1/training/jobs/{job_id}/remote/cleanup
POST /api/v1/training/jobs/{job_id}/remote/reattach
```

### Watchdogs

Implement watchdogs:

- Pod provisioning timeout.
- Agent boot timeout.
- No-heartbeat timeout.
- No-log-progress timeout.
- Cleanup timeout.

On timeout:

- Warn first.
- Keep user-visible stop/delete action.
- Do not silently leave billable Pods running.

### Acceptance Criteria

- Pulling the API process mid-training does not orphan the job from the UI.
- A failed final sync can be retried.
- A cleanup failure leaves a visible Pod ID and delete action.

## Phase 11: Testing Strategy

Goal: prove the system trains fully and cleans up spend.

### Unit Tests

Test:

- executor selection
- bundle manifest generation
- local file dataset rewriting
- RunPod request payload creation
- RunPod response parsing
- pod-agent auth
- path traversal rejection
- checkpoint sync resume
- cleanup policy resolution

### Integration Tests Without RunPod

Use a local fake RunPod API plus local fake pod-agent:

- create pod returns fake ID
- pod transitions through provisioning to running
- port mapping appears
- agent accepts bundle
- agent simulates training state
- agent exposes fake checkpoints
- cleanup is called

### Real RunPod Smoke Test

Add an opt-in test gated by env var:

```text
RUNPOD_API_KEY
RUNPOD_TEST_GPU_TYPE
RUNPOD_TEST_DATA_CENTER
RUNPOD_TEST_MAX_COST_USD
```

Smoke test:

1. Create tiny model config.
2. Create tiny tokenizer.
3. Use tiny local text dataset.
4. Run `max_steps = 2`.
5. Launch RunPod Pod.
6. Wait for completion.
7. Sync final checkpoint.
8. Verify checkpoint can be loaded locally.
9. Delete Pod.
10. Delete test volume if created for the test.

### Regression Tests

Test:

- API restart while Pod is training.
- User cancellation.
- Pod agent crash.
- Pod stopped externally.
- Final checkpoint download interrupted.
- Invalid API key.
- GPU unavailable.

### Acceptance Criteria

- CI passes local/fake tests.
- Manual or scheduled real RunPod smoke proves end-to-end training.
- Real smoke test never runs unless explicitly enabled.

## Phase 12: Documentation

Add user docs:

```text
docs/runpod-training-user-guide.md
docs/runpod-training-troubleshooting.md
docs/runpod-training-security.md
```

User guide must explain:

- How to create a RunPod API key.
- How to paste it into `llm-studio`.
- How to choose GPU and cleanup policy.
- What happens during provisioning, upload, training, sync, and cleanup.
- How to stop a running job.
- How to recover if cleanup fails.

Troubleshooting must include:

- Invalid key.
- No GPU capacity.
- Pod stuck booting.
- Agent unreachable.
- Training failed.
- Checkpoints not synced.
- Pod cleanup failed.
- Unexpected RunPod charges.

Security doc must include:

- API key handling.
- Pod-agent token handling.
- Why S3 API key is not required.
- What data is uploaded to RunPod.
- How to delete remote Pods and volumes.

## Detailed File-Level Plan

### Backend

Create:

```text
apps/llm-studio/api/app/training_executors/base.py
apps/llm-studio/api/app/training_executors/local.py
apps/llm-studio/api/app/training_executors/runpod_client.py
apps/llm-studio/api/app/training_executors/runpod_pod.py
apps/llm-studio/api/app/training_executors/remote_sync.py
apps/llm-studio/api/app/training_bundle.py
apps/llm-studio/api/app/runpod_models.py
apps/llm-studio/api/app/runpod_settings.py
apps/llm-studio/remote_agent/app.py
apps/llm-studio/remote_agent/auth.py
apps/llm-studio/remote_agent/bundle.py
apps/llm-studio/remote_agent/runner.py
```

Modify:

```text
apps/llm-studio/api/app/config.py
apps/llm-studio/api/app/main.py
apps/llm-studio/api/app/training_jobs.py
apps/llm-studio/api/app/training_models.py
apps/llm-studio/api/app/training_storage.py
apps/llm-studio/api/requirements.txt
apps/llm-studio/api/tests/
training/runner.py
```

Expected dependency additions:

```text
httpx
zstandard
```

Use `boto3` only for optional S3-enhanced mode, not the default API-key-only path.

### Frontend

Modify:

```text
apps/llm-studio/web/lib/trainingApi.ts
apps/llm-studio/web/app/training/types.ts
apps/llm-studio/web/app/training/page.tsx
apps/llm-studio/web/app/styles/training.css
```

Potential extracts if the page is too large:

```text
apps/llm-studio/web/app/training/components/ExecutionTargetPanel.tsx
apps/llm-studio/web/app/training/components/RunPodSettingsPanel.tsx
apps/llm-studio/web/app/training/components/RemoteRuntimeStatus.tsx
apps/llm-studio/web/app/training/hooks/useRunPodProvider.ts
```

### Docker/CI

Create:

```text
docker/training/Dockerfile
docker/training/entrypoint.sh
.github/workflows/training-image.yml
```

If this repo does not use GitHub Actions, add equivalent build documentation and keep the image name configurable.

## Rollout Plan

### Milestone 1: Executor Refactor

- Add executor interface.
- Move local subprocess into local executor.
- No RunPod UI yet.

### Milestone 2: Pod Agent Local Test

- Build pod agent.
- Test against local process, not RunPod.
- Confirm bundle upload and `training.runner` invocation.

### Milestone 3: RunPod Provisioning

- Add RunPod REST client.
- Create Pod from public image.
- Wait for health.
- Stop/delete Pod.

### Milestone 4: Remote Training Smoke

- Launch tiny real training job.
- Sync logs/metrics/checkpoints.
- Delete Pod.

### Milestone 5: Full UI

- Add RunPod execution controls.
- Add remote status and cleanup UI.
- Add key validation.

### Milestone 6: Reliability

- API restart recovery.
- resync endpoint.
- cleanup endpoint.
- fake RunPod test suite.

### Milestone 7: Production Hardening

- cost guardrails
- interruptible mode warnings
- volume reuse
- checksum validation everywhere
- user docs

## Cost And Cleanup Guardrails

Implement guardrails before exposing this broadly:

- Default `delete_pod_after_sync = true`.
- Show active Pod status prominently.
- Add a maximum run duration option.
- Add a cleanup watchdog.
- Add a "delete remote resources now" button.
- Refuse to launch if previous RunPod job has unresolved active Pod unless user confirms.
- Store Pod IDs and volume IDs in the job record.
- Provide CLI/admin cleanup command:

```text
python -m app.runpod_cleanup --older-than-hours 24 --dry-run
python -m app.runpod_cleanup --older-than-hours 24 --delete
```

## Open Questions To Resolve During Implementation

1. Exact RunPod create-Pod image field name should be confirmed against the current OpenAPI schema before coding.
2. Whether RunPod HTTP port exposure provides a stable HTTPS proxy URL should be confirmed in a real Pod smoke test.
3. Default GPU/datacenter should be chosen based on availability and cost once the client can query current RunPod resources.
4. If the training image is private, the "API key only" requirement breaks unless container registry auth is also managed. Standard path should use a public image.
5. Private Hugging Face datasets/models require a separate Hugging Face token and should be explicitly out of scope for API-key-only v1.

## Final Definition Of Done

RunPod training is complete when:

- A fresh user can provide only a RunPod API key and launch a GPU training run from `llm-studio`.
- `llm-studio` creates any required RunPod Pod and optional network volume.
- The Pod trains using the same `training.runner` path as local jobs.
- The UI shows live status, metrics, logs, samples, and checkpoints.
- Final checkpoints and artifacts are synced back locally.
- The completed run can be used by the existing inference flow.
- Stop/cancel reliably ends remote GPU spend.
- API restart can recover or clearly mark remote jobs.
- No RunPod API key is stored in job metadata, sent to the Pod, or printed in logs.
- No RunPod S3 API key, SSH session, Jupyter session, or console action is required for the standard flow.
