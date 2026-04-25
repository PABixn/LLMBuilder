# RunPod Training Troubleshooting

## Invalid API Key

Use `Validate key` in the Training page. If validation fails, create a new RunPod API key and paste it again. The key is not logged or stored in job metadata.

## No GPU Capacity

Try another GPU type, datacenter, or cloud type. Secure Cloud is the default. Community Cloud can be cheaper but may be less predictable.

## Pod Stuck Provisioning

The job remains visible locally with the RunPod Pod ID when available. Stop or delete the Pod from `llm-studio`; if the API process has restarted and cannot recover the pod-agent token, use the RunPod console to stop the visible Pod ID.

If the RunPod container logs show Uvicorn listening on `0.0.0.0:8021` but no `/health` or `/v1/jobs/...` requests from the desktop API, the local API is probably trying the wrong agent URL. HTTP ports must be reached through RunPod's proxy URL, `https://<pod-id>-8021.proxy.runpod.net`, not a private `100.x` pod IP.

## Agent Unreachable

The training image must expose the configured agent port, default `8021/http`. Check that `LLM_STUDIO_RUNPOD_TRAINING_IMAGE` points to an image built from `docker/training/Dockerfile`.

## Pod Agent Is Healthy But Training Never Uses CPU Or GPU

If the RunPod logs repeatedly show `200 OK` health or runtime-state requests while the Pod has no CPU/GPU/VRAM load, verify the image tag. The old `ghcr.io/pabixn/llm-builder-training:sha-7037615` image can boot the HTTP agent but does not contain the shared `llm_builder` package required by the current trainer. Use `ghcr.io/pabixn/llm-builder-training:latest` or another image built from the current `docker/training/Dockerfile`.

The local API now performs an authenticated `/v1/system` compatibility check before uploading a job. A stale image should fail during launch with a trainer-import error instead of leaving a healthy idle Pod.

## Training Failed

Open the active run logs. `stdout.log`, `stderr.log`, `runtime_state.json`, and checkpoints are synced into the local job directory when the pod agent is reachable.

## Checkpoints Not Synced

Keep the Pod until sync finishes. The local API cannot read a RunPod network volume after deleting the last Pod that can access it unless separate S3 credentials are introduced.

## Cleanup Failed Or Unexpected Charges

Use the Pod ID shown in the active run monitor to stop or delete the Pod in RunPod. Prefer `Delete after sync` for the Pod cleanup policy when you do not need to inspect the remote machine after training.
