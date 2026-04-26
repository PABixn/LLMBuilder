# RunPod Training Troubleshooting

## Invalid API Key

Use `Validate key` in the Training page. If validation fails, create a new RunPod API key and paste it again. The key is not logged or stored in job metadata.

## No GPU Capacity

Try another GPU type, datacenter, or cloud type. Secure Cloud is the default. Community Cloud can be cheaper but may be less predictable.

## Pod Stuck Provisioning

The job remains visible locally with the RunPod Pod ID when available. Stop or delete the Pod from `llm-studio`; if the API process has restarted and cannot recover the pod-agent token, use the RunPod console to stop the visible Pod ID.

If the RunPod container logs show Uvicorn listening on `0.0.0.0:8021` but no `/health` or `/v1/jobs/...` requests from the desktop API, the local API is probably trying the wrong agent URL. Current launches expose the pod-agent as a direct TCP port by default and should resolve to `http://<public-ip>:<mapped-port>`. If you explicitly set `LLM_STUDIO_RUNPOD_AGENT_PORT_PROTOCOL=http`, HTTP ports must be reached through RunPod's proxy URL, `https://<pod-id>-8021.proxy.runpod.net`, not a private `100.x` pod IP.

If the desktop API reports Cloudflare 1010 `browser_signature_banned` or an empty 404 from `proxy.runpod.net`, the request was rejected before it reached the Pod. The default `LLM_STUDIO_RUNPOD_AGENT_PORT_PROTOCOL=tcp` bypasses that Cloudflare proxy. If you opt back into `http`, current clients send a browser-compatible user agent for pod-agent requests and treat Cloudflare 1010 as a permanent launch failure instead of retrying until the health timeout. If RunPod changes its proxy policy, override the pod-agent user agent with `LLM_STUDIO_RUNPOD_AGENT_USER_AGENT`.

## Agent Unreachable

The training image must expose the configured agent port, default `8021/tcp` at the RunPod networking layer. Check that `LLM_STUDIO_RUNPOD_TRAINING_IMAGE` points to an image built from `docker/training/Dockerfile`.

For current launches the configured RunPod port is `8021/tcp`, even though the service inside the container is still an HTTP FastAPI server. This avoids the Cloudflare-backed HTTP proxy for authenticated bundle uploads. Use `LLM_STUDIO_RUNPOD_AGENT_PORT_PROTOCOL=http` only when you intentionally want the RunPod proxy transport.

## Pod Agent Is Healthy But Training Never Uses CPU Or GPU

If the RunPod logs repeatedly show `200 OK` health or runtime-state requests while the Pod has no CPU/GPU/VRAM load, verify the image tag. The old `ghcr.io/pabixn/llm-builder-training:sha-7037615` image can boot the HTTP agent but does not contain the shared `llm_builder` package required by the current trainer. Use `ghcr.io/pabixn/llm-builder-training:latest` or another image built from the current `docker/training/Dockerfile`.

The local API uses authenticated `/v1/system` compatibility details when the pod image exposes them. It sends the job id as both the legacy query parameter and the current header because older remote-agent builds require `?job_id=...` on this endpoint. Legacy or buggy pod-agent images that return 404, the known missing-query 422, or a diagnostic-only 5xx from `/v1/system` are allowed to continue to upload/start; if the trainer import is actually broken, the start request should fail with the runner error instead of stopping at the compatibility probe.

On launch failure, the local API stops the Pod by default instead of deleting it so the RunPod console and remote volume remain available for inspection. Use remote cleanup after inspecting the failure if you want to delete the Pod.

## Container Diagnostics

Current training images write structured startup diagnostics to `/workspace/llm-studio/logs/startup.log`, pod-agent events to `/workspace/llm-studio/logs/agent.log`, and runner launch events to `/workspace/llm-studio/logs/runner.log`. These include image revision, Python and CUDA probes, `nvidia-smi`, key import checks, bundle receipt/extraction, start requests, subprocess command, missing input files, immediate exit code, and stdout/stderr tails.

When the pod agent is reachable, the local API syncs those files into the job artifact directory as `runpod_startup.log`, `runpod_agent.log`, and `runpod_runner.log`, and includes them in the Training page log panel with `runpod_lifecycle.log`.

If container access logs show repeated `GET /v1/jobs/<job>/files?path=training_data_preview.json` 404 responses immediately after launch, that file is optional and may not exist until the dataloader preview has been generated. Current clients request optional artifact files with `optional=1`, and current agents return an empty 200 while those files are absent. Older images still log the 404, but the local API treats it as "not available yet"; it is not a launch failure by itself.

If the container logs show `TypeError: remote_agent.app.agent_log() got multiple values for keyword argument 'job_id'` on `GET /v1/system`, the Pod is running an image built before the remote-agent logging fix. Rebuild and push `ghcr.io/pabixn/llm-builder-training:latest` from the current workspace, or set `LLM_STUDIO_RUNPOD_TRAINING_IMAGE` to a freshly built tag. The local launcher no longer depends on `/v1/system` during normal startup so stale images can still reach the bundle/start path, but rebuilt images are required to remove that container-side exception.

The current custom GPT training runner requires `torch`, `datasets`, `tokenizers`, `llm_builder.local_text_data`, and `training.runner`. It does not require Hugging Face `transformers`; older diagnostics briefly probed it as if it were required, which could produce a misleading `ModuleNotFoundError: No module named 'transformers'` line even though the image had the dependencies this runner uses.

## Training Failed

Open the active run logs. `stdout.log`, `stderr.log`, `runtime_state.json`, and checkpoints are synced into the local job directory when the pod agent is reachable.

## Checkpoints Not Synced

Keep the Pod until sync finishes. The local API cannot read a RunPod network volume after deleting the last Pod that can access it unless separate S3 credentials are introduced.

## Cleanup Failed Or Unexpected Charges

Use the Pod ID shown in the active run monitor to stop or delete the Pod in RunPod. Prefer `Delete after sync` for the Pod cleanup policy when you do not need to inspect the remote machine after training.
