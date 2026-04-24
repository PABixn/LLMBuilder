# RunPod Training Troubleshooting

## Invalid API Key

Use `Validate key` in the Training page. If validation fails, create a new RunPod API key and paste it again. The key is not logged or stored in job metadata.

## No GPU Capacity

Try another GPU type, datacenter, or cloud type. Secure Cloud is the default. Community Cloud can be cheaper but may be less predictable.

## Pod Stuck Provisioning

The job remains visible locally with the RunPod Pod ID when available. Stop or delete the Pod from `llm-studio`; if the API process has restarted and cannot recover the pod-agent token, use the RunPod console to stop the visible Pod ID.

## Agent Unreachable

The training image must expose the configured agent port, default `8021/http`. Check that `LLM_STUDIO_RUNPOD_TRAINING_IMAGE` points to an image built from `docker/training/Dockerfile`.

## Training Failed

Open the active run logs. `stdout.log`, `stderr.log`, `runtime_state.json`, and checkpoints are synced into the local job directory when the pod agent is reachable.

## Checkpoints Not Synced

Keep the Pod until sync finishes. The local API cannot read a RunPod network volume after deleting the last Pod that can access it unless separate S3 credentials are introduced.

## Cleanup Failed Or Unexpected Charges

Use the Pod ID shown in the active run monitor to stop or delete the Pod in RunPod. Prefer `Delete after sync` for the Pod cleanup policy when you do not need to inspect the remote machine after training.
