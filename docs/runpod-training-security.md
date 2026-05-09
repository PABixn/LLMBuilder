# RunPod Training Security

## API Key Handling

The RunPod API key is used only by the local `llm-studio` API to create, inspect, stop, and delete RunPod resources. It is never sent to the frontend after validation, never sent to the Pod, and never written to training job metadata.

Keys can come from `LLM_STUDIO_RUNPOD_API_KEY` or from the Training page. UI-pasted keys are in-memory only and disappear when the API process exits.

## Pod-Agent Token

Each RunPod job gets a random pod-agent bearer token. The token is injected into that Pod only and every agent endpoint requires:

- `Authorization: Bearer <token>`
- `X-LLM-Studio-Job-Id: <job_id>`

The local database stores only a SHA-256 token hash. The raw token is kept in process memory so API restarts do not leak it to disk. After an API restart, the app can still show the RunPod pod ID from job metadata, but it cannot authenticate to the pod agent unless a new supported recovery flow is added.

## Uploaded Data

The job bundle contains model config, tokenizer artifact, training config, dataloader config, resolved preflight metadata, and local dataset files referenced by the dataloader. Hugging Face dataset references remain remote references. Bundle staging directories are removed after a successful archive is written.

## No S3 Key Requirement

RunPod network-volume S3 access requires separate S3 credentials. The standard `llm-studio` path avoids that requirement by using the Pod's attached volume and uploading/downloading through the authenticated pod agent over the exposed HTTP port.

## File Access Controls

The pod agent rejects path traversal and serves files only from the current job output directory. Downloaded checkpoints are size/checksum verified when the pod-agent manifest provides those fields, and `Delete after sync` waits for a verified final `artifact_manifest.json`.

## Deleting Remote Resources

Use the Training page stop/cleanup controls first. If cleanup fails, use the displayed Pod ID in RunPod to stop or delete the Pod.
