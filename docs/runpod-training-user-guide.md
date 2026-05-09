# RunPod Training User Guide

`llm-studio` can launch a managed RunPod GPU Pod for a model-training run, upload the job inputs, stream metrics/logs/checkpoints back to the local training UI, verify the final artifact manifest, and stop or delete the Pod after artifacts sync.

## Setup

1. Create a RunPod API key in the RunPod account settings.
2. Start the `llm-studio` API and web app.
3. Open the Training page.
4. In `Execution target`, choose `RunPod Pod`.
5. Paste the API key and click `Validate key`.

You can also set `LLM_STUDIO_RUNPOD_API_KEY` in the API environment. UI-pasted keys are held only in memory for the current API process.

## Launch Flow

1. Select a saved model project.
2. Select a completed tokenizer artifact.
3. Configure datasets, training settings, and sampling prompts.
4. Confirm preflight is valid.
5. Choose GPU type, GPU count, Secure or Community Cloud, optional datacenter, pod volume size, interruptible mode, and Pod cleanup policy.
6. Click `Start training`.

The local API creates the job record first, creates a RunPod Pod with the training image, exposes the pod-agent over a direct TCP port by default, checks pod-agent protocol compatibility, uploads a signed bundle to the pod agent, starts `python -m training.runner`, then polls and syncs outputs into the normal local job directory.

The Training page shows the setup summary, launch lifecycle, pod ID, last heartbeat, last sync, remote error, and cleanup policy for active RunPod jobs. Recovery actions that require persistent pod-agent credentials are shown as unavailable after an API restart.

## Cleanup Policies

`Delete after sync` deletes the Pod only after final outputs are local and `artifact_manifest.json` has been downloaded and parsed. This is the default spend-safe setting.

`Stop after sync` stops the Pod but leaves it in the RunPod account.

`Keep running` leaves the Pod alive after training and can continue billing until stopped.

The current implementation uses the Pod's attached volume size. It does not create a separate RunPod network volume resource, so volume cleanup is not offered separately from Pod cleanup.

## Stopping A Run

Use `Stop run` in the active-run monitor. `llm-studio` asks the pod agent to cancel the training process, syncs what is available, and stops the Pod. If the agent is unreachable, the API still attempts to stop the RunPod Pod.

## Synced Artifacts

The local API syncs append-only logs, metrics, samples, latest JSON files, checkpoints listed by the pod-agent checkpoint manifest, and the final artifact manifest. Checkpoint files are written under the local job directory at `checkpoints/<step>/` and are size/checksum verified when the remote manifest includes those fields.
