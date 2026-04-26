# RunPod Training User Guide

`llm-studio` can launch a managed RunPod GPU Pod for a model-training run, upload the job inputs, stream metrics/logs/checkpoints back to the local training UI, and stop or delete the Pod after artifacts sync.

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
5. Choose GPU type, GPU count, Secure or Community Cloud, optional datacenter, volume size, interruptible mode, and cleanup policy.
6. Click `Start training`.

The local API creates the job record first, creates a RunPod Pod with the training image, exposes the pod-agent over a direct TCP port by default, uploads a signed bundle to the pod agent, starts `python -m training.runner`, then polls and syncs outputs into the normal local job directory.

## Cleanup Policies

`Delete after sync` deletes the Pod after final outputs are local. This is the default spend-safe setting.

`Stop after sync` stops the Pod but leaves it in the RunPod account.

`Keep running` leaves the Pod alive after training and can continue billing until stopped.

Network volumes default to `keep` so dataset caches can be reused. Choose `delete after sync` only when you intentionally want to remove the remote cache.

## Stopping A Run

Use `Stop run` in the active-run monitor. `llm-studio` asks the pod agent to cancel the training process, syncs what is available, and stops the Pod. If the agent is unreachable, the API still attempts to stop the RunPod Pod.
