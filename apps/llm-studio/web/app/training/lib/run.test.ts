import assert from "node:assert/strict";
import test from "node:test";

import type { TrainingJob, TrainingJobStatus } from "../../../lib/training/types";
import { shouldPollTrainingRun } from "./display";
import {
  RUNPOD_INTERRUPTIBLE_CONFIRMATION,
  RUNPOD_KEEP_POD_CONFIRMATION,
  buildRunPodExecutionTarget,
  buildTrainingExecutionTarget,
  confirmRunPodLaunch,
  getRunPodLaunchConfirmationMessages,
  isRunPodLaunchReady,
  type RunPodLaunchSettings,
} from "./runPod";

function jobWithStatus(status: TrainingJobStatus): TrainingJob {
  return { status } as TrainingJob;
}

const runPodSettings: RunPodLaunchSettings = {
  apiKey: "  rp_test_key  ",
  gpuType: "  NVIDIA A100 80GB PCIe  ",
  gpuCount: 2,
  cloudType: "SECURE",
  dataCenterId: "  EU-RO-1  ",
  interruptible: false,
  volumeSizeGb: 120,
  cleanupPod: "delete_after_sync",
};

test("training polling continues only for non-terminal statuses", () => {
  assert.equal(shouldPollTrainingRun(jobWithStatus("pending")), true);
  assert.equal(shouldPollTrainingRun(jobWithStatus("running")), true);
  assert.equal(shouldPollTrainingRun(jobWithStatus("completed")), false);
  assert.equal(shouldPollTrainingRun(jobWithStatus("failed")), false);
  assert.equal(shouldPollTrainingRun(jobWithStatus("cancelled")), false);
});

test("runpod execution target preserves launch settings and trims optional strings", () => {
  assert.deepEqual(buildRunPodExecutionTarget(runPodSettings), {
    kind: "runpod_pod",
    api_key: "rp_test_key",
    gpu_type_id: "NVIDIA A100 80GB PCIe",
    gpu_count: 2,
    cloud_type: "SECURE",
    data_center_id: "EU-RO-1",
    interruptible: false,
    network_volume_size_gb: 120,
    cleanup_policy: {
      pod: "delete_after_sync",
      network_volume: "keep",
    },
  });
});

test("runpod execution target sends null for blank optional strings", () => {
  assert.deepEqual(
    buildRunPodExecutionTarget({
      ...runPodSettings,
      apiKey: "   ",
      gpuType: "",
      dataCenterId: " ",
    }),
    {
      kind: "runpod_pod",
      api_key: null,
      gpu_type_id: null,
      gpu_count: 2,
      cloud_type: "SECURE",
      data_center_id: null,
      interruptible: false,
      network_volume_size_gb: 120,
      cleanup_policy: {
        pod: "delete_after_sync",
        network_volume: "keep",
      },
    }
  );
});

test("training execution target keeps local launches minimal", () => {
  assert.deepEqual(buildTrainingExecutionTarget("local", runPodSettings), {
    kind: "local",
  });
});

test("runpod launch confirmations match current warning behavior", () => {
  assert.deepEqual(
    getRunPodLaunchConfirmationMessages("runpod_pod", {
      ...runPodSettings,
      interruptible: true,
      cleanupPod: "keep",
    }),
    [RUNPOD_INTERRUPTIBLE_CONFIRMATION, RUNPOD_KEEP_POD_CONFIRMATION]
  );
  assert.deepEqual(
    getRunPodLaunchConfirmationMessages("local", {
      ...runPodSettings,
      interruptible: true,
      cleanupPod: "keep",
    }),
    []
  );
});

test("runpod launch confirmation short-circuits when the user rejects a warning", () => {
  const prompts: string[] = [];
  const confirmed = confirmRunPodLaunch(
    "runpod_pod",
    {
      ...runPodSettings,
      interruptible: true,
      cleanupPod: "keep",
    },
    (message) => {
      prompts.push(message);
      return false;
    }
  );

  assert.equal(confirmed, false);
  assert.deepEqual(prompts, [RUNPOD_INTERRUPTIBLE_CONFIRMATION]);
});

test("runpod readiness allows local runs and requires either configured status or typed key for pods", () => {
  assert.equal(isRunPodLaunchReady("local", null, ""), true);
  assert.equal(isRunPodLaunchReady("runpod_pod", null, ""), false);
  assert.equal(isRunPodLaunchReady("runpod_pod", null, " rp_test_key "), true);
  assert.equal(
    isRunPodLaunchReady(
      "runpod_pod",
      {
        configured: true,
        validated: true,
        source: "environment",
        defaults: {
          gpu_type_id: "NVIDIA A100",
          gpu_count: 1,
          cloud_type: "SECURE",
          data_center_id: null,
          network_volume_size_gb: 100,
          container_disk_gb: 40,
          volume_mount_path: "/workspace",
          training_image: "example/image:latest",
          agent_port: 8000,
          agent_port_protocol: "http",
          cleanup_policy: {
            pod: "delete_after_sync",
            network_volume: "keep",
          },
        },
      },
      ""
    ),
    true
  );
});
