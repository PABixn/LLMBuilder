import type {
  RunPodProviderStatus,
  TrainingExecutionTarget,
} from "../../../lib/training/types";

export type RunPodCloudType = NonNullable<TrainingExecutionTarget["cloud_type"]>;
export type RunPodCleanupPodAction = NonNullable<
  TrainingExecutionTarget["cleanup_policy"]
>["pod"];

export interface RunPodLaunchSettings {
  apiKey: string;
  gpuType: string;
  gpuCount: number;
  cloudType: RunPodCloudType;
  dataCenterId: string;
  interruptible: boolean;
  volumeSizeGb: number;
  cleanupPod: RunPodCleanupPodAction;
}

export const RUNPOD_INTERRUPTIBLE_CONFIRMATION =
  "Start an interruptible RunPod pod? It can be preempted while training is running.";

export const RUNPOD_KEEP_POD_CONFIRMATION =
  "Keep the RunPod pod alive after training? This can continue billing until you stop it.";

export function buildRunPodExecutionTarget(
  settings: RunPodLaunchSettings
): TrainingExecutionTarget {
  return {
    kind: "runpod_pod",
    api_key: settings.apiKey.trim() || null,
    gpu_type_id: settings.gpuType.trim() || null,
    gpu_count: settings.gpuCount,
    cloud_type: settings.cloudType,
    data_center_id: settings.dataCenterId.trim() || null,
    interruptible: settings.interruptible,
    network_volume_size_gb: settings.volumeSizeGb,
    cleanup_policy: {
      pod: settings.cleanupPod,
      network_volume: "keep",
    },
  };
}

export function buildTrainingExecutionTarget(
  executionKind: TrainingExecutionTarget["kind"],
  settings: RunPodLaunchSettings
): TrainingExecutionTarget {
  return executionKind === "local"
    ? { kind: "local" }
    : buildRunPodExecutionTarget(settings);
}

export function getRunPodLaunchConfirmationMessages(
  executionKind: TrainingExecutionTarget["kind"],
  settings: RunPodLaunchSettings
): string[] {
  if (executionKind !== "runpod_pod") {
    return [];
  }

  const messages: string[] = [];
  if (settings.interruptible) {
    messages.push(RUNPOD_INTERRUPTIBLE_CONFIRMATION);
  }
  if (settings.cleanupPod === "keep") {
    messages.push(RUNPOD_KEEP_POD_CONFIRMATION);
  }
  return messages;
}

export function confirmRunPodLaunch(
  executionKind: TrainingExecutionTarget["kind"],
  settings: RunPodLaunchSettings,
  confirm: (message: string) => boolean
): boolean {
  return getRunPodLaunchConfirmationMessages(executionKind, settings).every(confirm);
}

export function isRunPodLaunchReady(
  executionKind: TrainingExecutionTarget["kind"],
  status: RunPodProviderStatus | null,
  apiKey: string
): boolean {
  return executionKind === "local" || Boolean(status?.configured || apiKey.trim());
}
