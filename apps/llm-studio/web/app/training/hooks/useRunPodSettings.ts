"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  fetchRunPodStatus,
  validateRunPodKey,
} from "../../../lib/training/providers";
import type {
  RunPodProviderStatus,
  RunPodValidateKeyResponse,
  TrainingExecutionTarget,
} from "../../../lib/training/types";
import {
  buildTrainingExecutionTarget,
  confirmRunPodLaunch,
  isRunPodLaunchReady,
  type RunPodCleanupPodAction,
  type RunPodCloudType,
  type RunPodLaunchSettings,
} from "../lib/runPod";

interface UseRunPodSettingsOptions {
  fetchStatus?: () => Promise<RunPodProviderStatus>;
  validateKey?: (apiKey: string) => Promise<RunPodValidateKeyResponse>;
  confirm?: (message: string) => boolean;
}

export function useRunPodSettings({
  fetchStatus = fetchRunPodStatus,
  validateKey = validateRunPodKey,
  confirm = (message) => window.confirm(message),
}: UseRunPodSettingsOptions = {}) {
  const [executionKind, setExecutionKind] = useState<TrainingExecutionTarget["kind"]>("local");
  const [runPodApiKey, setRunPodApiKey] = useState("");
  const [runPodStatus, setRunPodStatus] = useState<RunPodProviderStatus | null>(null);
  const [runPodValidationMessage, setRunPodValidationMessage] = useState<string | null>(null);
  const [runPodValidationLoading, setRunPodValidationLoading] = useState(false);
  const [runPodGpuType, setRunPodGpuType] = useState("");
  const [runPodGpuCount, setRunPodGpuCount] = useState(1);
  const [runPodCloudType, setRunPodCloudType] = useState<RunPodCloudType>("SECURE");
  const [runPodDataCenterId, setRunPodDataCenterId] = useState("");
  const [runPodVolumeSizeGb, setRunPodVolumeSizeGb] = useState(100);
  const [runPodInterruptible, setRunPodInterruptible] = useState(false);
  const [runPodCleanupPod, setRunPodCleanupPod] =
    useState<RunPodCleanupPodAction>("delete_after_sync");

  useEffect(() => {
    let cancelled = false;
    fetchStatus()
      .then((status) => {
        if (cancelled) {
          return;
        }
        setRunPodStatus(status);
        setRunPodGpuType(status.defaults.gpu_type_id);
        setRunPodGpuCount(status.defaults.gpu_count);
        setRunPodCloudType(status.defaults.cloud_type);
        setRunPodDataCenterId(status.defaults.data_center_id ?? "");
        setRunPodVolumeSizeGb(status.defaults.network_volume_size_gb);
        setRunPodCleanupPod(status.defaults.cleanup_policy.pod);
      })
      .catch((error) => {
        if (!cancelled) {
          setRunPodValidationMessage(error instanceof Error ? error.message : "RunPod settings unavailable.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [fetchStatus]);

  const runPodLaunchSettings = useMemo<RunPodLaunchSettings>(
    () => ({
      apiKey: runPodApiKey,
      gpuType: runPodGpuType,
      gpuCount: runPodGpuCount,
      cloudType: runPodCloudType,
      dataCenterId: runPodDataCenterId,
      interruptible: runPodInterruptible,
      volumeSizeGb: runPodVolumeSizeGb,
      cleanupPod: runPodCleanupPod,
    }),
    [
      runPodApiKey,
      runPodCleanupPod,
      runPodCloudType,
      runPodDataCenterId,
      runPodGpuCount,
      runPodGpuType,
      runPodInterruptible,
      runPodVolumeSizeGb,
    ]
  );

  const buildExecutionTarget = useCallback(
    () => buildTrainingExecutionTarget(executionKind, runPodLaunchSettings),
    [executionKind, runPodLaunchSettings]
  );

  const confirmLaunch = useCallback(
    () => confirmRunPodLaunch(executionKind, runPodLaunchSettings, confirm),
    [confirm, executionKind, runPodLaunchSettings]
  );

  const handleValidateRunPodKey = useCallback(async () => {
    const key = runPodApiKey.trim();
    if (!key) {
      setRunPodValidationMessage("Paste a RunPod API key first.");
      return;
    }
    setRunPodValidationLoading(true);
    try {
      const result = await validateKey(key);
      setRunPodValidationMessage(result.message);
      if (result.valid) {
        const status = await fetchStatus();
        setRunPodStatus(status);
      }
    } catch (error) {
      setRunPodValidationMessage(error instanceof Error ? error.message : "RunPod key validation failed.");
    } finally {
      setRunPodValidationLoading(false);
    }
  }, [fetchStatus, runPodApiKey, validateKey]);

  return {
    buildExecutionTarget,
    confirmLaunch,
    executionKind,
    handleValidateRunPodKey,
    runPodApiKey,
    runPodCleanupPod,
    runPodCloudType,
    runPodDataCenterId,
    runPodGpuCount,
    runPodGpuType,
    runPodInterruptible,
    runPodReady: isRunPodLaunchReady(executionKind, runPodStatus, runPodApiKey),
    runPodStatus,
    runPodValidationLoading,
    runPodValidationMessage,
    runPodVolumeSizeGb,
    setExecutionKind,
    setRunPodApiKey,
    setRunPodCleanupPod,
    setRunPodCloudType,
    setRunPodDataCenterId,
    setRunPodGpuCount,
    setRunPodGpuType,
    setRunPodInterruptible,
    setRunPodVolumeSizeGb,
  };
}
