"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  fetchRunPodCatalog,
  fetchRunPodStatus,
  validateRunPodKey,
} from "../../../lib/training/providers";
import type {
  RunPodProviderCatalog,
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
  fetchCatalog?: () => Promise<RunPodProviderCatalog>;
  fetchStatus?: () => Promise<RunPodProviderStatus>;
  validateKey?: (apiKey: string) => Promise<RunPodValidateKeyResponse>;
  confirm?: (message: string) => boolean;
}

export function useRunPodSettings({
  fetchCatalog = fetchRunPodCatalog,
  fetchStatus = fetchRunPodStatus,
  validateKey = validateRunPodKey,
  confirm = (message) => window.confirm(message),
}: UseRunPodSettingsOptions = {}) {
  const [executionKind, setExecutionKind] = useState<TrainingExecutionTarget["kind"]>("local");
  const [runPodApiKey, setRunPodApiKey] = useState("");
  const [runPodCatalog, setRunPodCatalog] = useState<RunPodProviderCatalog | null>(null);
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
    void Promise.allSettled([fetchStatus(), fetchCatalog()]).then(([statusResult, catalogResult]) => {
      if (cancelled) {
        return;
      }
      if (statusResult.status === "fulfilled") {
        const status = statusResult.value;
        setRunPodStatus(status);
        setRunPodGpuType(status.defaults.gpu_type_id);
        setRunPodGpuCount(status.defaults.gpu_count);
        setRunPodCloudType(status.defaults.cloud_type);
        setRunPodDataCenterId(status.defaults.data_center_id ?? "");
        setRunPodVolumeSizeGb(status.defaults.network_volume_size_gb);
        setRunPodCleanupPod(status.defaults.cleanup_policy.pod);
      } else {
        const error = statusResult.reason;
        setRunPodValidationMessage(error instanceof Error ? error.message : "RunPod settings unavailable.");
      }
      if (catalogResult.status === "fulfilled") {
        setRunPodCatalog(catalogResult.value);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [fetchCatalog, fetchStatus]);

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
    runPodCatalog,
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
    setRunPodGpuCount,
    setRunPodGpuType,
    setRunPodInterruptible,
    setRunPodVolumeSizeGb,
  };
}
