import {
  FiCpu,
  FiServer,
} from "react-icons/fi";

import type {
  RunPodProviderStatus,
  TrainingExecutionTarget,
} from "../../../lib/training/types";
import type {
  RunPodCleanupPodAction,
  RunPodCloudType,
} from "../lib/runPod";
import { RunPodSettingsPanel } from "./RunPodSettingsPanel";

interface ExecutionTargetPanelProps {
  executionKind: TrainingExecutionTarget["kind"];
  onExecutionKindChange: (value: TrainingExecutionTarget["kind"]) => void;
  runPodApiKey: string;
  runPodCleanupPod: RunPodCleanupPodAction;
  runPodCloudType: RunPodCloudType;
  runPodDataCenterId: string;
  runPodGpuCount: number;
  runPodGpuType: string;
  runPodInterruptible: boolean;
  runPodStatus: RunPodProviderStatus | null;
  runPodValidationLoading: boolean;
  runPodValidationMessage: string | null;
  runPodVolumeSizeGb: number;
  setRunPodApiKey: (value: string) => void;
  setRunPodCleanupPod: (value: RunPodCleanupPodAction) => void;
  setRunPodCloudType: (value: RunPodCloudType) => void;
  setRunPodDataCenterId: (value: string) => void;
  setRunPodGpuCount: (value: number) => void;
  setRunPodGpuType: (value: string) => void;
  setRunPodInterruptible: (value: boolean) => void;
  setRunPodVolumeSizeGb: (value: number) => void;
  onValidateRunPodKey: () => void;
}

export function ExecutionTargetPanel({
  executionKind,
  onExecutionKindChange,
  onValidateRunPodKey,
  runPodApiKey,
  runPodCleanupPod,
  runPodCloudType,
  runPodDataCenterId,
  runPodGpuCount,
  runPodGpuType,
  runPodInterruptible,
  runPodStatus,
  runPodValidationLoading,
  runPodValidationMessage,
  runPodVolumeSizeGb,
  setRunPodApiKey,
  setRunPodCleanupPod,
  setRunPodCloudType,
  setRunPodDataCenterId,
  setRunPodGpuCount,
  setRunPodGpuType,
  setRunPodInterruptible,
  setRunPodVolumeSizeGb,
}: ExecutionTargetPanelProps) {
  return (
    <details className="settingsPanel" open>
      <summary>Execution target</summary>
      <div className="settingsGrid">
        <div className="settingsGroup">
          <div className="settingsGroupHeader">
            <h3>Where this run trains</h3>
          </div>
          <div className="modeSwitch">
            <button
              type="button"
              className={`modeSwitchButton ${executionKind === "local" ? "modeSwitchButton-active" : ""}`}
              onClick={() => onExecutionKindChange("local")}
            >
              <FiCpu aria-hidden="true" />
              Local machine
            </button>
            <button
              type="button"
              className={`modeSwitchButton ${executionKind === "runpod_pod" ? "modeSwitchButton-active" : ""}`}
              onClick={() => onExecutionKindChange("runpod_pod")}
            >
              <FiServer aria-hidden="true" />
              RunPod Pod
            </button>
          </div>

          {executionKind === "runpod_pod" ? (
            <RunPodSettingsPanel
              apiKey={runPodApiKey}
              cleanupPod={runPodCleanupPod}
              cloudType={runPodCloudType}
              dataCenterId={runPodDataCenterId}
              gpuCount={runPodGpuCount}
              gpuType={runPodGpuType}
              interruptible={runPodInterruptible}
              onApiKeyChange={setRunPodApiKey}
              onCleanupPodChange={setRunPodCleanupPod}
              onCloudTypeChange={setRunPodCloudType}
              onDataCenterIdChange={setRunPodDataCenterId}
              onGpuCountChange={setRunPodGpuCount}
              onGpuTypeChange={setRunPodGpuType}
              onInterruptibleChange={setRunPodInterruptible}
              onValidateKey={onValidateRunPodKey}
              onVolumeSizeGbChange={setRunPodVolumeSizeGb}
              status={runPodStatus}
              validationLoading={runPodValidationLoading}
              validationMessage={runPodValidationMessage}
              volumeSizeGb={runPodVolumeSizeGb}
            />
          ) : null}
        </div>
      </div>
    </details>
  );
}
