import {
  FiCpu,
  FiServer,
} from "react-icons/fi";

import type {
  RunPodProviderCatalog,
  RunPodProviderStatus,
  TrainingExecutionTarget,
} from "../../../lib/training/types";
import type {
  RunPodCleanupPodAction,
  RunPodCloudType,
} from "../lib/runPod";
import { RunPodSettingsPanel } from "./RunPodSettingsPanel";
import { HelpTooltip, InfoTooltip } from "../../shared/components/HelpTooltip";

interface ExecutionTargetPanelProps {
  executionKind: TrainingExecutionTarget["kind"];
  onExecutionKindChange: (value: TrainingExecutionTarget["kind"]) => void;
  runPodApiKey: string;
  runPodCatalog: RunPodProviderCatalog | null;
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
  runPodCatalog,
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
  setRunPodGpuCount,
  setRunPodGpuType,
  setRunPodInterruptible,
  setRunPodVolumeSizeGb,
}: ExecutionTargetPanelProps) {
  return (
    <details className="settingsPanel" open>
      <summary>
        <span>Execution target</span>
        <InfoTooltip label="Execution target explanation" align="right" width="wide">
          <strong>Execution target</strong>
          <p>
            Choose where the training process runs. Local uses this machine; RunPod
            provisions a remote GPU pod and syncs the final artifacts back.
          </p>
        </InfoTooltip>
      </summary>
      <div className="settingsGrid">
        <div className="settingsGroup">
          <div className="settingsGroupHeader">
            <h3>Training location</h3>
          </div>
          <div className="modeSwitch">
            <HelpTooltip label="Local machine training" content="Runs the trainer on this computer using the local Python environment and available device. Use it for small tests or when you already have enough local GPU/CPU memory.">
              <button
                type="button"
                className={`modeSwitchButton ${executionKind === "local" ? "modeSwitchButton-active" : ""}`}
                onClick={() => onExecutionKindChange("local")}
              >
                <FiCpu aria-hidden="true" />
                Local machine
              </button>
            </HelpTooltip>
            <HelpTooltip label="RunPod training" content="Creates a remote RunPod GPU pod, uploads the run bundle, monitors the agent, downloads artifacts, then applies the cleanup policy you choose below.">
              <button
                type="button"
                className={`modeSwitchButton ${executionKind === "runpod_pod" ? "modeSwitchButton-active" : ""}`}
                onClick={() => onExecutionKindChange("runpod_pod")}
              >
                <FiServer aria-hidden="true" />
                RunPod
              </button>
            </HelpTooltip>
          </div>

          {executionKind === "runpod_pod" ? (
            <RunPodSettingsPanel
              apiKey={runPodApiKey}
              catalog={runPodCatalog}
              cleanupPod={runPodCleanupPod}
              cloudType={runPodCloudType}
              dataCenterId={runPodDataCenterId}
              gpuCount={runPodGpuCount}
              gpuType={runPodGpuType}
              interruptible={runPodInterruptible}
              onApiKeyChange={setRunPodApiKey}
              onCleanupPodChange={setRunPodCleanupPod}
              onCloudTypeChange={setRunPodCloudType}
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
