import {
  FiAlertTriangle,
  FiCheckCircle,
  FiList,
} from "react-icons/fi";

import type { RunPodProviderStatus } from "../../../lib/training/types";
import { ConfigNumberInput } from "../../shared/components/ConfigNumberInput";
import type {
  RunPodCleanupPodAction,
  RunPodCloudType,
} from "../lib/runPod";

interface RunPodSettingsPanelProps {
  apiKey: string;
  cleanupPod: RunPodCleanupPodAction;
  cloudType: RunPodCloudType;
  dataCenterId: string;
  gpuCount: number;
  gpuType: string;
  interruptible: boolean;
  onApiKeyChange: (value: string) => void;
  onCleanupPodChange: (value: RunPodCleanupPodAction) => void;
  onCloudTypeChange: (value: RunPodCloudType) => void;
  onDataCenterIdChange: (value: string) => void;
  onGpuCountChange: (value: number) => void;
  onGpuTypeChange: (value: string) => void;
  onInterruptibleChange: (value: boolean) => void;
  onValidateKey: () => void;
  status: RunPodProviderStatus | null;
  validationLoading: boolean;
  validationMessage: string | null;
  volumeSizeGb: number;
  onVolumeSizeGbChange: (value: number) => void;
}

export function RunPodSettingsPanel({
  apiKey,
  cleanupPod,
  cloudType,
  dataCenterId,
  gpuCount,
  gpuType,
  interruptible,
  onApiKeyChange,
  onCleanupPodChange,
  onCloudTypeChange,
  onDataCenterIdChange,
  onGpuCountChange,
  onGpuTypeChange,
  onInterruptibleChange,
  onValidateKey,
  onVolumeSizeGbChange,
  status,
  validationLoading,
  validationMessage,
  volumeSizeGb,
}: RunPodSettingsPanelProps) {
  const keySource = status?.configured
    ? status.source === "environment"
      ? "Environment key"
      : "Validated key"
    : apiKey.trim()
      ? "Typed key not validated"
      : "No key configured";
  const cleanupLabel =
    cleanupPod === "delete_after_sync"
      ? "Delete pod after final sync"
      : cleanupPod === "stop_after_sync"
        ? "Stop pod after sync"
        : "Keep pod running";
  const datacenterLabel = dataCenterId.trim() || "Any available";

  return (
    <div className="settingsStack">
      <div className="statusGrid">
        <div className="statusCard">
          <div className="statusCardIcon"><FiCheckCircle /></div>
          <div>
            <div className="statusCardTitle">RunPod access</div>
            <div className="statusCardValue">{keySource}</div>
            <div className="statusCardDetail">{validationMessage ?? "Keys are sent only with launch and validation requests."}</div>
          </div>
        </div>
        <div className="statusCard">
          <div className="statusCardIcon"><FiList /></div>
          <div>
            <div className="statusCardTitle">Selected pod</div>
            <div className="statusCardValue">{gpuCount} x {gpuType}</div>
            <div className="statusCardDetail">
              {cloudType} cloud · {datacenterLabel} · {volumeSizeGb} GB pod volume · TCP port
            </div>
          </div>
        </div>
        <div className="statusCard">
          <div className="statusCardIcon"><FiAlertTriangle /></div>
          <div>
            <div className="statusCardTitle">Cleanup</div>
            <div className="statusCardValue">{cleanupLabel}</div>
            <div className="statusCardDetail">
              {cleanupPod === "keep"
                ? "Keeping a pod can continue billing until you stop or delete it in RunPod."
                : "Cleanup runs only after final outputs are synced back here."}
            </div>
          </div>
        </div>
      </div>

      <div className="fieldGrid trainingSettingsCompactGrid">
        <label className="fieldLabel">
          <span>RunPod API key</span>
          <input
            className="textInput"
            type="password"
            value={apiKey}
            placeholder={status?.source === "environment" ? "Using environment key" : "Paste RunPod API key"}
            onChange={(event) => onApiKeyChange(event.target.value)}
          />
        </label>
        <label className="fieldLabel">
          <span>GPU type</span>
          <input
            className="textInput"
            value={gpuType}
            onChange={(event) => onGpuTypeChange(event.target.value)}
          />
        </label>
        <label className="fieldLabel">
          <span>GPU count</span>
          <ConfigNumberInput value={gpuCount} onCommit={onGpuCountChange} />
        </label>
        <label className="fieldLabel">
          <span>Cloud</span>
          <select className="selectInput" value={cloudType} onChange={(event) => onCloudTypeChange(event.target.value as RunPodCloudType)}>
            <option value="SECURE">Secure Cloud</option>
            <option value="COMMUNITY">Community Cloud</option>
          </select>
        </label>
        <label className="fieldLabel">
          <span>Datacenter</span>
          <input
            className="textInput"
            value={dataCenterId}
            placeholder="Any available"
            onChange={(event) => onDataCenterIdChange(event.target.value)}
          />
        </label>
        <label className="fieldLabel">
          <span>Pod volume GB</span>
          <ConfigNumberInput value={volumeSizeGb} onCommit={onVolumeSizeGbChange} />
        </label>
        <label className="fieldLabel">
          <span>Pod cleanup</span>
          <select className="selectInput" value={cleanupPod} onChange={(event) => onCleanupPodChange(event.target.value as RunPodCleanupPodAction)}>
            <option value="delete_after_sync">Delete after sync</option>
            <option value="stop_after_sync">Stop after sync</option>
            <option value="keep">Keep running</option>
          </select>
        </label>
      </div>
      <label className="trainingCheckboxLine">
        <input
          type="checkbox"
          checked={interruptible}
          onChange={(event) => onInterruptibleChange(event.target.checked)}
        />
        <span>Use interruptible capacity</span>
      </label>
      {interruptible ? (
        <p className="settingsGroupHint">
          Interruptible pods can be preempted by RunPod. Use them only when restarting the run is acceptable.
        </p>
      ) : null}
      {cleanupPod === "keep" ? (
        <p className="settingsGroupHint">
          Keep running is for debugging. The app will not automatically stop billing for that pod.
        </p>
      ) : null}
      <div className="settingsGroup">
        <div className="settingsGroupHeader">
          <h3>What happens next</h3>
        </div>
        <div className="trainingWorkflowList">
          {[
            "Create RunPod pod",
            "Wait for exposed agent port",
            "Start pod agent",
            "Upload training bundle",
            "Start trainer",
            "Sync logs, metrics, samples, checkpoints, and manifest",
            cleanupLabel,
          ].map((item) => (
            <div key={item} className="trainingWorkflowStep trainingWorkflowStep-waiting">
              <span className="trainingWorkflowStepStatus">Next</span>
              <span>{item}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="trainingPromptToolbar">
        <button type="button" className="buttonSecondary" onClick={onValidateKey} disabled={validationLoading || apiKey.trim() === ""}>
          <FiCheckCircle aria-hidden="true" />
          {validationLoading ? "Validating..." : "Validate key"}
        </button>
        <span className={`pillBadge ${status?.configured ? "tone-good" : "tone-neutral"}`}>
          {status?.configured ? `Key ready (${status.source})` : "Key required"}
        </span>
      </div>
      {validationMessage ? (
        <p className="settingsGroupHint">{validationMessage}</p>
      ) : null}
    </div>
  );
}
