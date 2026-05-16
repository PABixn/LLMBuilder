import {
  FiAlertTriangle,
  FiCheckCircle,
  FiCpu,
  FiMapPin,
} from "react-icons/fi";

import type {
  RunPodProviderCatalog,
  RunPodProviderStatus,
} from "../../../lib/training/types";
import { ConfigNumberInput } from "../../shared/components/ConfigNumberInput";
import type {
  RunPodCleanupPodAction,
  RunPodCloudType,
} from "../lib/runPod";
import {
  RUNPOD_GPU_COUNT_OPTIONS,
  findRunPodGpuOption,
  getRunPodGpuOptions,
} from "../lib/runPod";

interface RunPodSettingsPanelProps {
  apiKey: string;
  catalog: RunPodProviderCatalog | null;
  cleanupPod: RunPodCleanupPodAction;
  cloudType: RunPodCloudType;
  dataCenterId: string;
  gpuCount: number;
  gpuType: string;
  interruptible: boolean;
  onApiKeyChange: (value: string) => void;
  onCleanupPodChange: (value: RunPodCleanupPodAction) => void;
  onCloudTypeChange: (value: RunPodCloudType) => void;
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
  catalog,
  cleanupPod,
  cloudType,
  dataCenterId,
  gpuCount,
  gpuType,
  interruptible,
  onApiKeyChange,
  onCleanupPodChange,
  onCloudTypeChange,
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
  const cloudLabel = cloudType === "SECURE" ? "Secure Cloud" : "Community Cloud";
  const datacenterLabel = dataCenterId.trim() || "Any available";
  const capacityLabel = interruptible ? "Interruptible capacity" : "Standard capacity";
  const gpuOptions = getRunPodGpuOptions(catalog, gpuType);
  const selectedGpu = findRunPodGpuOption(catalog, gpuType);
  const selectedGpuLabel = selectedGpu?.display_name ?? (gpuType.trim() || "Loading GPU target");
  const selectedGpuMemory =
    selectedGpu?.memory_gb != null ? `${selectedGpu.memory_gb} GB VRAM each` : "VRAM not catalogued";
  const gpuCountOptions = Array.from(new Set<number>([...RUNPOD_GPU_COUNT_OPTIONS, gpuCount])).sort(
    (left, right) => left - right
  );
  const groupedGpuOptions = Array.from(
    gpuOptions.reduce((groups, option) => {
      const label = option.memory_gb != null ? `${option.memory_gb} GB VRAM` : "Other supported GPUs";
      const entries = groups.get(label) ?? [];
      entries.push(option);
      groups.set(label, entries);
      return groups;
    }, new Map<string, typeof gpuOptions>())
  ).sort(([leftLabel], [rightLabel]) => {
    const leftWeight = leftLabel === "Other supported GPUs" ? Number.POSITIVE_INFINITY : Number.parseInt(leftLabel, 10);
    const rightWeight = rightLabel === "Other supported GPUs" ? Number.POSITIVE_INFINITY : Number.parseInt(rightLabel, 10);
    return leftWeight - rightWeight;
  });

  return (
    <div className="settingsStack">
      <div className="statusGrid runPodStatusGrid">
        <div className="statusCard">
          <div className="statusCardIcon"><FiCheckCircle /></div>
          <div>
            <div className="statusCardTitle">RunPod access</div>
            <div className="statusCardValue">{keySource}</div>
            <div className="statusCardDetail">{validationMessage ?? "Keys are sent only with launch and validation requests."}</div>
          </div>
        </div>
        <div className="statusCard">
          <div className="statusCardIcon"><FiCpu /></div>
          <div>
            <div className="statusCardTitle">GPU target</div>
            <div className="statusCardValue">{selectedGpuLabel}</div>
            <div className="statusCardDetail">
              {gpuCount} GPU{gpuCount === 1 ? "" : "s"} · {selectedGpuMemory}
            </div>
          </div>
        </div>
        <div className="statusCard">
          <div className="statusCardIcon"><FiMapPin /></div>
          <div>
            <div className="statusCardTitle">Provisioning</div>
            <div className="statusCardValue">{cloudLabel}</div>
            <div className="statusCardDetail">
              {datacenterLabel} · {volumeSizeGb} GB workspace volume
            </div>
          </div>
        </div>
        <div className="statusCard">
          <div className="statusCardIcon"><FiAlertTriangle /></div>
          <div>
            <div className="statusCardTitle">Lifecycle</div>
            <div className="statusCardValue">{cleanupLabel}</div>
            <div className="statusCardDetail">
              {capacityLabel}
            </div>
          </div>
        </div>
      </div>

      <div className="runPodSettingsSection">
        <div className="settingsGroupHeader">
          <h3>Access</h3>
        </div>
        <div className="runPodCredentialRow">
          <label className="fieldLabel runPodApiKeyField">
            <span>RunPod API key</span>
            <input
              className="textInput"
              type="password"
              value={apiKey}
              placeholder={status?.source === "environment" ? "Using environment key" : "Paste RunPod API key"}
              onChange={(event) => onApiKeyChange(event.target.value)}
            />
          </label>
          <button
            type="button"
            className="secondaryButton runPodValidateButton"
            onClick={onValidateKey}
            disabled={validationLoading || apiKey.trim() === ""}
          >
            <FiCheckCircle aria-hidden="true" />
            {validationLoading ? "Validating..." : "Validate key"}
          </button>
          <span className={`pillBadge ${status?.configured ? "tone-good" : "tone-neutral"}`}>
            {status?.configured ? `Key ready (${status.source})` : "Key required"}
          </span>
        </div>
      </div>

      <div className="runPodSettingsSection">
        <div className="settingsGroupHeader">
          <h3>Pod configuration</h3>
        </div>
        <div className="fieldGrid trainingSettingsCompactGrid">
          <label className="fieldLabel fullWidthField">
            <span>GPU target</span>
            <select
              className="selectInput"
              value={gpuType}
              disabled={gpuOptions.length === 0}
              onChange={(event) => onGpuTypeChange(event.target.value)}
            >
              {gpuOptions.length === 0 ? <option value="">Loading GPU choices</option> : null}
              {groupedGpuOptions.map(([label, options]) => (
                <optgroup key={label} label={label}>
                  {options.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.display_name}
                      {option.memory_gb != null ? ` · ${option.memory_gb} GB` : ""}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </label>
          <label className="fieldLabel">
            <span>GPU count</span>
            <select
              className="selectInput"
              value={gpuCount}
              onChange={(event) => onGpuCountChange(Number(event.target.value))}
            >
              {gpuCountOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label className="fieldLabel">
            <span>Cloud</span>
            <select className="selectInput" value={cloudType} onChange={(event) => onCloudTypeChange(event.target.value as RunPodCloudType)}>
              <option value="SECURE">Secure Cloud</option>
              <option value="COMMUNITY">Community Cloud</option>
            </select>
          </label>
          <label className="fieldLabel">
            <span>Workspace volume GB</span>
            <ConfigNumberInput value={volumeSizeGb} onCommit={onVolumeSizeGbChange} />
          </label>
        </div>
      </div>

      <div className="runPodSettingsSection">
        <div className="settingsGroupHeader">
          <h3>Lifecycle</h3>
        </div>
        <div className="fieldGrid trainingSettingsCompactGrid">
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
      </div>

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
    </div>
  );
}
