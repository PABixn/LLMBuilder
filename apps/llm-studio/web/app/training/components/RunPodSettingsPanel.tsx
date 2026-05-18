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
import { FieldLabelText, HelpTooltip, InfoTooltip } from "../../shared/components/HelpTooltip";

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
      ? "Key not validated"
      : "No key configured";
  const cleanupLabel =
    cleanupPod === "delete_after_sync"
      ? "Delete after sync"
      : cleanupPod === "stop_after_sync"
        ? "Stop pod after sync"
        : "Keep pod running";
  const cloudLabel = cloudType === "SECURE" ? "Secure Cloud" : "Community Cloud";
  const datacenterLabel = dataCenterId.trim() || "Any available";
  const capacityLabel = interruptible ? "Interruptible capacity" : "Standard capacity";
  const gpuOptions = getRunPodGpuOptions(catalog, gpuType);
  const selectedGpu = findRunPodGpuOption(catalog, gpuType);
  const selectedGpuLabel = selectedGpu?.display_name ?? (gpuType.trim() || "Loading GPU");
  const selectedGpuMemory =
    selectedGpu?.memory_gb != null ? `${selectedGpu.memory_gb} GB VRAM each` : "VRAM unknown";
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
            <div className="statusCardDetail">{validationMessage ?? "Keys are used only for RunPod requests."}</div>
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
              {datacenterLabel} · {volumeSizeGb} GB volume
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
          <h3>
            Access
            <InfoTooltip label="RunPod access explanation" align="left" width="wide">
              <strong>RunPod access</strong>
              <p>
                The API key is sent only to the backend endpoints that validate or create
                RunPod resources. If an environment key is configured, you can leave this blank.
              </p>
            </InfoTooltip>
          </h3>
        </div>
        <div className="runPodCredentialRow">
          <label className="fieldLabel runPodApiKeyField">
            <FieldLabelText tooltipLabel="RunPod API key explanation" tooltip="A RunPod API key lets this app create and inspect pods for your training runs. Validate it before launch so the app can verify access without starting training.">
              RunPod API key
            </FieldLabelText>
            <input
              className="textInput"
              type="password"
              value={apiKey}
              placeholder={status?.source === "environment" ? "Using environment key" : "Paste RunPod API key"}
              onChange={(event) => onApiKeyChange(event.target.value)}
            />
          </label>
          <HelpTooltip label="Validate RunPod key" content="Checks that the key can reach RunPod and updates the access status. This does not create a pod or start billing.">
            <button
              type="button"
              className="secondaryButton runPodValidateButton"
              onClick={onValidateKey}
              disabled={validationLoading || apiKey.trim() === ""}
            >
              <FiCheckCircle aria-hidden="true" />
              {validationLoading ? "Validating..." : "Validate key"}
            </button>
          </HelpTooltip>
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
            <FieldLabelText tooltipLabel="GPU target explanation" tooltip="The RunPod GPU model used for the training pod. More VRAM supports larger micro batches, longer sequences, or larger models.">
              GPU target
            </FieldLabelText>
            <select
              className="selectInput"
              value={gpuType}
              disabled={gpuOptions.length === 0}
              onChange={(event) => onGpuTypeChange(event.target.value)}
            >
              {gpuOptions.length === 0 ? <option value="">Loading GPUs</option> : null}
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
            <FieldLabelText tooltipLabel="GPU count explanation" tooltip="Number of GPUs requested for the pod. The trainer can use multiple GPUs, but availability and cost increase with this number.">
              GPU count
            </FieldLabelText>
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
            <FieldLabelText tooltipLabel="RunPod cloud explanation" tooltip="Secure Cloud uses RunPod-vetted infrastructure. Community Cloud can be cheaper or more available, but capacity and reliability vary by host.">
              Cloud
            </FieldLabelText>
            <select className="selectInput" value={cloudType} onChange={(event) => onCloudTypeChange(event.target.value as RunPodCloudType)}>
              <option value="SECURE">Secure Cloud</option>
              <option value="COMMUNITY">Community Cloud</option>
            </select>
          </label>
          <label className="fieldLabel">
            <FieldLabelText tooltipLabel="Volume size explanation" tooltip="Disk space attached to the pod for datasets, checkpoints, logs, and synced artifacts. Increase it when checkpoints or datasets are large.">
              Volume GB
            </FieldLabelText>
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
            <FieldLabelText tooltipLabel="Pod cleanup explanation" tooltip="What the app does after artifacts sync. Delete minimizes cost, stop keeps the pod image available but stopped, and keep running leaves billing active.">
              Pod cleanup
            </FieldLabelText>
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
          <span className="fieldLabelText">
            <span>Use interruptible capacity</span>
            <InfoTooltip label="Interruptible capacity explanation" width="wide">
              <strong>Interruptible capacity</strong>
              <p>
                Requests cheaper capacity that RunPod may reclaim. Use it for experiments
                that can restart from checkpoints; avoid it for runs that must finish uninterrupted.
              </p>
            </InfoTooltip>
          </span>
        </label>
      </div>

      {interruptible ? (
        <p className="settingsGroupHint">
          Interruptible pods can stop during training.
        </p>
      ) : null}
      {cleanupPod === "keep" ? (
        <p className="settingsGroupHint">
          Keeping the pod running can keep billing active.
        </p>
      ) : null}
    </div>
  );
}
