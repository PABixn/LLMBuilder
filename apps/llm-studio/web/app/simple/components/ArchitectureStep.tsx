import { FiExternalLink, FiSave } from "react-icons/fi";

import { formatBytes } from "../../../lib/workspaceAssets";
import {
  getSimpleModelPreset,
  SIMPLE_MODEL_PRESETS,
  targetVocabForPresetDataset,
} from "../lib/modelPresets";
import type { SimpleModeController } from "../types";

interface ArchitectureStepProps {
  controller: SimpleModeController;
}

function formatParameterCount(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Analyzing";
  }
  if (value >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(2)}B params`;
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M params`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K params`;
  }
  return `${value.toLocaleString()} params`;
}

export function ArchitectureStep({ controller }: ArchitectureStepProps) {
  const { flow, modelStep, updateFlow } = controller;
  const preset = getSimpleModelPreset(flow.presetId);
  const selectedAnalysis = modelStep.selectedAnalysis?.analysis ?? null;

  return (
    <div className="simpleStepGrid">
      <div className="simplePanel">
        <div className="simplePanelHeader">
          <div>
            <p className="simpleEyebrow">Template</p>
            <h3>Choose a template</h3>
          </div>
          <a className="buttonGhost buttonSmall" href="/studio">
            <FiExternalLink aria-hidden="true" /> Expert
          </a>
        </div>

        <div className="simplePresetGrid">
          {SIMPLE_MODEL_PRESETS.map((item) => {
            const analysis = modelStep.analysisByPresetId[item.id]?.analysis ?? null;
            const analysisError = modelStep.analysisErrorsByPresetId[item.id];
            return (
              <button
                key={item.id}
                type="button"
                className={`simplePresetCard${item.id === flow.presetId ? " is-selected" : ""}`}
                onClick={() =>
                  updateFlow((current) => ({
                    ...current,
                    presetId: item.id,
                    targetContextLength: item.defaultContextLength,
                    targetVocabSize: targetVocabForPresetDataset(item.id, current.datasetSource),
                  }))
                }
              >
                <span className="simplePresetHead">
                  <strong>{item.name}</strong>
                  <span>{item.support}</span>
                </span>
                <span>{item.bestUse}</span>
                <span className="simplePresetMeta">
                  <span>{item.relativeSize}</span>
                  <span>{item.defaultContextLength.toLocaleString()} ctx</span>
                  <span>{item.headLayout}</span>
                </span>
                <span className="simplePresetEstimate">
                  {analysisError
                    ? "Analysis unavailable"
                    : formatParameterCount(analysis?.total_parameters)}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="simplePanel">
        <div className="simplePanelHeader">
          <div>
            <p className="simpleEyebrow">Architecture</p>
            <h3>Create architecture</h3>
          </div>
          {modelStep.project ? (
            <span className="simpleStatusPill is-completed">Saved</span>
          ) : null}
        </div>

        <label className="fieldLabel">
          <span>Model name</span>
          <input
            value={flow.modelName}
            onChange={(event) =>
              updateFlow((current) => ({
                ...current,
                modelName: event.currentTarget.value,
              }))
            }
          />
        </label>

        <div className="simpleFieldPair">
          <label className="fieldLabel">
            <span>Vocabulary target</span>
            <input
              type="number"
              min={128}
              step={128}
              value={flow.targetVocabSize}
              onChange={(event) =>
                updateFlow((current) => ({
                  ...current,
                  targetVocabSize: Math.max(1, Number(event.currentTarget.value) || 1),
                }))
              }
            />
          </label>

          <label className="fieldLabel">
            <span>Context length</span>
            <select
              value={flow.targetContextLength}
              onChange={(event) =>
                updateFlow((current) => ({
                  ...current,
                  targetContextLength: Number(event.currentTarget.value),
                }))
              }
            >
              {preset.contextLengthOptions.map((contextLength) => (
                <option key={contextLength} value={contextLength}>
                  {contextLength.toLocaleString()} tokens
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="simpleSummaryGrid">
          <span>
            <strong>{formatParameterCount(selectedAnalysis?.total_parameters)}</strong>
            <small>Backend estimate</small>
          </span>
          <span>
            <strong>
              {selectedAnalysis
                ? formatBytes(selectedAnalysis.parameter_memory_bytes_bf16)
                : "Analyzing"}
            </strong>
            <small>BF16 weights</small>
          </span>
          <span>
            <strong>{preset.normActivationLabel}</strong>
            <small>Block family</small>
          </span>
        </div>

        {preset.hardwareWarning ? (
          <div className="inlineNotice tone-info">{preset.hardwareWarning}</div>
        ) : null}
        {modelStep.projectError ? (
          <div className="inlineNotice tone-error">{modelStep.projectError}</div>
        ) : null}

        <div className="simpleActionRow">
          <button
            type="button"
            className="buttonPrimary"
            disabled={modelStep.creating}
            onClick={() => void modelStep.createArchitecture()}
          >
            <FiSave aria-hidden="true" />
            {modelStep.creating ? "Creating" : "Create architecture"}
          </button>
        </div>

        <details className="simpleDetails">
          <summary>Advanced details</summary>
          <p>{preset.honestyNote}</p>
          <pre>{JSON.stringify({ preset, target: { vocab: flow.targetVocabSize, context: flow.targetContextLength } }, null, 2)}</pre>
        </details>
      </div>
    </div>
  );
}
