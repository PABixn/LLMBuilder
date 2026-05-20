import { FiExternalLink, FiSave } from "react-icons/fi";

import { formatBytes } from "../../../lib/workspaceAssets";
import {
  getSimpleModelPreset,
  type SimpleModelPreset,
  SIMPLE_MODEL_PRESETS,
  targetVocabForPresetDataset,
} from "../lib/modelPresets";
import type { SimpleDatasetSource, SimpleModeController } from "../types";

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

const DATASET_LABELS = {
  starter: "Starter data",
  upload: "Upload data",
  streaming: "Streaming data",
} satisfies Record<SimpleDatasetSource, string>;

const TRAINING_PROFILE_LABELS = {
  quick: "Quick",
  balanced: "Balanced",
  longer: "Longer",
};

const EXECUTION_LABELS = {
  local: "Local",
  runpod_pod: "RunPod-ready",
};

function datasetForPresetSelection(
  currentDatasetSource: SimpleDatasetSource,
  localFileCount: number,
  preset: SimpleModelPreset
): SimpleDatasetSource {
  if (currentDatasetSource === "upload" && localFileCount > 0) {
    return "upload";
  }
  return preset.defaultDatasetSource;
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
                  updateFlow((current) => {
                    const datasetSource = datasetForPresetSelection(
                      current.datasetSource,
                      current.localTrainFiles.length,
                      item
                    );
                    return {
                      ...current,
                      presetId: item.id,
                      datasetSource,
                      trainingProfile: item.defaultTrainingProfile,
                      executionKind: item.defaultExecutionKind,
                      targetContextLength: item.defaultContextLength,
                      targetVocabSize: targetVocabForPresetDataset(item.id, datasetSource),
                    };
                  })
                }
              >
                <span className="simplePresetHead">
                  <strong>{item.name}</strong>
                  <span>{item.relativeSize}</span>
                </span>
                <span>{item.bestUse}</span>
                <span className="simplePresetMeta">
                  <span>{item.defaultContextLength.toLocaleString()} ctx</span>
                  <span>{item.headLayout}</span>
                  <span>{DATASET_LABELS[item.defaultDatasetSource]}</span>
                  <span>{TRAINING_PROFILE_LABELS[item.defaultTrainingProfile]}</span>
                  <span>{EXECUTION_LABELS[item.defaultExecutionKind]}</span>
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
            <span className="simpleStatusPill is-completed">Created</span>
          ) : null}
        </div>

        <label className="fieldLabel">
          <span>Model name</span>
          <input
            value={flow.modelName}
            onChange={(event) => {
              const modelName = event.currentTarget.value;
              updateFlow((current) => ({
                ...current,
                modelName,
              }));
            }}
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
              onChange={(event) => {
                const targetVocabSize = Math.max(1, Number(event.currentTarget.value) || 1);
                updateFlow((current) => ({
                  ...current,
                  targetVocabSize,
                }));
              }}
            />
          </label>

          <label className="fieldLabel">
            <span>Context length</span>
            <select
              value={flow.targetContextLength}
              onChange={(event) => {
                const targetContextLength = Number(event.currentTarget.value);
                updateFlow((current) => ({
                  ...current,
                  targetContextLength,
                }));
              }}
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

        <div className="simpleOutputBox">
          <strong>{preset.intent}</strong>
          <span>{preset.honestyNote}</span>
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
      </div>
    </div>
  );
}
