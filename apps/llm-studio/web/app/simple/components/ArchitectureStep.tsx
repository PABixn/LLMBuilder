import { useState } from "react";
import { FiCpu, FiHardDrive, FiLayers, FiSave, FiExternalLink } from "react-icons/fi";

import { formatBytes } from "../../../lib/workspaceAssets";
import { StatusCard } from "../../studio/components/primitives";
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

type ParameterBreakdownMode = "all" | "trainable";

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

function formatCompactCount(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Analyzing";
  }
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function formatPercentage(value: number): string {
  if (!Number.isFinite(value) || value <= 0) {
    return "0%";
  }
  if (value >= 10) {
    return `${value.toFixed(1)}%`;
  }
  if (value >= 1) {
    return `${value.toFixed(2)}%`;
  }
  return `${value.toFixed(3)}%`;
}

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
  const [parameterBreakdownMode, setParameterBreakdownMode] =
    useState<ParameterBreakdownMode>("all");
  const parameterBreakdownTotal = selectedAnalysis
    ? parameterBreakdownMode === "trainable"
      ? selectedAnalysis.trainable_parameters
      : selectedAnalysis.total_parameters
    : 0;
  const parameterBreakdownEntries = selectedAnalysis
    ? selectedAnalysis.parameter_breakdown
        .map((entry) => {
          const displayCount =
            parameterBreakdownMode === "trainable"
              ? entry.trainable_parameters
              : entry.parameters;
          const displayPercentage =
            parameterBreakdownMode === "trainable"
              ? entry.trainable_percentage
              : entry.percentage;
          return {
            ...entry,
            displayCount,
            displayPercentage,
          };
        })
        .filter((entry) => entry.displayCount > 0)
        .sort(
          (entryA, entryB) =>
            entryB.displayCount - entryA.displayCount ||
            entryA.label.localeCompare(entryB.label)
        )
    : [];
  const parameterBreakdownShownCount = parameterBreakdownEntries.reduce(
    (sum, entry) => sum + entry.displayCount,
    0
  );
  const parameterBreakdownCoverage =
    parameterBreakdownTotal > 0
      ? (parameterBreakdownShownCount / parameterBreakdownTotal) * 100
      : 0;
  const hasTrainableParameters = (selectedAnalysis?.trainable_parameters ?? 0) > 0;

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

        <div className="statusGrid simpleArchitectureStatsGrid">
          <StatusCard
            title="Parameters"
            value={formatCompactCount(selectedAnalysis?.total_parameters)}
            detail="Backend estimate"
            tone="good"
            icon={<FiLayers />}
            tooltipLabel="Parameter breakdown by layer type"
            tooltipContent={
              selectedAnalysis ? (
                <div className="parameterBreakdownTooltip">
                  <div className="parameterBreakdownHeaderRow">
                    <div className="parameterBreakdownHeader">
                      <strong>Parameter breakdown</strong>
                      <span>
                        {parameterBreakdownEntries.length} layer type
                        {parameterBreakdownEntries.length === 1 ? "" : "s"}
                      </span>
                    </div>
                    <div
                      className="parameterBreakdownToggleGroup"
                      role="group"
                      aria-label="Parameter breakdown mode"
                    >
                      <button
                        type="button"
                        className={`parameterBreakdownToggle${
                          parameterBreakdownMode === "all" ? " isActive" : ""
                        }`}
                        aria-pressed={parameterBreakdownMode === "all"}
                        onClick={() => setParameterBreakdownMode("all")}
                      >
                        All
                      </button>
                      <button
                        type="button"
                        className={`parameterBreakdownToggle${
                          parameterBreakdownMode === "trainable" ? " isActive" : ""
                        }`}
                        aria-pressed={parameterBreakdownMode === "trainable"}
                        onClick={() => setParameterBreakdownMode("trainable")}
                        disabled={!hasTrainableParameters}
                      >
                        Trainable
                      </button>
                    </div>
                  </div>
                  <div className="parameterBreakdownSubhead">
                    {formatCompactCount(parameterBreakdownShownCount)} /{" "}
                    {formatCompactCount(parameterBreakdownTotal)}{" "}
                    {parameterBreakdownMode === "trainable" ? "trainable" : "all"} params ·{" "}
                    {formatPercentage(parameterBreakdownCoverage)} coverage
                  </div>
                  {parameterBreakdownEntries.length > 0 ? (
                    <div className="parameterBreakdownList">
                      {parameterBreakdownEntries.map((entry) => {
                        const widthPercent = Math.min(
                          100,
                          Math.max(entry.displayPercentage, 2)
                        );
                        return (
                          <div key={entry.key} className="parameterBreakdownRow">
                            <div className="parameterBreakdownRowHead">
                              <span className="parameterBreakdownLabel">{entry.label}</span>
                              <span className="parameterBreakdownValue">
                                {formatCompactCount(entry.displayCount)}
                              </span>
                            </div>
                            <div className="parameterBreakdownTrack" aria-hidden>
                              <span
                                className="parameterBreakdownFill"
                                style={{ width: `${widthPercent}%` }}
                              />
                            </div>
                            <div className="parameterBreakdownMeta">
                              <span>{formatPercentage(entry.displayPercentage)}</span>
                              <span>
                                {entry.module_count} module
                                {entry.module_count === 1 ? "" : "s"}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="parameterBreakdownEmpty">
                      {parameterBreakdownMode === "trainable"
                        ? "No trainable parameters are currently enabled in this model."
                        : "No parameterized layers were detected in the current model."}
                    </div>
                  )}
                </div>
              ) : (
                <>
                  <strong>Parameter breakdown</strong>
                  <p>Create or refresh the architecture to load backend parameter analysis.</p>
                </>
              )
            }
          />
          <StatusCard
            title="BF16 weights"
            value={
              selectedAnalysis
                ? formatBytes(selectedAnalysis.parameter_memory_bytes_bf16)
                : "Analyzing"
            }
            detail="Weight memory"
            tone="neutral"
            icon={<FiHardDrive />}
          />
          <StatusCard
            title="Block family"
            value={preset.normActivationLabel}
            detail={`${preset.blockCount} blocks · ${preset.nEmbeddings} hidden`}
            tone="neutral"
            icon={<FiCpu />}
            tooltipLabel="Architecture details"
            tooltipContent={
              <div className="simpleArchitectureTooltip">
                <strong>{preset.normActivationLabel}</strong>
                <p>{preset.intent}</p>
                <div className="simpleArchitectureTooltipGrid">
                  <span>
                    <strong>{preset.blockCount}</strong>
                    <small>Blocks</small>
                  </span>
                  <span>
                    <strong>{preset.nEmbeddings}</strong>
                    <small>Hidden width</small>
                  </span>
                  <span>
                    <strong>
                      {preset.nHead}/{preset.nKvHead}
                    </strong>
                    <small>Query/KV heads</small>
                  </span>
                  <span>
                    <strong>{flow.targetContextLength.toLocaleString()}</strong>
                    <small>Context</small>
                  </span>
                </div>
                <p>{preset.honestyNote}</p>
              </div>
            }
          />
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
