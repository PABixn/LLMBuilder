import { useMemo, useState } from "react";
import {
  FiChevronDown,
  FiCloud,
  FiCpu,
  FiExternalLink,
  FiHardDrive,
  FiLayers,
  FiSave,
  FiSearch,
} from "react-icons/fi";

import { formatBytes } from "../../../lib/workspaceAssets";
import { StatusCard } from "../../studio/components/primitives";
import {
  estimatePresetBf16MemoryBytes,
  estimatePresetParameterCount,
  getSimpleModelPreset,
  SIMPLE_PRESET_ARCHITECTURE_TYPES,
  SIMPLE_PRESET_SIZE_GROUPS,
  SIMPLE_PRESET_TRAINING_TARGETS,
  type SimpleModelPreset,
  type SimplePresetArchitectureType,
  type SimplePresetSize,
  type SimplePresetTrainingTarget,
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

type PresetFilterValue<T extends string> = T | "all";

function formatEstimatedParameterCount(value: number): string {
  return `~${formatParameterCount(value)}`;
}

function sizeLabelFor(size: SimplePresetSize): string {
  return SIMPLE_PRESET_SIZE_GROUPS.find((group) => group.id === size)?.label ?? size;
}

function architectureTypeLabelFor(type: SimplePresetArchitectureType): string {
  return (
    SIMPLE_PRESET_ARCHITECTURE_TYPES.find((option) => option.id === type)?.label ??
    type.toUpperCase()
  );
}

function trainingTargetLabelFor(target: SimplePresetTrainingTarget): string {
  return (
    SIMPLE_PRESET_TRAINING_TARGETS.find((option) => option.id === target)?.label ??
    target
  );
}

function presetMatchesQuery(preset: SimpleModelPreset, query: string): boolean {
  if (!query) {
    return true;
  }
  const haystack = [
    preset.name,
    preset.intent,
    preset.bestUse,
    preset.headLayout,
    preset.normActivationLabel,
    preset.hardwareTier,
    preset.relativeSize,
    preset.architectureType,
    preset.trainingTarget,
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(query);
}

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
  const selectedEstimate = estimatePresetParameterCount(preset, {
    vocabSize: flow.targetVocabSize,
    contextLength: flow.targetContextLength,
  });
  const selectedBf16Bytes =
    selectedAnalysis?.parameter_memory_bytes_bf16 ??
    estimatePresetBf16MemoryBytes(preset, {
      vocabSize: flow.targetVocabSize,
      contextLength: flow.targetContextLength,
    });
  const selectedAnalysisError = modelStep.analysisErrorsByPresetId[preset.id] ?? null;
  const [parameterBreakdownMode, setParameterBreakdownMode] =
    useState<ParameterBreakdownMode>("all");
  const [sizeFilter, setSizeFilter] =
    useState<PresetFilterValue<SimplePresetSize>>("all");
  const [architectureTypeFilter, setArchitectureTypeFilter] =
    useState<PresetFilterValue<SimplePresetArchitectureType>>("all");
  const [trainingTargetFilter, setTrainingTargetFilter] =
    useState<PresetFilterValue<SimplePresetTrainingTarget>>("all");
  const [collapsedSizeGroups, setCollapsedSizeGroups] = useState<
    Partial<Record<SimplePresetSize, boolean>>
  >({});
  const [query, setQuery] = useState("");
  const normalizedQuery = query.trim().toLowerCase();
  const groupedPresets = useMemo(
    () =>
      SIMPLE_PRESET_SIZE_GROUPS.map((group) => ({
        group,
        presets: SIMPLE_MODEL_PRESETS.filter(
          (item) =>
            item.relativeSize === group.id &&
            (sizeFilter === "all" || item.relativeSize === sizeFilter) &&
            (architectureTypeFilter === "all" ||
              item.architectureType === architectureTypeFilter) &&
            (trainingTargetFilter === "all" ||
              item.trainingTarget === trainingTargetFilter) &&
            presetMatchesQuery(item, normalizedQuery)
        ),
      })).filter((group) => group.presets.length > 0),
    [architectureTypeFilter, normalizedQuery, sizeFilter, trainingTargetFilter]
  );
  const filteredPresetCount = groupedPresets.reduce(
    (sum, group) => sum + group.presets.length,
    0
  );
  const selectPreset = (item: SimpleModelPreset) => {
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
    });
  };
  const toggleSizeGroup = (groupId: SimplePresetSize) => {
    setCollapsedSizeGroups((current) => ({
      ...current,
      [groupId]: !current[groupId],
    }));
  };
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
            <h3>Architecture templates</h3>
          </div>
          <a className="buttonGhost buttonSmall" href="/studio">
            <FiExternalLink aria-hidden="true" /> Expert
          </a>
        </div>

        <div className="simplePresetToolbar">
          <label className="simplePresetSearch">
            <FiSearch aria-hidden="true" />
            <input
              value={query}
              onChange={(event) => setQuery(event.currentTarget.value)}
              placeholder="Search architectures"
            />
          </label>

          <div className="simplePresetFilters" aria-label="Architecture filters">
            <div className="simplePresetFilterGroup">
              <span>Size</span>
              <div className="simpleFilterChips">
                <button
                  type="button"
                  className={sizeFilter === "all" ? "is-selected" : ""}
                  aria-pressed={sizeFilter === "all"}
                  onClick={() => setSizeFilter("all")}
                >
                  All
                </button>
                {SIMPLE_PRESET_SIZE_GROUPS.map((group) => (
                  <button
                    key={group.id}
                    type="button"
                    className={sizeFilter === group.id ? "is-selected" : ""}
                    aria-pressed={sizeFilter === group.id}
                    onClick={() => setSizeFilter(group.id)}
                  >
                    {group.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="simplePresetFilterGroup">
              <span>Type</span>
              <div className="simpleFilterChips">
                <button
                  type="button"
                  className={architectureTypeFilter === "all" ? "is-selected" : ""}
                  aria-pressed={architectureTypeFilter === "all"}
                  onClick={() => setArchitectureTypeFilter("all")}
                >
                  All
                </button>
                {SIMPLE_PRESET_ARCHITECTURE_TYPES.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    className={architectureTypeFilter === option.id ? "is-selected" : ""}
                    aria-pressed={architectureTypeFilter === option.id}
                    title={option.description}
                    onClick={() => setArchitectureTypeFilter(option.id)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="simplePresetFilterGroup">
              <span>Target</span>
              <div className="simpleFilterChips">
                <button
                  type="button"
                  className={trainingTargetFilter === "all" ? "is-selected" : ""}
                  aria-pressed={trainingTargetFilter === "all"}
                  onClick={() => setTrainingTargetFilter("all")}
                >
                  All
                </button>
                {SIMPLE_PRESET_TRAINING_TARGETS.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    className={trainingTargetFilter === option.id ? "is-selected" : ""}
                    aria-pressed={trainingTargetFilter === option.id}
                    title={option.description}
                    onClick={() => setTrainingTargetFilter(option.id)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="simplePresetToolbarMeta">
            {filteredPresetCount} architecture
            {filteredPresetCount === 1 ? "" : "s"} shown
          </div>
        </div>

        <div className="simplePresetSelectionBox">
          {groupedPresets.length > 0 ? (
            groupedPresets.map(({ group, presets }) => (
              <section key={group.id} className="simplePresetGroup">
                <button
                  type="button"
                  className="simplePresetGroupHeader"
                  aria-expanded={!collapsedSizeGroups[group.id]}
                  onClick={() => toggleSizeGroup(group.id)}
                >
                  <span>
                    <FiChevronDown aria-hidden="true" />
                    <strong>{group.label}</strong>
                  </span>
                  <span>
                    {presets.length} · {group.target}
                  </span>
                </button>
                {!collapsedSizeGroups[group.id] ? (
                  <div className="simplePresetGrid">
                    {presets.map((item) => {
                      const analysis = modelStep.analysisByPresetId[item.id]?.analysis ?? null;
                      const parameterLabel = analysis
                        ? formatParameterCount(analysis.total_parameters)
                        : formatEstimatedParameterCount(estimatePresetParameterCount(item));
                      return (
                        <button
                          key={item.id}
                          type="button"
                          className={`simplePresetCard${
                            item.id === flow.presetId ? " is-selected" : ""
                          }`}
                          title={item.bestUse}
                          onClick={() => selectPreset(item)}
                        >
                          <span className="simplePresetHead">
                            <strong>{item.name}</strong>
                            <span className="simplePresetEstimate">
                              <strong>{parameterLabel}</strong>
                            </span>
                          </span>
                          <span className="simplePresetMeta">
                            <span>{architectureTypeLabelFor(item.architectureType)}</span>
                            <span>{trainingTargetLabelFor(item.trainingTarget)}</span>
                            <span>{sizeLabelFor(item.relativeSize)}</span>
                          </span>
                        </button>
                      );
                    })}
                  </div>
                ) : null}
              </section>
            ))
          ) : (
            <div className="simplePresetEmpty">
              No architecture templates match the current filters.
            </div>
          )}
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
            value={formatCompactCount(selectedAnalysis?.total_parameters ?? selectedEstimate)}
            detail={selectedAnalysis ? "Backend" : "Catalog"}
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
                  <p>
                    {selectedAnalysisError ??
                      "Backend parameter breakdown is not loaded for this template yet."}
                  </p>
                </>
              )
            }
          />
          <StatusCard
            title="BF16 weights"
            value={
              selectedAnalysis
                ? formatBytes(selectedAnalysis.parameter_memory_bytes_bf16)
                : formatBytes(selectedBf16Bytes)
            }
            detail={selectedAnalysis ? "Weight memory" : "Weights"}
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
                  <span>
                    <strong>{architectureTypeLabelFor(preset.architectureType)}</strong>
                    <small>Type</small>
                  </span>
                  <span>
                    <strong>
                      {preset.trainingTarget === "runpod" ? "RunPod" : "Local"}
                    </strong>
                    <small>Target</small>
                  </span>
                </div>
                <p>{preset.honestyNote}</p>
              </div>
            }
          />
        </div>

        {preset.hardwareWarning ? (
          <div className="inlineNotice tone-info">
            {preset.trainingTarget === "runpod" ? (
              <FiCloud aria-hidden="true" />
            ) : (
              <FiCpu aria-hidden="true" />
            )}
            <span>{preset.hardwareWarning}</span>
          </div>
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
