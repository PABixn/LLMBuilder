"use client";

import { useCallback, useEffect, useLayoutEffect, useState } from "react";

import {
  DEFAULT_SIMPLE_FLOW_STATE,
  SIMPLE_FLOW_STORAGE_KEY,
  SIMPLE_FLOW_VERSION,
} from "../constants";
import { isSimpleModelPresetId, targetVocabForPresetDataset } from "../lib/modelPresets";
import { normalizeSimpleStreamingSelection } from "../lib/streamingDatasets";
import type {
  SimpleDatasetSource,
  SimpleExecutionKind,
  SimpleFlowState,
  SimpleLocalTrainFile,
  SimpleStepId,
  SimpleTrainingProfile,
} from "../types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function asPositiveInteger(value: unknown, fallback: number): number {
  const parsed = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  return Number.isFinite(parsed) && parsed > 0 ? Math.trunc(parsed) : fallback;
}

function asNullableString(value: unknown): string | null {
  return typeof value === "string" && value.trim() !== "" ? value : null;
}

function asDatasetSource(value: unknown): SimpleDatasetSource {
  return value === "starter" || value === "upload" || value === "streaming"
    ? value
    : DEFAULT_SIMPLE_FLOW_STATE.datasetSource;
}

function asTrainingProfile(value: unknown): SimpleTrainingProfile {
  return value === "quick" || value === "balanced" || value === "longer"
    ? value
    : DEFAULT_SIMPLE_FLOW_STATE.trainingProfile;
}

function asExecutionKind(value: unknown): SimpleExecutionKind {
  return value === "runpod_pod" ? "runpod_pod" : "local";
}

function asPresetId(value: unknown): string {
  return isSimpleModelPresetId(value) ? value : DEFAULT_SIMPLE_FLOW_STATE.presetId;
}

function asSimpleStep(value: unknown): SimpleStepId | null {
  return value === "architecture" ||
    value === "tokenizer" ||
    value === "training" ||
    value === "inference"
    ? value
    : null;
}

function parseLocalTrainFile(value: unknown): SimpleLocalTrainFile | null {
  if (!isRecord(value)) {
    return null;
  }
  const filePath = asString(value.filePath, "").trim();
  if (!filePath) {
    return null;
  }
  return {
    id: asString(value.id, `local-file-${filePath}`),
    fileName: asString(value.fileName, filePath.split("/").pop() ?? "Training file"),
    filePath,
    sizeBytes:
      typeof value.sizeBytes === "number" && Number.isFinite(value.sizeBytes) && value.sizeBytes >= 0
        ? value.sizeBytes
        : null,
    sizeChars:
      typeof value.sizeChars === "number" && Number.isFinite(value.sizeChars) && value.sizeChars >= 0
        ? value.sizeChars
        : null,
  };
}

export function parseSimpleFlowState(value: unknown): SimpleFlowState {
  if (!isRecord(value)) {
    return DEFAULT_SIMPLE_FLOW_STATE;
  }
  const presetId = asPresetId(value.presetId);
  const hasValidPresetId = isSimpleModelPresetId(value.presetId);
  const datasetSource = hasValidPresetId
    ? asDatasetSource(value.datasetSource)
    : DEFAULT_SIMPLE_FLOW_STATE.datasetSource;
  const recommendedVocabSize = targetVocabForPresetDataset(presetId, datasetSource);
  const localFiles = Array.isArray(value.localTrainFiles)
    ? value.localTrainFiles
        .map(parseLocalTrainFile)
        .filter((entry): entry is SimpleLocalTrainFile => entry !== null)
    : DEFAULT_SIMPLE_FLOW_STATE.localTrainFiles;
  const streamingSelection = normalizeSimpleStreamingSelection(
    value.streamingPrimaryDatasetId,
    value.streamingAdditionalDatasetIds
  );

  return {
    version: SIMPLE_FLOW_VERSION,
    presetId,
    modelName: asString(value.modelName, DEFAULT_SIMPLE_FLOW_STATE.modelName),
    targetVocabSize: hasValidPresetId
      ? asPositiveInteger(value.targetVocabSize, recommendedVocabSize)
      : recommendedVocabSize,
    targetContextLength: hasValidPresetId
      ? asPositiveInteger(value.targetContextLength, DEFAULT_SIMPLE_FLOW_STATE.targetContextLength)
      : DEFAULT_SIMPLE_FLOW_STATE.targetContextLength,
    projectId: asNullableString(value.projectId),
    tokenizerJobId: asNullableString(value.tokenizerJobId),
    trainingJobId: asNullableString(value.trainingJobId),
    datasetSource,
    localTrainFiles: localFiles,
    streamingPrimaryDatasetId: streamingSelection.primaryId,
    streamingAdditionalDatasetIds: streamingSelection.additionalIds,
    trainingProfile: hasValidPresetId
      ? asTrainingProfile(value.trainingProfile)
      : DEFAULT_SIMPLE_FLOW_STATE.trainingProfile,
    executionKind: hasValidPresetId
      ? asExecutionKind(value.executionKind)
      : DEFAULT_SIMPLE_FLOW_STATE.executionKind,
    checkpointValue: asString(
      value.checkpointValue,
      DEFAULT_SIMPLE_FLOW_STATE.checkpointValue
    ),
    lastCompletedStep: asSimpleStep(value.lastCompletedStep),
  };
}

function clearRestoredArtifacts(state: SimpleFlowState): SimpleFlowState {
  return {
    ...state,
    projectId: null,
    tokenizerJobId: null,
    trainingJobId: null,
    lastCompletedStep: null,
  };
}

function readStoredSimpleFlow(): SimpleFlowState {
  if (typeof window === "undefined") {
    return DEFAULT_SIMPLE_FLOW_STATE;
  }
  try {
    const raw = window.localStorage.getItem(SIMPLE_FLOW_STORAGE_KEY);
    return raw ? clearRestoredArtifacts(parseSimpleFlowState(JSON.parse(raw))) : DEFAULT_SIMPLE_FLOW_STATE;
  } catch {
    return DEFAULT_SIMPLE_FLOW_STATE;
  }
}

function writeStoredSimpleFlow(state: SimpleFlowState): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(SIMPLE_FLOW_STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Ignore local storage failures.
  }
}

export function useSimpleFlowPersistence() {
  const [flow, setFlow] = useState<SimpleFlowState>(DEFAULT_SIMPLE_FLOW_STATE);
  const [hydrated, setHydrated] = useState(false);
  const updateFlow = useCallback(
    (updater: (current: SimpleFlowState) => SimpleFlowState) => {
      setFlow((current) => parseSimpleFlowState(updater(current)));
    },
    []
  );
  const resetFlow = useCallback(() => setFlow(DEFAULT_SIMPLE_FLOW_STATE), []);

  useLayoutEffect(() => {
    setFlow(readStoredSimpleFlow());
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) {
      return;
    }
    writeStoredSimpleFlow(flow);
  }, [flow, hydrated]);

  return {
    flow,
    hydrated,
    setFlow,
    updateFlow,
    resetFlow,
  };
}
