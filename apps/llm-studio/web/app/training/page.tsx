"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  Suspense,
  startTransition,
  type ChangeEvent,
  type DragEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  FiActivity,
  FiAlertTriangle,
  FiArchive,
  FiBarChart2,
  FiCheckCircle,
  FiClock,
  FiCpu,
  FiDownload,
  FiFileText,
  FiLayers,
  FiMoon,
  FiPlay,
  FiPlus,
  FiRefreshCw,
  FiSearch,
  FiSun,
  FiTrash2,
  FiX,
  FiXCircle,
} from "react-icons/fi";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  fetchProject,
  fetchProjects,
  type ProjectDetail,
  type ProjectSummary,
} from "../../lib/api";
import { useThemeMode } from "../../lib/theme";
import {
  formatAge,
  formatBytes,
  formatDate,
} from "../../lib/workspaceAssets";
import {
  deleteTrainingJob,
  fetchTrainingCheckpoints,
  fetchTrainingConfigTemplates,
  fetchTrainingJob,
  fetchTrainingJobs,
  fetchTrainingLogs,
  fetchTrainingMetrics,
  fetchTrainingSamples,
  stopTrainingJob,
  trainingArtifactDownloadUrl,
  validateTrainingPreflight,
  type TrainingCheckpointEntry,
  type TrainingFixSuggestion,
  type TrainingIssue,
  type TrainingJob,
  type TrainingMetricPoint,
  type TrainingPreflightResponse,
  type TrainingSampleEntry,
  createTrainingJob,
} from "../../lib/trainingApi";
import {
  fetchLocalTrainFileStats,
  fetchTrainingJob as fetchTokenizerJob,
  fetchTrainingJobs as fetchTokenizerJobs,
  type TrainingJob as TokenizerTrainingJob,
  uploadTrainFile,
} from "../../lib/tokenizerLegacyApi";

type ToastLevel = "info" | "success" | "error";

interface ToastState {
  id: string;
  level: ToastLevel;
  title: string;
  body: string;
}

type AssetPickerKind = "project" | "tokenizer";
type DatasetSourceMode = "local_file" | "streaming_hf";
type FilterOperator = "==" | "!=" | ">" | ">=" | "<" | "<=" | "in" | "not in";
type WorkflowTarget = "model" | "tokenizer" | "training" | "dataset" | "preflight";
type MetricChartKey = "loss" | "lr" | "norm" | "tok_per_sec";

interface StreamingFilterFormState {
  id: string;
  column: string;
  operator: FilterOperator;
  value: string;
}

interface StreamingDatasetFormState {
  id: string;
  name: string;
  config: string;
  split: string;
  textColumns: string;
  weight: string;
  filters: StreamingFilterFormState[];
}

interface LocalTrainFileFormState {
  id: string;
  fileName: string;
  filePath: string;
  sizeBytes: number | null;
  sizeChars: number | null;
}

const FILTER_OPERATORS: FilterOperator[] = ["==", "!=", ">", ">=", "<", "<=", "in", "not in"];
const WEIGHT_SUM_EPSILON = 1e-9;
const WEIGHT_SCALE = 1_000_000;

const TRAINING_CONFIG_STORAGE_KEY = "llm-training-config-v1";
const DATALOADER_CONFIG_STORAGE_KEY = "llm-training-dataloader-v1";
const TRAINING_SELECTION_STORAGE_KEY = "llm-training-selection-v1";
const ACTIVE_RUN_STORAGE_KEY = "llm-training-active-run-v1";
const POLL_INTERVAL_MS = 1800;
const WORKFLOW_TARGET_HASH_MAP: Record<WorkflowTarget, string> = {
  model: "#settings-model",
  tokenizer: "#settings-tokenizer",
  training: "#settings-training",
  dataset: "#settings-dataset",
  preflight: "#settings-preflight",
};

function stripGeneratedUploadPrefix(value: string): string {
  const trimmed = value.trim();
  const separatorIndex = trimmed.indexOf("-");
  if (separatorIndex <= 0) {
    return trimmed;
  }
  const prefix = trimmed.slice(0, separatorIndex);
  if (!/^[0-9a-f]{12}$/i.test(prefix)) {
    return trimmed;
  }
  const stripped = trimmed.slice(separatorIndex + 1).trim();
  return stripped === "" ? trimmed : stripped;
}

function fileNameFromPath(value: string): string {
  const normalized = value.replaceAll("\\", "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] ?? "";
}

function compactWorkflowMessage(value: string): string {
  return value.replace(/(?:[A-Za-z]:)?[\\/][^\s]+/g, (path) => {
    const fileName = fileNameFromPath(path);
    return fileName || path;
  });
}

function makeStreamingFilterEntry(
  value?: Partial<StreamingFilterFormState>
): StreamingFilterFormState {
  return {
    id: value?.id ?? `filter-${Math.random().toString(36).slice(2, 10)}`,
    column: value?.column ?? "",
    operator: value?.operator ?? "==",
    value: value?.value ?? "",
  };
}

function makeStreamingDatasetEntry(
  value?: Partial<StreamingDatasetFormState>
): StreamingDatasetFormState {
  return {
    id: value?.id ?? `dataset-${Math.random().toString(36).slice(2, 10)}`,
    name: value?.name ?? "",
    config: value?.config ?? "",
    split: value?.split ?? "train",
    textColumns: value?.textColumns ?? "text",
    weight: value?.weight ?? "1",
    filters: value?.filters ?? [],
  };
}

function makeLocalTrainFileEntry(
  value?: Partial<LocalTrainFileFormState>
): LocalTrainFileFormState {
  const filePath = value?.filePath?.trim() ?? "";
  const fallbackFileName = stripGeneratedUploadPrefix(fileNameFromPath(filePath));
  const providedFileName = stripGeneratedUploadPrefix(value?.fileName?.trim() ?? "");
  return {
    id: value?.id ?? `train-file-${Math.random().toString(36).slice(2, 10)}`,
    fileName: providedFileName || fallbackFileName || "uploaded-file.txt",
    filePath,
    sizeBytes:
      typeof value?.sizeBytes === "number" &&
      Number.isFinite(value.sizeBytes) &&
      value.sizeBytes >= 0
        ? value.sizeBytes
        : null,
    sizeChars:
      typeof value?.sizeChars === "number" &&
      Number.isFinite(value.sizeChars) &&
      value.sizeChars >= 0
        ? value.sizeChars
        : typeof value?.sizeBytes === "number" &&
            Number.isFinite(value.sizeBytes) &&
            value.sizeBytes >= 0
          ? Math.trunc(value.sizeBytes)
          : null,
  };
}

function normalizeLocalTrainFiles(
  files: LocalTrainFileFormState[]
): LocalTrainFileFormState[] {
  const dedupedByPath = new Map<string, LocalTrainFileFormState>();
  files.forEach((entry) => {
    const filePath = entry.filePath.trim();
    if (filePath === "") {
      return;
    }
    dedupedByPath.set(filePath, {
      ...entry,
      filePath,
      fileName:
        stripGeneratedUploadPrefix(entry.fileName.trim()) ||
        stripGeneratedUploadPrefix(fileNameFromPath(filePath)),
      sizeBytes:
        typeof entry.sizeBytes === "number" && Number.isFinite(entry.sizeBytes) && entry.sizeBytes >= 0
          ? entry.sizeBytes
          : null,
      sizeChars:
        typeof entry.sizeChars === "number" && Number.isFinite(entry.sizeChars) && entry.sizeChars >= 0
          ? entry.sizeChars
          : null,
    });
  });
  return Array.from(dedupedByPath.values());
}

function extractTrainFilePaths(dataFiles: unknown): string[] {
  if (typeof dataFiles === "string") {
    const trimmed = dataFiles.trim();
    return trimmed === "" ? [] : [trimmed];
  }
  if (Array.isArray(dataFiles)) {
    return dataFiles
      .filter((entry) => typeof entry === "string")
      .map((entry) => String(entry).trim())
      .filter((entry) => entry !== "");
  }
  const record = asRecord(dataFiles);
  const trainField = record.train;
  if (typeof trainField === "string") {
    const trimmed = trainField.trim();
    return trimmed === "" ? [] : [trimmed];
  }
  if (Array.isArray(trainField)) {
    return trainField
      .filter((entry) => typeof entry === "string")
      .map((entry) => String(entry).trim())
      .filter((entry) => entry !== "");
  }
  return [];
}

function formatCharCount(value: number | null): string | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return null;
  }
  return new Intl.NumberFormat().format(Math.trunc(value));
}

function asFilterOperator(value: unknown): FilterOperator {
  return FILTER_OPERATORS.includes(value as FilterOperator)
    ? (value as FilterOperator)
    : "==";
}

function stringifyFilterValue(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (value === null) {
    return "null";
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function sanitizePositiveDecimalInput(value: string): string {
  const digitsAndDot = value.replace(/[^0-9.]/g, "");
  const firstDotIndex = digitsAndDot.indexOf(".");
  if (firstDotIndex === -1) {
    return digitsAndDot;
  }
  return `${digitsAndDot.slice(0, firstDotIndex + 1)}${digitsAndDot
    .slice(firstDotIndex + 1)
    .replace(/\./g, "")}`;
}

function parseWeightInput(value: string): number | null {
  const trimmed = value.trim();
  if (trimmed === "") {
    return null;
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  return parsed;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function formatWeight(value: number): string {
  const rounded = Math.round(clamp(value, 0, 1) * WEIGHT_SCALE) / WEIGHT_SCALE;
  return Number(rounded.toFixed(6)).toString();
}

function sanitizeWeightInput(value: string): string {
  const sanitized = sanitizePositiveDecimalInput(value);
  if (sanitized === "") {
    return "0";
  }
  if (sanitized === ".") {
    return "0.";
  }
  return parseWeightInput(sanitized) === null ? "0" : sanitized;
}

function sanitizePositiveIntegerInput(value: string): string {
  return value.replace(/[^0-9]/g, "");
}

function formatNumberInputValue(value: number): string {
  if (!Number.isFinite(value)) {
    return "";
  }
  const asText = String(value);
  if (!/[eE]/.test(asText)) {
    return asText;
  }
  return value.toLocaleString("en-US", {
    useGrouping: false,
    maximumFractionDigits: 20,
  });
}

function parseConfigNumberInput(
  value: string,
  mode: "integer" | "decimal"
): number | null {
  const trimmed = value.trim();
  if (trimmed === "" || trimmed === ".") {
    return null;
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  if (mode === "integer" && !Number.isInteger(parsed)) {
    return null;
  }
  return parsed;
}

function normalizeWeights(weights: number[]): number[] {
  if (weights.length === 0) {
    return [];
  }
  const safeWeights = weights.map((weight) => (Number.isFinite(weight) ? Math.max(0, weight) : 0));
  const total = safeWeights.reduce((sum, weight) => sum + weight, 0);
  if (total <= WEIGHT_SUM_EPSILON) {
    const shared = 1 / safeWeights.length;
    return safeWeights.map(() => shared);
  }
  return safeWeights.map((weight) => weight / total);
}

function normalizeWeightsWithLockedIndex(weights: number[], lockedIndex: number): number[] {
  if (weights.length === 0) {
    return [];
  }
  if (weights.length === 1) {
    return [1];
  }
  const safeWeights = weights.map((weight) => (Number.isFinite(weight) ? Math.max(0, weight) : 0));
  const normalized = new Array(safeWeights.length).fill(0);
  const lockedWeight = clamp(safeWeights[lockedIndex] ?? 0, 0, 1);
  normalized[lockedIndex] = lockedWeight;
  const otherIndexes = safeWeights.map((_, index) => index).filter((index) => index !== lockedIndex);
  const remaining = 1 - lockedWeight;
  if (remaining <= WEIGHT_SUM_EPSILON) {
    otherIndexes.forEach((index) => {
      normalized[index] = 0;
    });
    return normalized;
  }
  const totalOthers = otherIndexes.reduce((sum, index) => sum + safeWeights[index], 0);
  if (totalOthers <= WEIGHT_SUM_EPSILON) {
    const shared = remaining / otherIndexes.length;
    otherIndexes.forEach((index) => {
      normalized[index] = shared;
    });
  } else {
    otherIndexes.forEach((index) => {
      normalized[index] = (safeWeights[index] / totalOthers) * remaining;
    });
  }
  return normalized;
}

function normalizeStreamingDatasetWeights(
  datasets: StreamingDatasetFormState[],
  lockedId?: string,
  lockedRawWeight?: string
): StreamingDatasetFormState[] {
  if (datasets.length === 0) {
    return [];
  }
  if (datasets.length === 1) {
    return [{ ...datasets[0], weight: "1" }];
  }
  const lockedIndex =
    typeof lockedId === "string" ? datasets.findIndex((entry) => entry.id === lockedId) : -1;
  const weights = datasets.map((entry) => Math.max(0, parseWeightInput(entry.weight) ?? 0));
  if (lockedIndex >= 0 && typeof lockedRawWeight === "string") {
    weights[lockedIndex] = Math.max(0, parseWeightInput(lockedRawWeight) ?? 0);
  }
  const lockedRawParsed =
    typeof lockedRawWeight === "string" ? parseWeightInput(lockedRawWeight) : null;
  const shouldLock =
    lockedIndex >= 0 && lockedRawParsed !== null && lockedRawParsed <= 1;
  const normalizedWeights = shouldLock
    ? normalizeWeightsWithLockedIndex(weights, lockedIndex)
    : normalizeWeights(weights);
  const roundedWeights = normalizedWeights.map(
    (weight) => Math.round(weight * WEIGHT_SCALE) / WEIGHT_SCALE
  );
  const roundedSum = roundedWeights.reduce((sum, weight) => sum + weight, 0);
  const drift = 1 - roundedSum;
  const adjustmentIndex =
    lockedIndex >= 0
      ? lockedIndex
      : roundedWeights.reduce(
          (bestIndex, weight, index) =>
            weight > roundedWeights[bestIndex] ? index : bestIndex,
          0
        );
  roundedWeights[adjustmentIndex] = clamp(roundedWeights[adjustmentIndex] + drift, 0, 1);
  return datasets.map((entry, index) => ({
    ...entry,
    weight:
      shouldLock &&
      index === lockedIndex &&
      typeof lockedRawWeight === "string" &&
      lockedRawParsed !== null &&
      Math.abs(lockedRawParsed - roundedWeights[index]) <= WEIGHT_SUM_EPSILON
        ? lockedRawWeight
        : formatWeight(roundedWeights[index]),
  }));
}

function parseFilterValue(value: string, operator: FilterOperator, label: string): unknown {
  const trimmed = value.trim();
  if (trimmed === "") {
    throw new Error(`${label} value is required`);
  }
  if (operator === "in" || operator === "not in") {
    try {
      const parsedJson = JSON.parse(trimmed);
      if (Array.isArray(parsedJson)) {
        return parsedJson;
      }
    } catch {
      // fall back to comma-separated inference
    }
    const parts = value
      .split(",")
      .map((entry) => entry.trim())
      .filter((entry) => entry !== "");
    if (parts.length === 0) {
      throw new Error(`${label} value is required`);
    }
    return parts.map((entry) => parseFilterValue(entry, "==", label));
  }
  if (trimmed === "true") {
    return true;
  }
  if (trimmed === "false") {
    return false;
  }
  if (/^-?(?:\d+|\d*\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed)) {
    const parsed = Number(trimmed);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  if (
    trimmed === "null" ||
    (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
    (trimmed.startsWith("[") && trimmed.endsWith("]"))
  ) {
    try {
      return JSON.parse(trimmed);
    } catch {
      throw new Error(`${label} value must be valid JSON`);
    }
  }
  return trimmed;
}

function buildFiltersFromForm(filters: StreamingFilterFormState[]): Record<string, unknown>[] {
  return filters.map((filter, index) => {
    const label = `Streaming dataset filter ${index + 1}`;
    const column = filter.column.trim();
    if (column === "") {
      throw new Error(`${label} column is required`);
    }
    return {
      column,
      operator: filter.operator,
      value: parseFilterValue(filter.value, filter.operator, label),
    };
  });
}

function hydrateDatasetUiFromConfig(config: Record<string, unknown> | null): {
  sourceMode: DatasetSourceMode;
  localTrainFiles: LocalTrainFileFormState[];
  hfToken: string;
  streamingDatasets: StreamingDatasetFormState[];
} {
  const datasetsRaw = asRecordArray(config?.datasets);
  const firstDataset = asRecord(datasetsRaw[0]);
  const localTrainPaths = extractTrainFilePaths(firstDataset.data_files);
  const sourceMode: DatasetSourceMode =
    datasetsRaw.length === 1 && localTrainPaths.length > 0 ? "local_file" : "streaming_hf";

  const localTrainFiles =
    sourceMode === "local_file"
      ? normalizeLocalTrainFiles(
          localTrainPaths.map((filePath) =>
            makeLocalTrainFileEntry({
              filePath,
              fileName: fileNameFromPath(filePath),
            })
          )
        )
      : [];

  const hfToken =
    sourceMode === "streaming_hf"
      ? asString(firstDataset.hf_token).trim()
      : "";

  const streamingDatasets =
    sourceMode === "streaming_hf" && datasetsRaw.length > 0
      ? normalizeStreamingDatasetWeights(
          datasetsRaw.map((entry) => {
            const datasetRecord = asRecord(entry);
            const datasetTextColumnsRaw = datasetRecord.text_columns;
            const datasetTextColumns = Array.isArray(datasetTextColumnsRaw)
              ? datasetTextColumnsRaw.map((item) => String(item)).join(", ")
              : "text";
            const datasetFiltersRaw = Array.isArray(datasetRecord.filters)
              ? datasetRecord.filters
              : [];
            return makeStreamingDatasetEntry({
              name: asString(datasetRecord.name),
              config: asString(datasetRecord.config),
              split: asString(datasetRecord.split, "train"),
              textColumns: datasetTextColumns,
              weight: sanitizeWeightInput(
                typeof datasetRecord.weight === "number" || typeof datasetRecord.weight === "string"
                  ? String(datasetRecord.weight)
                  : "1"
              ),
              filters: datasetFiltersRaw.map((filter) => {
                const filterRecord = asRecord(filter);
                return makeStreamingFilterEntry({
                  column: asString(filterRecord.column),
                  operator: asFilterOperator(filterRecord.operator),
                  value: stringifyFilterValue(filterRecord.value),
                });
              }),
            });
          })
        )
      : [makeStreamingDatasetEntry()];

  return {
    sourceMode,
    localTrainFiles,
    hfToken,
    streamingDatasets,
  };
}

function buildDatasetsFromUi(
  sourceMode: DatasetSourceMode,
  localTrainFiles: LocalTrainFileFormState[],
  hfToken: string,
  streamingDatasets: StreamingDatasetFormState[]
): Record<string, unknown>[] {
  if (sourceMode === "local_file") {
    const filePaths = normalizeLocalTrainFiles(localTrainFiles)
      .map((entry) => entry.filePath.trim())
      .filter((entry) => entry !== "");
    return [
      {
        name: "text",
        split: "train",
        text_columns: ["text"],
        weight: 1,
        streaming: true,
        data_files: {
          train: filePaths.length <= 1 ? (filePaths[0] ?? "") : filePaths,
        },
      },
    ];
  }

  const normalizedDatasets =
    streamingDatasets.length > 0
      ? normalizeStreamingDatasetWeights(streamingDatasets)
      : [makeStreamingDatasetEntry()];
  const token = hfToken.trim();

  return normalizedDatasets.map((entry) => {
    const datasetConfig: Record<string, unknown> = {
      name: entry.name.trim(),
      split: entry.split.trim() || "train",
      text_columns: entry.textColumns
        .split(",")
        .map((item) => item.trim())
        .filter((item) => item !== ""),
      weight: Math.max(0, parseWeightInput(entry.weight) ?? 0),
      streaming: true,
    };
    const configName = entry.config.trim();
    if (configName !== "") {
      datasetConfig.config = configName;
    }
    if (token !== "") {
      datasetConfig.hf_token = token;
    }
    const filters = entry.filters
      .filter((filter) => filter.column.trim() !== "" && filter.value.trim() !== "")
      .map((filter, index) => {
        try {
          return {
            column: filter.column.trim(),
            operator: filter.operator,
            value: parseFilterValue(
              filter.value,
              filter.operator,
              `Streaming dataset filter ${index + 1}`
            ),
          };
        } catch {
          return {
            column: filter.column.trim(),
            operator: filter.operator,
            value: filter.value,
          };
        }
      });
    if (filters.length > 0) {
      datasetConfig.filters = filters;
    }
    return datasetConfig;
  });
}

function readStoredJson<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") {
    return fallback;
  }
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) {
      return fallback;
    }
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function writeStoredJson(key: string, value: unknown): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // ignore local storage failures
  }
}

function cloneRecord<T>(value: T): T {
  if (typeof structuredClone === "function") {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value)) as T;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asRecord(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

function asRecordArray(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is Record<string, unknown> => isRecord(item));
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function formatStatusLabel(value: string): string {
  const normalized = value.replaceAll("_", " ").trim();
  if (normalized === "") {
    return "";
  }
  return `${normalized.slice(0, 1).toUpperCase()}${normalized.slice(1)}`;
}

function asNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function updateAtPath(
  source: Record<string, unknown>,
  path: string[],
  value: unknown
): Record<string, unknown> {
  const next = cloneRecord(source);
  let cursor: Record<string, unknown> = next;
  for (const segment of path.slice(0, -1)) {
    const existing = cursor[segment];
    const child = isRecord(existing) ? cloneRecord(existing) : {};
    cursor[segment] = child;
    cursor = child;
  }
  cursor[path[path.length - 1]] = value;
  return next;
}

function deleteAtPath(
  source: Record<string, unknown>,
  path: string[]
): Record<string, unknown> {
  const next = cloneRecord(source);
  let cursor: Record<string, unknown> = next;
  for (const segment of path.slice(0, -1)) {
    const existing = cursor[segment];
    if (!isRecord(existing)) {
      return next;
    }
    const child = cloneRecord(existing);
    cursor[segment] = child;
    cursor = child;
  }
  delete cursor[path[path.length - 1]];
  return next;
}

function replaceRunInOrder(runs: TrainingJob[], job: TrainingJob): TrainingJob[] {
  let replaced = false;
  const next = runs.map((item) => {
    if (item.id !== job.id) {
      return item;
    }
    replaced = true;
    return job;
  });
  return replaced ? next : [...next, job];
}

function metricChartData(
  metrics: TrainingMetricPoint[],
  metricKey: MetricChartKey
) {
  return metrics
    .map((item) => {
      const value = item[metricKey];
      if (typeof value !== "number" || !Number.isFinite(value)) {
        return null;
      }
      return {
        step: item.step,
        value,
        plotValue: value,
      };
    })
    .filter((item): item is { step: number; value: number; plotValue: number } => item !== null);
}

function metricChartStats(data: Array<{ value: number }>) {
  if (!data.length) {
    return null;
  }

  let min = data[0].value;
  let max = data[0].value;
  let total = 0;
  data.forEach((item) => {
    min = Math.min(min, item.value);
    max = Math.max(max, item.value);
    total += item.value;
  });

  return {
    latest: data[data.length - 1].value,
    min,
    max,
    average: total / data.length,
  };
}

function metricAxisDomain(data: Array<{ plotValue: number }>): [number | string | ((value: number) => number), number | string | ((value: number) => number)] {
  if (!data.length) {
    return [0, 1];
  }

  const values = data.map((item) => item.plotValue);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min;
  const padding = spread === 0 ? Math.max(Math.abs(max) * 0.08, 1) : spread * 0.08;

  return [min - padding, max + padding];
}

function formatMetricValue(value: number | null | undefined, digits: number): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a";
  }
  if (Math.abs(value) > 0 && Math.abs(value) < 0.0001) {
    return value.toExponential(2);
  }
  if (Math.abs(value) >= 1000) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
  }
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

function formatMetricAxis(value: number, digits: number): string {
  return formatMetricValue(value, digits);
}

function clampMetricRange(
  range: { startIndex: number; endIndex: number } | null,
  pointCount: number
): { startIndex: number; endIndex: number } | null {
  if (pointCount <= 0) {
    return null;
  }
  const lastIndex = pointCount - 1;
  if (!range) {
    return { startIndex: 0, endIndex: lastIndex };
  }
  const startIndex = Math.max(0, Math.min(range.startIndex, lastIndex));
  const endIndex = Math.max(startIndex, Math.min(range.endIndex, lastIndex));
  return { startIndex, endIndex };
}

function chartBrushHandlePosition(percent: number): string {
  const handleHalfWidth = 5;
  const edgeOffset = (1 - 2 * (percent / 100)) * handleHalfWidth;
  return `calc(${percent}% + ${edgeOffset.toFixed(2)}px)`;
}

function formatMetric(value: number | null | undefined, digits = 3): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a";
  }
  return value.toFixed(digits);
}

function formatInteger(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a";
  }
  return value.toLocaleString();
}

function humanizeConfigToken(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\bbpe\b/i, "BPE")
    .replace(/\bwordpiece\b/i, "WordPiece")
    .replace(/\bunigram\b/i, "Unigram");
}

function firstAttentionConfig(modelConfig: Record<string, unknown>): Record<string, unknown> {
  const blocks = Array.isArray(modelConfig.blocks) ? modelConfig.blocks : [];
  for (const block of blocks) {
    const components = asRecordArray(asRecord(block).components);
    for (const component of components) {
      const attention = asRecord(component.attention);
      if (Object.keys(attention).length > 0) {
        return attention;
      }
    }
  }
  return {};
}

function formatModelConfigMeta(value: unknown): string {
  const modelConfig = asRecord(value);
  const blocks = Array.isArray(modelConfig.blocks) ? modelConfig.blocks : [];
  const attention = firstAttentionConfig(modelConfig);
  const parts: string[] = [];
  if (blocks.length > 0) {
    parts.push(`${formatInteger(blocks.length)} layers`);
  }
  const headCount = asNumber(attention.n_head, 0);
  if (headCount > 0) {
    parts.push(`${formatInteger(headCount)} attention heads`);
  }
  const embeddingSize = asNumber(modelConfig.n_embd, 0);
  if (embeddingSize > 0) {
    parts.push(`${formatInteger(embeddingSize)} embedding width`);
  }
  const contextLength = asNumber(modelConfig.context_length, 0);
  if (contextLength > 0) {
    parts.push(`${formatInteger(contextLength)} context length`);
  }
  const vocabSize = asNumber(modelConfig.vocab_size, 0);
  if (vocabSize > 0) {
    parts.push(`${formatInteger(vocabSize)} vocabulary size`);
  }
  return parts.length > 0 ? parts.join(" • ") : "Model dimensions unavailable";
}

function formatTokenizerMeta(job: TokenizerTrainingJob): string {
  const config = job.tokenizer_config;
  const stats = job.stats;
  const parts: string[] = [];
  const vocabSize = stats?.vocab_size ?? asNumber(config.vocab_size, 0);
  if (vocabSize > 0) {
    parts.push(`${formatInteger(vocabSize)} vocabulary size`);
  }
  const tokenizerType = asString(config.tokenizer_type);
  const preTokenizer = asString(config.pre_tokenizer);
  if (tokenizerType || preTokenizer) {
    parts.push(
      [tokenizerType, preTokenizer]
        .filter(Boolean)
        .map(humanizeConfigToken)
        .join(" / ")
    );
  }
  if (typeof stats?.chars_per_token === "number" && Number.isFinite(stats.chars_per_token)) {
    parts.push(`${stats.chars_per_token.toFixed(2)} characters per token`);
  }
  const specialTokenCount = Array.isArray(config.special_tokens) ? config.special_tokens.length : 0;
  if (specialTokenCount > 0) {
    parts.push(`${formatInteger(specialTokenCount)} special tokens`);
  }
  return parts.length > 0 ? parts.join(" • ") : "Tokenizer details unavailable";
}

function formatDuration(seconds: number | null | undefined): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) {
    return "n/a";
  }
  const whole = Math.floor(seconds);
  const hrs = Math.floor(whole / 3600);
  const mins = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;
  if (hrs > 0) {
    return `${hrs}h ${mins}m`;
  }
  if (mins > 0) {
    return `${mins}m ${secs}s`;
  }
  return `${secs}s`;
}

function statusTone(status: string): string {
  if (status === "completed") {
    return "tone-good";
  }
  if (status === "failed" || status === "cancelled") {
    return "tone-error";
  }
  if (status === "running") {
    return "tone-neutral";
  }
  return "tone-warn";
}

function issueTone(issue: TrainingIssue): "error" | "warning" {
  return issue.severity === "warning" ? "warning" : "error";
}

function prettyJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function defaultRunName(project: ProjectDetail | null, tokenizer: TokenizerTrainingJob | null): string {
  const projectName = project?.name ?? "model";
  const tokenizerName = asString(tokenizer?.tokenizer_config?.name, "tokenizer");
  return `${projectName} x ${tokenizerName}`;
}

function MetricChartTooltip({
  active,
  payload,
  label,
  title,
  digits,
}: {
  active?: boolean;
  payload?: Array<{ payload?: { value?: number } }>;
  label?: number | string;
  title: string;
  digits: number;
}) {
  const value = payload?.[0]?.payload?.value;

  if (!active || typeof value !== "number") {
    return null;
  }

  return (
    <div className="trainingChartTooltip">
      <span>Step {label}</span>
      <strong>{title}: {formatMetricValue(value, digits)}</strong>
    </div>
  );
}

function MetricRangeSelector({
  data,
  range,
  onChange,
}: {
  data: Array<{ step: number; plotValue: number }>;
  range: { startIndex: number; endIndex: number };
  onChange: (range: { startIndex: number; endIndex: number }) => void;
}) {
  const brushRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{
    mode: "start" | "end" | "window";
    index: number;
    startIndex: number;
    endIndex: number;
  } | null>(null);
  const [dragMode, setDragMode] = useState<"start" | "end" | "window" | null>(null);
  const lastIndex = data.length - 1;
  const startStep = data[range.startIndex]?.step ?? 0;
  const endStep = data[range.endIndex]?.step ?? 0;
  const selectedLeft = lastIndex > 0 ? (range.startIndex / lastIndex) * 100 : 0;
  const selectedRight = lastIndex > 0 ? (range.endIndex / lastIndex) * 100 : 100;
  const selectedWidth = Math.max(0, selectedRight - selectedLeft);
  const overviewPath = useMemo(() => {
    if (!data.length) {
      return "";
    }
    const width = 1000;
    const height = 72;
    const padding = 7;
    const values = data.map((item) => item.plotValue);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const rangeSize = max - min || 1;
    return data
      .map((item, index) => {
        const x = data.length === 1 ? width / 2 : (index / (data.length - 1)) * width;
        const normalized = (item.plotValue - min) / rangeSize;
        const y = height - padding - normalized * (height - padding * 2);
        return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(" ");
  }, [data]);

  const indexFromEvent = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const bounds = brushRef.current?.getBoundingClientRect();
      if (!bounds || lastIndex <= 0) {
        return 0;
      }
      const ratio = Math.max(0, Math.min((event.clientX - bounds.left) / bounds.width, 1));
      return Math.round(ratio * lastIndex);
    },
    [lastIndex]
  );

  const commitRange = useCallback(
    (startIndex: number, endIndex: number) => {
      const start = Math.max(0, Math.min(startIndex, lastIndex));
      const end = Math.max(start, Math.min(endIndex, lastIndex));
      onChange({ startIndex: start, endIndex: end });
    },
    [lastIndex, onChange]
  );

  const beginDrag = (
    mode: "start" | "end" | "window",
    event: ReactPointerEvent<HTMLButtonElement | HTMLDivElement>
  ) => {
    event.preventDefault();
    event.stopPropagation();
    brushRef.current?.setPointerCapture(event.pointerId);
    const index = indexFromEvent(event as ReactPointerEvent<HTMLDivElement>);
    dragRef.current = {
      mode,
      index,
      startIndex: range.startIndex,
      endIndex: range.endIndex,
    };
    setDragMode(mode);
  };

  const handleTrackPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    const index = indexFromEvent(event);
    const startDistance = Math.abs(index - range.startIndex);
    const endDistance = Math.abs(index - range.endIndex);
    if (index >= range.startIndex && index <= range.endIndex) {
      beginDrag("window", event);
      return;
    }
    commitRange(
      startDistance <= endDistance ? index : range.startIndex,
      startDistance <= endDistance ? range.endIndex : index
    );
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) {
      return;
    }
    const index = indexFromEvent(event);
    if (drag.mode === "start") {
      commitRange(index, drag.endIndex);
      return;
    }
    if (drag.mode === "end") {
      commitRange(drag.startIndex, index);
      return;
    }

    const windowSize = drag.endIndex - drag.startIndex;
    const delta = index - drag.index;
    const nextStart = Math.max(0, Math.min(drag.startIndex + delta, lastIndex - windowSize));
    commitRange(nextStart, nextStart + windowSize);
  };

  const handlePointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (dragRef.current) {
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
    }
    dragRef.current = null;
    setDragMode(null);
  };

  const handleHandleKeyDown = (
    mode: "start" | "end",
    event: ReactKeyboardEvent<HTMLButtonElement>
  ) => {
    const direction = event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0;
    if (direction === 0) {
      return;
    }
    event.preventDefault();
    const delta = direction * (event.shiftKey ? 10 : 1);
    if (mode === "start") {
      commitRange(range.startIndex + delta, range.endIndex);
      return;
    }
    commitRange(range.startIndex, range.endIndex + delta);
  };

  return (
    <div
      ref={brushRef}
      className={`trainingChartRange ${dragMode ? "isDragging" : ""}`}
      aria-label={`Visible steps ${startStep} to ${endStep}`}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
    >
      <div className="trainingChartRangeMeta">
        <span>Visible steps</span>
        <strong>
          {startStep} - {endStep}
        </strong>
      </div>
      <div className="trainingChartBrush" onPointerDown={handleTrackPointerDown}>
        <svg viewBox="0 0 1000 72" preserveAspectRatio="none" aria-hidden>
          <path d={overviewPath} />
        </svg>
        <div className="trainingChartBrushShade left" style={{ width: `${selectedLeft}%` }} />
        <div
          className="trainingChartBrushSelection"
          style={{ left: `${selectedLeft}%`, width: `${selectedWidth}%` }}
          onPointerDown={(event) => beginDrag("window", event)}
        />
        <button
          type="button"
          className="trainingChartBrushHandle"
          style={{ left: chartBrushHandlePosition(selectedLeft) }}
          aria-label={`Start visible range at step ${startStep}`}
          onPointerDown={(event) => beginDrag("start", event)}
          onKeyDown={(event) => handleHandleKeyDown("start", event)}
        />
        <button
          type="button"
          className="trainingChartBrushHandle"
          style={{ left: chartBrushHandlePosition(selectedRight) }}
          aria-label={`End visible range at step ${endStep}`}
          onPointerDown={(event) => beginDrag("end", event)}
          onKeyDown={(event) => handleHandleKeyDown("end", event)}
        />
        <div className="trainingChartBrushShade right" style={{ left: `${selectedRight}%` }} />
      </div>
    </div>
  );
}

function MetricChart({
  title,
  metricKey,
  metrics,
  latestValue,
  stroke,
  digits,
}: {
  title: string;
  metricKey: MetricChartKey;
  metrics: TrainingMetricPoint[];
  latestValue: string;
  stroke: string;
  digits: number;
}) {
  const data = useMemo(
    () => metricChartData(metrics, metricKey),
    [metricKey, metrics]
  );
  const previousPointCountRef = useRef(0);
  const [range, setRange] = useState<{ startIndex: number; endIndex: number } | null>(null);

  useEffect(() => {
    setRange((current) => {
      if (!data.length) {
        previousPointCountRef.current = 0;
        return null;
      }

      const previousLastIndex = Math.max(previousPointCountRef.current - 1, 0);
      const nextLastIndex = data.length - 1;
      previousPointCountRef.current = data.length;

      if (!current) {
        return { startIndex: 0, endIndex: nextLastIndex };
      }

      const wasPinnedToEnd = current.endIndex >= previousLastIndex;
      const endIndex = wasPinnedToEnd ? nextLastIndex : Math.min(current.endIndex, nextLastIndex);
      const startIndex = Math.min(current.startIndex, endIndex);
      return { startIndex, endIndex };
    });
  }, [data.length]);

  const visibleRange = clampMetricRange(range, data.length);
  const visibleData = useMemo(
    () => (visibleRange ? data.slice(visibleRange.startIndex, visibleRange.endIndex + 1) : []),
    [data, visibleRange]
  );
  const stats = useMemo(() => metricChartStats(visibleData), [visibleData]);
  const yDomain = useMemo(() => metricAxisDomain(visibleData), [visibleData]);
  const handleRangeChange = useCallback((nextRange: { startIndex: number; endIndex: number }) => {
    setRange(nextRange);
  }, []);

  return (
    <div className="trainingChartCard">
      <div className="trainingChartHead">
        <div>
          <strong>{title}</strong>
          <span>{data.length ? `${formatInteger(data.length)} points` : "Waiting for data"}</span>
        </div>
        <div className="trainingChartLatest">
          <span>Current</span>
          <strong>{latestValue}</strong>
        </div>
      </div>
      {data.length ? (
        <>
          <div className="trainingChartStats" aria-label={`${title} summary`}>
            <span>Min {formatMetricValue(stats?.min, digits)}</span>
            <span>Max {formatMetricValue(stats?.max, digits)}</span>
            <span>Avg {formatMetricValue(stats?.average, digits)}</span>
          </div>
          <div className="trainingChartFrame">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={visibleData} margin={{ top: 12, right: 12, bottom: 0, left: 4 }}>
                <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="step"
                  tickLine={false}
                  axisLine={false}
                  minTickGap={24}
                  tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                  tickFormatter={(value) => `s${value}`}
                />
                <YAxis
                  width={58}
                  domain={yDomain}
                  tickLine={false}
                  axisLine={false}
                  tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                  tickFormatter={(value) => formatMetricAxis(Number(value), digits)}
                />
                <Tooltip
                  cursor={{ stroke: "var(--text-muted)", strokeDasharray: "4 4" }}
                  content={<MetricChartTooltip title={title} digits={digits} />}
                />
                <Line
                  type="monotone"
                  dataKey="plotValue"
                  stroke={stroke}
                  strokeWidth={2.4}
                  dot={false}
                  activeDot={{ r: 4, strokeWidth: 2, stroke }}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          {data.length > 8 && visibleRange ? (
            <MetricRangeSelector data={data} range={visibleRange} onChange={handleRangeChange} />
          ) : null}
        </>
      ) : (
        <div className="trainingEmpty">Metrics will appear after the first logged steps.</div>
      )}
    </div>
  );
}

function ConfigNumberInput({
  value,
  onCommit,
  mode = "integer",
  step,
  min,
  max,
  placeholder,
}: {
  value: number;
  onCommit: (value: number) => void;
  mode?: "integer" | "decimal";
  step?: number | string;
  min?: number | string;
  max?: number | string;
  placeholder?: string;
}) {
  const [draft, setDraft] = useState(() => formatNumberInputValue(value));
  const [focused, setFocused] = useState(false);
  const formattedValue = formatNumberInputValue(value);

  useEffect(() => {
    if (!focused) {
      setDraft(formattedValue);
    }
  }, [focused, formattedValue]);

  const sanitize =
    mode === "decimal" ? sanitizePositiveDecimalInput : sanitizePositiveIntegerInput;

  return (
    <input
      type="text"
      inputMode={mode === "decimal" ? "decimal" : "numeric"}
      pattern={mode === "decimal" ? "[0-9]*[.]?[0-9]*" : "[0-9]*"}
      step={step}
      min={min}
      max={max}
      placeholder={placeholder}
      value={draft}
      onFocus={() => setFocused(true)}
      onChange={(event) => {
        setDraft(sanitize(event.target.value));
      }}
      onBlur={() => {
        setFocused(false);
        const parsed = parseConfigNumberInput(draft, mode);
        if (parsed === null) {
          setDraft(formattedValue);
          return;
        }
        onCommit(parsed);
        setDraft(formatNumberInputValue(parsed));
      }}
    />
  );
}

function OptionalConfigNumberInput({
  value,
  onCommit,
  mode = "integer",
  step,
  min,
  max,
  placeholder,
}: {
  value: number | null;
  onCommit: (value: number | null) => void;
  mode?: "integer" | "decimal";
  step?: number | string;
  min?: number | string;
  max?: number | string;
  placeholder?: string;
}) {
  const [draft, setDraft] = useState(() =>
    value === null ? "" : formatNumberInputValue(value)
  );
  const [focused, setFocused] = useState(false);
  const formattedValue = value === null ? "" : formatNumberInputValue(value);

  useEffect(() => {
    if (!focused) {
      setDraft(formattedValue);
    }
  }, [focused, formattedValue]);

  const sanitize =
    mode === "decimal" ? sanitizePositiveDecimalInput : sanitizePositiveIntegerInput;

  return (
    <input
      type="text"
      inputMode={mode === "decimal" ? "decimal" : "numeric"}
      pattern={mode === "decimal" ? "[0-9]*[.]?[0-9]*" : "[0-9]*"}
      step={step}
      min={min}
      max={max}
      placeholder={placeholder}
      value={draft}
      onFocus={() => setFocused(true)}
      onChange={(event) => {
        setDraft(sanitize(event.target.value));
      }}
      onBlur={() => {
        setFocused(false);
        if (draft.trim() === "") {
          onCommit(null);
          setDraft("");
          return;
        }
        const parsed = parseConfigNumberInput(draft, mode);
        if (parsed === null) {
          setDraft(formattedValue);
          return;
        }
        onCommit(parsed);
        setDraft(formatNumberInputValue(parsed));
      }}
    />
  );
}

function TrainingPageContent() {
  const searchParams = useSearchParams();
  const [theme, setTheme] = useThemeMode();
  const [trainingConfig, setTrainingConfig] = useState<Record<string, unknown> | null>(null);
  const [dataloaderConfig, setDataloaderConfig] = useState<Record<string, unknown> | null>(null);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [selectedTokenizerJobId, setSelectedTokenizerJobId] = useState<string | null>(null);
  const [selectedProject, setSelectedProject] = useState<ProjectDetail | null>(null);
  const [selectedTokenizer, setSelectedTokenizer] = useState<TokenizerTrainingJob | null>(null);
  const [runName, setRunName] = useState("");
  const [preflight, setPreflight] = useState<TrainingPreflightResponse | null>(null);
  const [preflightLoading, setPreflightLoading] = useState(false);
  const [preflightError, setPreflightError] = useState<string | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [activeRun, setActiveRun] = useState<TrainingJob | null>(null);
  const [recentRuns, setRecentRuns] = useState<TrainingJob[]>([]);
  const [metrics, setMetrics] = useState<TrainingMetricPoint[]>([]);
  const [samples, setSamples] = useState<TrainingSampleEntry[]>([]);
  const [logs, setLogs] = useState<{ stdout: string[]; stderr: string[] }>({
    stdout: [],
    stderr: [],
  });
  const [checkpoints, setCheckpoints] = useState<TrainingCheckpointEntry[]>([]);
  const [launching, setLaunching] = useState(false);
  const [toasts, setToasts] = useState<ToastState[]>([]);
  const [pickerKind, setPickerKind] = useState<AssetPickerKind | null>(null);
  const [pickerQuery, setPickerQuery] = useState("");
  const [pickerLoading, setPickerLoading] = useState(false);
  const [pickerError, setPickerError] = useState<string | null>(null);
  const [pickerProjects, setPickerProjects] = useState<ProjectSummary[]>([]);
  const [pickerTokenizerJobs, setPickerTokenizerJobs] = useState<TokenizerTrainingJob[]>([]);
  const [datasetSourceMode, setDatasetSourceMode] = useState<DatasetSourceMode>("streaming_hf");
  const [localTrainFiles, setLocalTrainFiles] = useState<LocalTrainFileFormState[]>([]);
  const [hfToken, setHfToken] = useState("");
  const [streamingDatasets, setStreamingDatasets] = useState<StreamingDatasetFormState[]>([
    makeStreamingDatasetEntry(),
  ]);
  const [isDraggingTrainFiles, setIsDraggingTrainFiles] = useState(false);
  const [isUploadingTrainFile, setIsUploadingTrainFile] = useState(false);
  const [isLoadingDatasetTemplate, setIsLoadingDatasetTemplate] = useState(false);
  const [highlightedWorkflowTarget, setHighlightedWorkflowTarget] =
    useState<WorkflowTarget | null>(null);
  const initializedRef = useRef(false);
  const pickerRequestIdRef = useRef(0);
  const datasetUiHydratedRef = useRef(false);
  const localFileDragDepthRef = useRef(0);
  const localTrainFileStatsPendingIdsRef = useRef(new Set<string>());
  const localTrainFileStatsFailedIdsRef = useRef(new Set<string>());
  const workflowHighlightTimeoutRef = useRef<number | null>(null);
  const trainingPlanPanelRef = useRef<HTMLDetailsElement | null>(null);
  const datasetPanelRef = useRef<HTMLDetailsElement | null>(null);
  const modelSelectionRef = useRef<HTMLDivElement | null>(null);
  const tokenizerSelectionRef = useRef<HTMLDivElement | null>(null);
  const trainingSettingsRef = useRef<HTMLDivElement | null>(null);
  const datasetSettingsRef = useRef<HTMLDivElement | null>(null);
  const preflightSectionRef = useRef<HTMLElement | null>(null);

  const deferredTrainingConfig = useDeferredValue(trainingConfig);
  const deferredDataloaderConfig = useDeferredValue(dataloaderConfig);

  const notify = useCallback((level: ToastLevel, title: string, body: string) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setToasts((current) => [...current, { id, level, title, body }]);
    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id));
    }, 3600);
  }, []);

  useEffect(() => {
    return () => {
      if (workflowHighlightTimeoutRef.current !== null) {
        window.clearTimeout(workflowHighlightTimeoutRef.current);
      }
    };
  }, []);

  const openWorkflowTarget = useCallback((target: WorkflowTarget) => {
    if (target === "training" && trainingPlanPanelRef.current) {
      trainingPlanPanelRef.current.open = true;
    }
    if (target === "dataset" && datasetPanelRef.current) {
      datasetPanelRef.current.open = true;
    }

    const targetRef =
      target === "model"
        ? modelSelectionRef
        : target === "tokenizer"
          ? tokenizerSelectionRef
          : target === "training"
            ? trainingSettingsRef
            : target === "dataset"
              ? datasetSettingsRef
              : preflightSectionRef;

    const hash = WORKFLOW_TARGET_HASH_MAP[target];
    if (window.location.hash !== hash) {
      window.history.replaceState(null, "", hash);
    }

    targetRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
      inline: "nearest",
    });

    setHighlightedWorkflowTarget(target);
    if (workflowHighlightTimeoutRef.current !== null) {
      window.clearTimeout(workflowHighlightTimeoutRef.current);
    }
    workflowHighlightTimeoutRef.current = window.setTimeout(() => {
      setHighlightedWorkflowTarget((previous) =>
        previous === target ? null : previous
      );
    }, 1800);
  }, []);

  const refreshRecentRuns = useCallback(async () => {
    try {
      const jobs = await fetchTrainingJobs();
      startTransition(() => {
        setRecentRuns(jobs);
        if (!activeRunId) {
          const nextRun = jobs[0] ?? null;
          if (nextRun) {
            setActiveRunId(nextRun.id);
          }
        }
      });
    } catch (error) {
      notify("error", "Recent runs unavailable", error instanceof Error ? error.message : "Failed to load training jobs.");
    }
  }, [activeRunId, notify]);

  useEffect(() => {
    if (initializedRef.current) {
      return;
    }
    initializedRef.current = true;

    const storedTraining = readStoredJson<Record<string, unknown> | null>(
      TRAINING_CONFIG_STORAGE_KEY,
      null
    );
    const storedDataloader = readStoredJson<Record<string, unknown> | null>(
      DATALOADER_CONFIG_STORAGE_KEY,
      null
    );
    const storedSelection = readStoredJson<{ projectId: string | null; tokenizerJobId: string | null }>(
      TRAINING_SELECTION_STORAGE_KEY,
      {
        projectId: null,
        tokenizerJobId: null,
      }
    );
    const storedActiveRun = readStoredJson<string | null>(ACTIVE_RUN_STORAGE_KEY, null);

    setActiveRunId(searchParams.get("run") ?? storedActiveRun);
    setSelectedProjectId(searchParams.get("project") ?? storedSelection.projectId);
    setSelectedTokenizerJobId(searchParams.get("tokenizerJob") ?? storedSelection.tokenizerJobId);

    void Promise.all([fetchTrainingConfigTemplates(), fetchTrainingJobs()])
      .then(([templates, jobs]) => {
        startTransition(() => {
          setTrainingConfig(storedTraining ?? templates.training_config_template);
          setDataloaderConfig(storedDataloader ?? templates.dataloader_config_template);
          setRecentRuns(jobs);
          if (!searchParams.get("run") && !storedActiveRun) {
            const nextRun = jobs[0] ?? null;
            if (nextRun) {
              setActiveRunId(nextRun.id);
            }
          }
        });
      })
      .catch((error) => {
        notify(
          "error",
          "Training config unavailable",
          error instanceof Error ? error.message : "Failed to load training defaults."
        );
      });
  }, [notify, searchParams]);

  useEffect(() => {
    writeStoredJson(TRAINING_SELECTION_STORAGE_KEY, {
      projectId: selectedProjectId,
      tokenizerJobId: selectedTokenizerJobId,
    });
  }, [selectedProjectId, selectedTokenizerJobId]);

  useEffect(() => {
    const hash = window.location.hash;
    if (hash === WORKFLOW_TARGET_HASH_MAP.model) {
      openWorkflowTarget("model");
      return;
    }
    if (hash === WORKFLOW_TARGET_HASH_MAP.tokenizer) {
      openWorkflowTarget("tokenizer");
      return;
    }
    if (hash === WORKFLOW_TARGET_HASH_MAP.training) {
      openWorkflowTarget("training");
      return;
    }
    if (hash === WORKFLOW_TARGET_HASH_MAP.dataset) {
      openWorkflowTarget("dataset");
      return;
    }
    if (hash === WORKFLOW_TARGET_HASH_MAP.preflight) {
      openWorkflowTarget("preflight");
    }
  }, [openWorkflowTarget]);

  useEffect(() => {
    writeStoredJson(ACTIVE_RUN_STORAGE_KEY, activeRunId);
  }, [activeRunId]);

  useEffect(() => {
    if (trainingConfig) {
      writeStoredJson(TRAINING_CONFIG_STORAGE_KEY, trainingConfig);
    }
  }, [trainingConfig]);

  useEffect(() => {
    if (dataloaderConfig) {
      writeStoredJson(DATALOADER_CONFIG_STORAGE_KEY, dataloaderConfig);
    }
  }, [dataloaderConfig]);

  useEffect(() => {
    if (!selectedProjectId) {
      setSelectedProject(null);
      return;
    }
    const controller = new AbortController();
    void fetchProject(selectedProjectId, controller.signal)
      .then((project) => {
        startTransition(() => {
          setSelectedProject(project);
          setRunName((current) => current || defaultRunName(project, selectedTokenizer));
        });
      })
      .catch((error) => {
        if (!controller.signal.aborted) {
          notify("error", "Model config unavailable", error instanceof Error ? error.message : "Failed to load selected model config.");
        }
      });
    return () => controller.abort();
  }, [notify, selectedProjectId, selectedTokenizer]);

  useEffect(() => {
    if (!selectedTokenizerJobId) {
      setSelectedTokenizer(null);
      return;
    }
    void fetchTokenizerJob(selectedTokenizerJobId)
      .then((job) => {
        startTransition(() => {
          setSelectedTokenizer(job);
          setRunName((current) => current || defaultRunName(selectedProject, job));
        });
      })
      .catch((error) => {
        notify("error", "Tokenizer unavailable", error instanceof Error ? error.message : "Failed to load selected tokenizer.");
      });
  }, [notify, selectedProject, selectedTokenizerJobId]);

  useEffect(() => {
    if (!selectedProjectId || !selectedTokenizerJobId || !deferredTrainingConfig || !deferredDataloaderConfig) {
      setPreflight(null);
      setPreflightError(null);
      return;
    }

    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => {
      setPreflightLoading(true);
      void validateTrainingPreflight(
        {
          project_id: selectedProjectId,
          tokenizer_job_id: selectedTokenizerJobId,
          training_config: deferredTrainingConfig,
          dataloader_config: deferredDataloaderConfig,
        },
        controller.signal
      )
        .then((result) => {
          startTransition(() => {
            setPreflight(result);
            setPreflightError(null);
          });
        })
        .catch((error) => {
          if (!controller.signal.aborted) {
            setPreflight(null);
            setPreflightError(error instanceof Error ? error.message : "Failed to validate preflight.");
          }
        })
        .finally(() => {
          if (!controller.signal.aborted) {
            setPreflightLoading(false);
          }
        });
    }, 420);

    return () => {
      controller.abort();
      window.clearTimeout(timeoutId);
    };
  }, [deferredDataloaderConfig, deferredTrainingConfig, selectedProjectId, selectedTokenizerJobId]);

  useEffect(() => {
    if (!activeRunId) {
      setActiveRun(null);
      setMetrics([]);
      setSamples([]);
      setLogs({ stdout: [], stderr: [] });
      setCheckpoints([]);
      return;
    }

    let cancelled = false;

    const poll = async () => {
      try {
        const [job, fetchedMetrics, fetchedSamples, fetchedLogs, fetchedCheckpoints] = await Promise.all([
          fetchTrainingJob(activeRunId),
          fetchTrainingMetrics(activeRunId),
          fetchTrainingSamples(activeRunId),
          fetchTrainingLogs(activeRunId),
          fetchTrainingCheckpoints(activeRunId),
        ]);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setActiveRun(job);
          setMetrics(fetchedMetrics);
          setSamples(fetchedSamples);
          setLogs({
            stdout: fetchedLogs.stdout_lines,
            stderr: fetchedLogs.stderr_lines,
          });
          setCheckpoints(fetchedCheckpoints);
          setRecentRuns((current) => replaceRunInOrder(current, job).slice(0, 12));
        });
      } catch (error) {
        if (!cancelled) {
          notify("error", "Run polling interrupted", error instanceof Error ? error.message : "Failed to refresh the active run.");
        }
      }
    };

    void poll();
    const intervalId = window.setInterval(() => {
      void poll();
    }, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [activeRunId, notify]);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      void refreshRecentRuns();
    }, 6000);
    return () => window.clearInterval(intervalId);
  }, [refreshRecentRuns]);

  const datasetEntries = useMemo(
    () => asRecordArray(dataloaderConfig?.datasets),
    [dataloaderConfig]
  );

  const promptEntries = useMemo(() => {
    const sampler = asRecord(trainingConfig?.sampler);
    return asRecordArray(sampler.prompts);
  }, [trainingConfig]);

  const normalizedPickerQuery = pickerQuery.trim().toLowerCase();

  const visiblePickerProjects = useMemo(() => {
    return [...pickerProjects]
      .sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at))
      .filter((project) => {
        if (normalizedPickerQuery === "") {
          return true;
        }
        return [project.name, project.id, project.artifact_file, project.artifact_path].some(
          (value) =>
            typeof value === "string" && value.toLowerCase().includes(normalizedPickerQuery)
        );
      });
  }, [normalizedPickerQuery, pickerProjects]);

  const visiblePickerTokenizerJobs = useMemo(() => {
    return [...pickerTokenizerJobs]
      .filter((job) => job.status === "completed")
      .sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at))
      .filter((job) => {
        if (normalizedPickerQuery === "") {
          return true;
        }
        const tokenizerName = asString(job.tokenizer_config.name, job.id);
        return [tokenizerName, job.id, job.artifact_file, job.artifact_path].some(
          (value) =>
            typeof value === "string" && value.toLowerCase().includes(normalizedPickerQuery)
        );
      });
  }, [normalizedPickerQuery, pickerTokenizerJobs]);

  useEffect(() => {
    if (!dataloaderConfig || datasetUiHydratedRef.current) {
      return;
    }
    const hydrated = hydrateDatasetUiFromConfig(dataloaderConfig);
    setDatasetSourceMode(hydrated.sourceMode);
    setLocalTrainFiles(hydrated.localTrainFiles);
    setHfToken(hydrated.hfToken);
    setStreamingDatasets(hydrated.streamingDatasets);
    datasetUiHydratedRef.current = true;
  }, [dataloaderConfig]);

  useEffect(() => {
    if (!datasetUiHydratedRef.current) {
      return;
    }
    const nextDatasets = buildDatasetsFromUi(
      datasetSourceMode,
      localTrainFiles,
      hfToken,
      streamingDatasets
    );
    setDataloaderConfig((current) => {
      const next = cloneRecord(current ?? {});
      next.datasets = nextDatasets;
      return next;
    });
  }, [datasetSourceMode, hfToken, localTrainFiles, streamingDatasets]);

  useEffect(() => {
    if (!datasetUiHydratedRef.current) {
      return;
    }

    const currentIds = new Set(localTrainFiles.map((entry) => entry.id));
    for (const entryId of Array.from(localTrainFileStatsPendingIdsRef.current)) {
      if (!currentIds.has(entryId)) {
        localTrainFileStatsPendingIdsRef.current.delete(entryId);
      }
    }
    for (const entryId of Array.from(localTrainFileStatsFailedIdsRef.current)) {
      if (!currentIds.has(entryId)) {
        localTrainFileStatsFailedIdsRef.current.delete(entryId);
      }
    }

    const entriesNeedingStats = localTrainFiles.filter((entry) => {
      if (
        typeof entry.sizeBytes === "number" &&
        Number.isFinite(entry.sizeBytes) &&
        entry.sizeBytes >= 0 &&
        typeof entry.sizeChars === "number" &&
        Number.isFinite(entry.sizeChars) &&
        entry.sizeChars >= 0
      ) {
        return false;
      }
      if (entry.filePath.trim() === "") {
        return false;
      }
      if (localTrainFileStatsPendingIdsRef.current.has(entry.id)) {
        return false;
      }
      if (localTrainFileStatsFailedIdsRef.current.has(entry.id)) {
        return false;
      }
      return true;
    });

    if (entriesNeedingStats.length === 0) {
      return;
    }

    entriesNeedingStats.forEach((entry) => {
      localTrainFileStatsPendingIdsRef.current.add(entry.id);
    });

    let cancelled = false;

    void Promise.allSettled(
      entriesNeedingStats.map(async (entry) => ({
        entryId: entry.id,
        stats: await fetchLocalTrainFileStats(entry.filePath),
      }))
    ).then((results) => {
      const updatesById = new Map<string, { sizeBytes: number; sizeChars: number }>();

      results.forEach((result, index) => {
        const entry = entriesNeedingStats[index];
        localTrainFileStatsPendingIdsRef.current.delete(entry.id);

        if (result.status === "fulfilled") {
          localTrainFileStatsFailedIdsRef.current.delete(entry.id);
          updatesById.set(entry.id, {
            sizeBytes: result.value.stats.size_bytes,
            sizeChars: result.value.stats.size_chars,
          });
          return;
        }

        localTrainFileStatsFailedIdsRef.current.add(entry.id);
      });

      if (cancelled || updatesById.size === 0) {
        return;
      }

      setLocalTrainFiles((previous) =>
        normalizeLocalTrainFiles(
          previous.map((entry) => {
            const stats = updatesById.get(entry.id);
            return stats ? { ...entry, ...stats } : entry;
          })
        )
      );
    });

    return () => {
      cancelled = true;
    };
  }, [localTrainFiles]);

  const updateStreamingDataset = useCallback(
    (datasetId: string, updates: Partial<Omit<StreamingDatasetFormState, "id">>) => {
      setStreamingDatasets((previous) =>
        previous.map((entry) => (entry.id === datasetId ? { ...entry, ...updates } : entry))
      );
    },
    []
  );

  const updateStreamingWeight = useCallback((datasetId: string, rawWeight: string) => {
    const sanitizedWeight = sanitizeWeightInput(rawWeight);
    setStreamingDatasets((previous) =>
      normalizeStreamingDatasetWeights(previous, datasetId, sanitizedWeight)
    );
  }, []);

  const addStreamingDataset = useCallback(() => {
    setStreamingDatasets((previous) =>
      normalizeStreamingDatasetWeights([...previous, makeStreamingDatasetEntry()])
    );
  }, []);

  const removeStreamingDataset = useCallback((datasetId: string) => {
    setStreamingDatasets((previous) =>
      normalizeStreamingDatasetWeights(
        previous.filter((entry) => entry.id !== datasetId).length > 0
          ? previous.filter((entry) => entry.id !== datasetId)
          : [makeStreamingDatasetEntry()]
      )
    );
  }, []);

  const updateStreamingFilter = useCallback(
    (
      datasetId: string,
      filterId: string,
      updates: Partial<Omit<StreamingFilterFormState, "id">>
    ) => {
      setStreamingDatasets((previous) =>
        previous.map((entry) => {
          if (entry.id !== datasetId) {
            return entry;
          }
          return {
            ...entry,
            filters: entry.filters.map((filter) =>
              filter.id === filterId ? { ...filter, ...updates } : filter
            ),
          };
        })
      );
    },
    []
  );

  const addStreamingFilter = useCallback((datasetId: string) => {
    setStreamingDatasets((previous) =>
      previous.map((entry) =>
        entry.id === datasetId
          ? { ...entry, filters: [...entry.filters, makeStreamingFilterEntry()] }
          : entry
      )
    );
  }, []);

  const removeStreamingFilter = useCallback((datasetId: string, filterId: string) => {
    setStreamingDatasets((previous) =>
      previous.map((entry) => {
        if (entry.id !== datasetId) {
          return entry;
        }
        return {
          ...entry,
          filters: entry.filters.filter((filter) => filter.id !== filterId),
        };
      })
    );
  }, []);

  const removeLocalTrainFile = useCallback((localFileId: string) => {
    setLocalTrainFiles((previous) => previous.filter((entry) => entry.id !== localFileId));
  }, []);

  const clearLocalTrainFiles = useCallback(() => {
    setLocalTrainFiles([]);
  }, []);

  const uploadLocalTrainFiles = useCallback(
    async (selectedFiles: File[]) => {
      if (selectedFiles.length === 0) {
        return;
      }

      notify(
        "info",
        "Uploading dataset files",
        selectedFiles.length === 1
          ? `Uploading ${selectedFiles[0].name}.`
          : `Uploading ${selectedFiles.length} local train files.`
      );
      setIsUploadingTrainFile(true);

      try {
        const uploadResults = await Promise.allSettled(
          selectedFiles.map((file) => uploadTrainFile(file))
        );

        const successfulUploads = uploadResults
          .filter(
            (
              result
            ): result is PromiseFulfilledResult<Awaited<ReturnType<typeof uploadTrainFile>>> =>
              result.status === "fulfilled"
          )
          .map((result) => result.value);

        if (successfulUploads.length > 0) {
          setLocalTrainFiles((previous) =>
            normalizeLocalTrainFiles([
              ...previous,
              ...successfulUploads.map((uploadedFile) =>
                makeLocalTrainFileEntry({
                  fileName: uploadedFile.file_name,
                  filePath: uploadedFile.file_path,
                  sizeBytes: uploadedFile.size_bytes,
                  sizeChars: uploadedFile.size_chars,
                })
              ),
            ])
          );
          notify(
            "success",
            "Dataset files added",
            successfulUploads.length === 1
              ? `Added ${stripGeneratedUploadPrefix(successfulUploads[0].file_name)}.`
              : `Added ${successfulUploads.length} local train files.`
          );
        }

        const failedUploads = uploadResults.filter(
          (result): result is PromiseRejectedResult => result.status === "rejected"
        );
        if (failedUploads.length > 0) {
          const firstFailure = failedUploads[0];
          const firstFailureMessage =
            firstFailure.reason instanceof Error ? firstFailure.reason.message : "Upload failed";
          notify(
            "error",
            "Dataset upload failed",
            `Failed to upload ${failedUploads.length} file(s). ${firstFailureMessage}`
          );
        }
      } catch (error) {
        notify(
          "error",
          "Dataset upload failed",
          error instanceof Error ? error.message : "Failed to upload local train files."
        );
      } finally {
        setIsUploadingTrainFile(false);
      }
    },
    [notify]
  );

  const handleTrainFilesSelected = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const selectedFiles = Array.from(event.target.files ?? []);
      event.target.value = "";
      await uploadLocalTrainFiles(selectedFiles);
    },
    [uploadLocalTrainFiles]
  );

  const handleLocalTrainFilesDragEnter = useCallback((event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (!Array.from(event.dataTransfer.types).includes("Files")) {
      return;
    }
    localFileDragDepthRef.current += 1;
    setIsDraggingTrainFiles(true);
  }, []);

  const handleLocalTrainFilesDragOver = useCallback((event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = "copy";
    setIsDraggingTrainFiles(true);
  }, []);

  const handleLocalTrainFilesDragLeave = useCallback((event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    localFileDragDepthRef.current = Math.max(0, localFileDragDepthRef.current - 1);
    if (localFileDragDepthRef.current === 0) {
      setIsDraggingTrainFiles(false);
    }
  }, []);

  const handleLocalTrainFilesDrop = useCallback(
    async (event: DragEvent<HTMLElement>) => {
      event.preventDefault();
      event.stopPropagation();
      localFileDragDepthRef.current = 0;
      setIsDraggingTrainFiles(false);
      const droppedFiles = Array.from(event.dataTransfer.files ?? []);
      await uploadLocalTrainFiles(droppedFiles);
    },
    [uploadLocalTrainFiles]
  );

  const handleLoadStreamingTemplate = useCallback(async () => {
    notify("info", "Loading streaming template", "Refreshing dataset settings from the template.");
    setIsLoadingDatasetTemplate(true);
    try {
      const templates = await fetchTrainingConfigTemplates();
      const templateDataloaderConfig = cloneRecord(
        asRecord(templates.dataloader_config_template)
      );
      const hydrated = hydrateDatasetUiFromConfig(templateDataloaderConfig);
      setDatasetSourceMode("streaming_hf");
      setHfToken(hydrated.hfToken);
      setStreamingDatasets(hydrated.streamingDatasets);
      setDataloaderConfig(templateDataloaderConfig);
      notify("success", "Template loaded", "Loaded streaming dataset defaults.");
    } catch (error) {
      notify(
        "error",
        "Template unavailable",
        error instanceof Error ? error.message : "Failed to load the streaming dataset template."
      );
    } finally {
      setIsLoadingDatasetTemplate(false);
    }
  }, [notify]);

  const handleAddPrompt = () => {
    const next = cloneRecord(trainingConfig ?? {});
    const sampler = asRecord(next.sampler);
    const prompts = asRecordArray(sampler.prompts);
    prompts.push({
      prompt: "New prompt",
      max_tokens: 64,
      temperature: 0.7,
      top_k: 40,
    });
    sampler.prompts = prompts;
    next.sampler = sampler;
    setTrainingConfig(next);
  };

  const handlePromptChange = (index: number, field: string, value: unknown) => {
    const next = cloneRecord(trainingConfig ?? {});
    const sampler = asRecord(next.sampler);
    const prompts = asRecordArray(sampler.prompts);
    const prompt = cloneRecord(prompts[index] ?? {});
    prompt[field] = value;
    prompts[index] = prompt;
    sampler.prompts = prompts;
    next.sampler = sampler;
    setTrainingConfig(next);
  };

  const handleRemovePrompt = (index: number) => {
    const next = cloneRecord(trainingConfig ?? {});
    const sampler = asRecord(next.sampler);
    sampler.prompts = asRecordArray(sampler.prompts).filter((_, currentIndex) => currentIndex !== index);
    next.sampler = sampler;
    setTrainingConfig(next);
  };

  const handleTrainingField = (path: string[], value: unknown) => {
    setTrainingConfig((current) => updateAtPath(current ?? {}, path, value));
  };

  const handleOptionalTrainingField = (path: string[], value: unknown | null) => {
    setTrainingConfig((current) =>
      value === null
        ? deleteAtPath(current ?? {}, path)
        : updateAtPath(current ?? {}, path, value)
    );
  };

  const handleDataloaderField = (path: string[], value: unknown) => {
    setDataloaderConfig((current) => updateAtPath(current ?? {}, path, value));
  };

  const applyFix = (fix: TrainingFixSuggestion) => {
    if (fix.path.startsWith("training_config.")) {
      const path = fix.path.replace("training_config.", "").split(".");
      setTrainingConfig((current) => updateAtPath(current ?? {}, path, cloneRecord(fix.value)));
      notify("success", fix.label, fix.description);
      return;
    }
    if (fix.path.startsWith("dataloader_config.")) {
      const path = fix.path.replace("dataloader_config.", "").split(".");
      setDataloaderConfig((current) => updateAtPath(current ?? {}, path, cloneRecord(fix.value)));
      notify("success", fix.label, fix.description);
    }
  };

  const closePicker = useCallback(() => {
    pickerRequestIdRef.current += 1;
    setPickerKind(null);
    setPickerQuery("");
    setPickerError(null);
    setPickerLoading(false);
  }, []);

  const openPicker = useCallback(async (kind: AssetPickerKind) => {
    const requestId = pickerRequestIdRef.current + 1;
    pickerRequestIdRef.current = requestId;

    setPickerKind(kind);
    setPickerQuery("");
    setPickerError(null);
    setPickerLoading(true);

    try {
      if (kind === "project") {
        const projects = await fetchProjects();
        if (pickerRequestIdRef.current !== requestId) {
          return;
        }
        startTransition(() => {
          setPickerProjects(projects);
        });
        return;
      }

      const jobs = await fetchTokenizerJobs();
      if (pickerRequestIdRef.current !== requestId) {
        return;
      }
      startTransition(() => {
        setPickerTokenizerJobs(jobs);
      });
    } catch (error) {
      if (pickerRequestIdRef.current !== requestId) {
        return;
      }
      setPickerError(error instanceof Error ? error.message : "Failed to load workspace assets.");
    } finally {
      if (pickerRequestIdRef.current === requestId) {
        setPickerLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    if (!pickerKind) {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closePicker();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [closePicker, pickerKind]);

  const handleStartTraining = async () => {
    if (!selectedProjectId || !selectedTokenizerJobId || !trainingConfig || !dataloaderConfig || !preflight?.valid) {
      notify("error", "Training blocked", "Resolve the preflight issues before launching.");
      return;
    }
    setLaunching(true);
    try {
      const job = await createTrainingJob({
        name: runName.trim() || undefined,
        project_id: selectedProjectId,
        tokenizer_job_id: selectedTokenizerJobId,
        training_config: trainingConfig,
        dataloader_config: dataloaderConfig,
      });
      startTransition(() => {
        setActiveRunId(job.id);
        setActiveRun(job);
        setRecentRuns((current) => [job, ...current.filter((item) => item.id !== job.id)]);
      });
      notify("success", "Training launched", `Run ${job.name} is now being tracked live.`);
    } catch (error) {
      notify("error", "Launch failed", error instanceof Error ? error.message : "Failed to start training.");
    } finally {
      setLaunching(false);
    }
  };

  const handleStopTraining = async () => {
    if (!activeRunId) {
      return;
    }
    try {
      const job = await stopTrainingJob(activeRunId);
      startTransition(() => {
        setActiveRun(job);
        setRecentRuns((current) => replaceRunInOrder(current, job));
      });
      notify("success", "Training stopped", `Run ${job.name} was cancelled.`);
    } catch (error) {
      notify("error", "Stop failed", error instanceof Error ? error.message : "Failed to stop the active run.");
    }
  };

  const handleDeleteRun = async (jobId: string) => {
    try {
      await deleteTrainingJob(jobId);
      startTransition(() => {
        setRecentRuns((current) => current.filter((job) => job.id !== jobId));
        if (activeRunId === jobId) {
          setActiveRunId(null);
        }
      });
      notify("success", "Run removed", "The training run was removed from the workspace.");
    } catch (error) {
      notify("error", "Delete failed", error instanceof Error ? error.message : "Failed to delete the training run.");
    }
  };

  const startReady = Boolean(preflight?.valid && selectedProjectId && selectedTokenizerJobId && !launching);
  const trainingRuntimeReady = Boolean(trainingConfig && dataloaderConfig);
  const hasTrainingInProgress =
    activeRun?.status === "running" || activeRun?.status === "pending";
  const trainingCompleted = activeRun?.status === "completed";
  const sequenceLength = trainingConfig ? asNumber(trainingConfig.seq_len, 0) : 0;
  const maxSteps = trainingConfig ? asNumber(trainingConfig.max_steps, 0) : 0;
  const datasetSummary =
    datasetSourceMode === "local_file"
      ? `${localTrainFiles.length} local file${localTrainFiles.length === 1 ? "" : "s"}`
      : `${streamingDatasets.length} streaming dataset${
          streamingDatasets.length === 1 ? "" : "s"
        }`;
  const workflowSteps = [
    {
      title: "Step 1 - Choose saved model",
      state: selectedProject ? "ready" : "waiting",
      status: selectedProject ? "Ready" : "Waiting for configuration",
      body: selectedProject
        ? selectedProject.name ?? selectedProject.id
        : "Pick a saved model config from the home workspace or query parameters.",
      actionLabel: "Open model selection",
      onAction: () => openWorkflowTarget("model"),
    },
    {
      title: "Step 2 - Choose tokenizer artifact",
      state:
        selectedTokenizer && selectedTokenizer.status === "completed"
          ? "ready"
          : "waiting",
      status:
        selectedTokenizer && selectedTokenizer.status === "completed"
          ? "Ready"
          : "Waiting for configuration",
      body:
        selectedTokenizer && selectedTokenizer.status === "completed"
          ? asString(selectedTokenizer.tokenizer_config.name, selectedTokenizer.id)
          : "Select a completed tokenizer artifact to ensure vocabulary compatibility.",
      actionLabel: "Open tokenizer selection",
      onAction: () => openWorkflowTarget("tokenizer"),
    },
    {
      title: "Step 3 - Configure training run",
      state: trainingRuntimeReady ? "ready" : "waiting",
      status: trainingRuntimeReady ? "Ready" : "Waiting for configuration",
      body: trainingRuntimeReady
        ? `Sequence length ${formatInteger(sequenceLength)}, maximum training steps ${formatInteger(
            maxSteps
          )}, ${datasetSummary} configured.`
        : "Tune sequence length, batch size, save cadence, prompts, and dataset sources.",
      actionLabel: "Open training settings",
      onAction: () => openWorkflowTarget("training"),
    },
    {
      title: "Step 4 - Validate configurations",
      state: preflight?.valid ? "ready" : preflightLoading ? "inProgress" : "waiting",
      status: preflight?.valid
        ? "Ready"
        : preflightLoading
          ? "In progress"
          : "Waiting for configuration",
      body: preflight?.valid
        ? "Preflight passed for compatibility, runtime math, and memory checks."
        : preflightLoading
          ? "Validating the latest training and dataset configuration changes..."
          : compactWorkflowMessage(
              preflightError ??
                preflight?.errors[0]?.message ??
                "Complete steps 1-3 first. Preflight runs automatically."
            ),
      actionLabel: "Review preflight",
      onAction: () => openWorkflowTarget("preflight"),
    },
    {
      title: "Step 5 - Start training",
      state: trainingCompleted
        ? "ready"
        : hasTrainingInProgress
          ? "inProgress"
          : "waiting",
      status: trainingCompleted
        ? "Ready (trained)"
        : hasTrainingInProgress
          ? "In progress"
          : "Not ready",
      body: trainingCompleted
        ? "Latest training run completed. Artifacts and telemetry are ready."
        : hasTrainingInProgress
          ? `Current run is ${activeRun?.status ?? "running"}.`
          : startReady
            ? "Preflight passed. Start training to complete this step."
            : preflightLoading
              ? "Waiting for automatic preflight to finish."
              : "A passing preflight is required before launch.",
    },
  ];

  return (
    <main className="studioRoot trainingPage">
      <header className="studioNav" role="navigation" aria-label="Primary">
        <div className="studioNavBrand">
          <span className="studioNavDot" />
          <span>LLM Builder</span>
        </div>
        <div className="studioNavLinks">
          <Link className="studioNavLink" href="/">
            Home
          </Link>
          <Link className="studioNavLink" href="/studio">
            LLM Studio
          </Link>
          <Link className="studioNavLink" href="/tokenizer">
            Tokenizer Studio
          </Link>
          <Link className="studioNavLink" href="/training" aria-current="page">
            Training
          </Link>
        </div>
        <button
          type="button"
          className="themeToggle"
          onClick={() => setTheme((current) => (current === "dark" ? "white" : "dark"))}
          aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
        >
          {theme === "dark" ? <FiSun /> : <FiMoon />}
        </button>
      </header>

      <section className="panelCard heroCard trainingHero">
        <div className="panelHead heroHead">
          <div>
            <h1>Launch training with asset pairing, preflight, and live telemetry.</h1>
            <p className="panelCopy">
              Pair a saved model config with a completed tokenizer, validate the full runtime before launch,
              and monitor loss, learning rate, throughput, samples, checkpoints, and logs from one workspace.
            </p>
          </div>
          <div className="actionCluster trainingHeroActions">
            <button
              type="button"
              className="buttonPrimary"
              onClick={handleStartTraining}
              disabled={!startReady}
            >
              <FiPlay /> {launching ? "Launching…" : "Start Training"}
            </button>
            <button
              type="button"
              className="buttonDanger"
              onClick={handleStopTraining}
              disabled={!activeRunId || activeRun?.status !== "running"}
            >
              <FiXCircle /> Stop Training
            </button>
            <Link className="buttonGhost" href="/">
              <FiLayers /> Open Workspace Assets
            </Link>
            {activeRunId ? (
              <a className="buttonGhost" href={trainingArtifactDownloadUrl(activeRunId)}>
                <FiDownload /> Download Run Bundle
              </a>
            ) : null}
          </div>
        </div>

        <div className="heroMetaRow" aria-label="Training launch summary">
          <div className="heroMetaPills">
            <span className={`pillBadge ${selectedProject ? "tone-good" : "tone-warn"}`}>
              {selectedProject ? "Model selected" : "Model needed"}
            </span>
            <span className={`pillBadge ${selectedTokenizer?.status === "completed" ? "tone-good" : "tone-warn"}`}>
              {selectedTokenizer?.status === "completed" ? "Tokenizer ready" : "Tokenizer needed"}
            </span>
            <span className={`pillBadge ${preflight?.valid ? "tone-good" : preflightLoading ? "tone-neutral" : "tone-warn"}`}>
              {preflightLoading ? "Running preflight" : preflight?.valid ? "Preflight passing" : "Preflight blocked"}
            </span>
          </div>
          <div className="heroMetaLine">
            <span>{selectedProject?.name ?? "No model config selected"}</span>
            <span className="heroMetaSeparator" aria-hidden>
              •
            </span>
            <span>
              {selectedTokenizer
                ? asString(selectedTokenizer.tokenizer_config.name, selectedTokenizer.id)
                : "No tokenizer selected"}
            </span>
            <span className="heroMetaSeparator" aria-hidden>
              •
            </span>
            <span>{activeRun ? `Tracking ${activeRun.name}` : "Ready to stage a new run"}</span>
          </div>
        </div>

        <div className="trainingHeroBody">
          <div className="trainingPairGrid">
            <div
              id="settings-model"
              ref={modelSelectionRef}
              className={`trainingAssetCard settingsCategoryAnchor ${
                highlightedWorkflowTarget === "model"
                  ? "settingsCategoryAnchor-highlight"
                  : ""
              }`}
            >
              <span className="trainingAssetLabel">Model Config</span>
              <span className="trainingAssetName">
                {selectedProject?.name ?? (selectedProjectId ? selectedProjectId : "No model selected")}
              </span>
              <span className="trainingAssetMeta">
                {selectedProject
                  ? formatModelConfigMeta(selectedProject.model_config)
                  : "Choose a saved model config from Home or pass ?project=..."}
              </span>
              {selectedProject ? (
                <span className="trainingAssetMeta">
                  {selectedProject.artifact_file || "Saved model config artifact"}
                </span>
              ) : null}
              <div className="trainingAssetActions">
                <button
                  type="button"
                  className="buttonGhost buttonSmall"
                  aria-haspopup="dialog"
                  aria-expanded={pickerKind === "project"}
                  onClick={() => {
                    void openPicker("project");
                  }}
                >
                  <FiSearch /> {selectedProject ? "Change model config" : "Choose model config"}
                </button>
              </div>
            </div>
            <div
              id="settings-tokenizer"
              ref={tokenizerSelectionRef}
              className={`trainingAssetCard settingsCategoryAnchor ${
                highlightedWorkflowTarget === "tokenizer"
                  ? "settingsCategoryAnchor-highlight"
                  : ""
              }`}
            >
              <span className="trainingAssetLabel">Tokenizer Artifact</span>
              <span className="trainingAssetName">
                {selectedTokenizer
                  ? asString(selectedTokenizer.tokenizer_config.name, selectedTokenizer.id)
                  : selectedTokenizerJobId ?? "No tokenizer selected"}
              </span>
              <span className="trainingAssetMeta">
                {selectedTokenizer
                  ? formatTokenizerMeta(selectedTokenizer)
                  : "Choose a completed tokenizer artifact from Home or pass ?tokenizerJob=..."}
              </span>
              {selectedTokenizer ? (
                <span className="trainingAssetMeta">
                  {selectedTokenizer.artifact_file ??
                    selectedTokenizer.artifact_path ??
                    "Tokenizer artifact path unavailable"}
                </span>
              ) : null}
              <div className="trainingAssetActions">
                <button
                  type="button"
                  className="buttonGhost buttonSmall"
                  aria-haspopup="dialog"
                  aria-expanded={pickerKind === "tokenizer"}
                  onClick={() => {
                    void openPicker("tokenizer");
                  }}
                >
                  <FiSearch /> {selectedTokenizer ? "Change tokenizer" : "Choose tokenizer"}
                </button>
              </div>
            </div>
          </div>

          <div className="fieldGrid compact">
            <label className="fieldLabel fullWidthField">
              <span>Run name</span>
              <input value={runName} onChange={(event) => setRunName(event.target.value)} placeholder="optional training run name" />
            </label>
          </div>
        </div>
      </section>

      <section id="workflow" className="panelCard actionDeck">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">Top Workflow</p>
            <h2>Steps to train the model</h2>
            <p className="panelCopy">
              Complete each step in order. A step turns green only when it is ready.
            </p>
          </div>
        </div>
        <div className="workflowStepGrid" role="list" aria-label="Training launch steps">
          {workflowSteps.map((step, index) => (
            <article
              key={step.title}
              className={`workflowStepTile workflowStepTile-${
                step.state === "ready"
                  ? "ready"
                  : step.state === "inProgress"
                    ? "inProgress"
                    : "waiting"
              }`}
              role="listitem"
            >
              <p className="workflowStepTitle">{step.title}</p>
              <strong>{formatStatusLabel(step.status)}</strong>
              <p className="fieldNote">{step.body}</p>
              {step.onAction && step.actionLabel ? (
                <button
                  type="button"
                  className={`${
                    index === 3
                      ? "secondaryButton workflowStepAction workflowStepButtonCompact"
                      : "workflowStepLink workflowStepAction"
                  }`}
                  onClick={step.onAction}
                >
                  {step.actionLabel}
                </button>
              ) : (
                <button
                  type="button"
                  className="primaryButton workflowStepAction"
                  onClick={handleStartTraining}
                  disabled={!startReady}
                >
                  {hasTrainingInProgress
                    ? "Training..."
                    : launching
                      ? "Starting..."
                      : "Start Training"}
                </button>
              )}
            </article>
          ))}
        </div>
      </section>

      <section className="trainingResultsGrid">
        <div className="trainingPanelStack">
          <section
            id="settings-preflight"
            ref={preflightSectionRef}
            className={`panelCard settingsCategoryAnchor ${
              highlightedWorkflowTarget === "preflight"
                ? "settingsCategoryAnchor-highlight"
                : ""
            }`}
          >
              <div className="panelHead">
                <div>
                  <h2>Preflight</h2>
                  <p className="panelCopy">
                    Compatibility, scheduler math, local dataset paths, special tokens, and runtime memory are validated here.
                  </p>
                </div>
                {preflight ? (
                  <span className={`pillBadge ${preflight.valid ? "tone-good" : "tone-error"}`}>
                    {preflight.valid ? "Ready to launch" : `${preflight.errors.length} blocking issues`}
                  </span>
                ) : null}
              </div>

              {preflightError ? <div className="trainingIssueCard tone-error">{preflightError}</div> : null}

              {preflight?.compatibility ? (
                <div className="statusGrid">
                  <div className={`statusCard ${preflight.valid ? "tone-good" : "tone-bad"}`}>
                    <div className="statusCardIcon">
                      <FiArchive />
                    </div>
                    <div>
                      <div className="statusCardTitle">Tokenizer vocabulary</div>
                      <div className="statusCardValue">{formatInteger(preflight.compatibility.tokenizer_vocab_size)}</div>
                      <div className="statusCardDetail">Model vocabulary size: {formatInteger(preflight.compatibility.model_vocab_size)}</div>
                    </div>
                  </div>
                  <div className="statusCard">
                    <div className="statusCardIcon">
                      <FiLayers />
                    </div>
                    <div>
                      <div className="statusCardTitle">Sequence length</div>
                      <div className="statusCardValue">{formatInteger(preflight.compatibility.seq_len)}</div>
                      <div className="statusCardDetail">Model context limit: {formatInteger(preflight.compatibility.model_context_length)}</div>
                    </div>
                  </div>
                  <div className="statusCard">
                    <div className="statusCardIcon">
                      <FiBarChart2 />
                    </div>
                    <div>
                      <div className="statusCardTitle">Micro batch size</div>
                      <div className="statusCardValue">
                        {formatInteger(preflight.derived_runtime?.micro_batch_size ?? null)}
                      </div>
                      <div className="statusCardDetail">
                        Gradient accumulation steps: {formatInteger(preflight.derived_runtime?.grad_accum_steps ?? null)}
                      </div>
                    </div>
                  </div>
                  <div className="statusCard">
                    <div className="statusCardIcon">
                      <FiCpu />
                    </div>
                    <div>
                      <div className="statusCardTitle">Device</div>
                      <div className="statusCardValue">{preflight.derived_runtime?.device_type ?? "n/a"}</div>
                      <div className="statusCardDetail">
                        Memory-estimated max batch size: {formatInteger((preflight.memory_estimate?.max_batch_size as number | undefined) ?? null)}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="trainingEmpty">
                  Select both assets and finish the configuration fields to run preflight.
                </div>
              )}

              {preflight?.errors.length ? (
                <div className="trainingIssueList">
                  {preflight.errors.map((item) => (
                    <div key={`${item.code}-${item.path}`} className={`trainingIssueCard tone-${issueTone(item)}`}>
                      <div className="trainingIssueTitle">{item.message}</div>
                      <div className="trainingIssueMeta">{item.path}</div>
                    </div>
                  ))}
                </div>
              ) : null}

              {preflight?.warnings.length ? (
                <details className="sectionDisclosure" open>
                  <summary className="sectionDisclosureSummary">Warnings</summary>
                  <div className="trainingIssueList">
                    {preflight.warnings.map((item) => (
                      <div key={`${item.code}-${item.path}`} className="trainingIssueCard tone-warning">
                        <div className="trainingIssueTitle">{item.message}</div>
                        <div className="trainingIssueMeta">{item.path}</div>
                      </div>
                    ))}
                  </div>
                </details>
              ) : null}

              {preflight?.recommended_fixes.length ? (
                <details className="sectionDisclosure" open>
                  <summary className="sectionDisclosureSummary">Suggested fixes</summary>
                  <div className="trainingFixList">
                    {preflight.recommended_fixes.map((fix) => (
                      <div key={fix.code} className="trainingFixCard">
                        <div className="trainingIssueTitle">{fix.label}</div>
                        <div className="trainingIssueMeta">{fix.description}</div>
                        <div className="trainingFixActions">
                          <button type="button" className="buttonGhost buttonSmall" onClick={() => applyFix(fix)}>
                            Apply
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </details>
              ) : null}
            </section>

          <section className="panelCard">
              <div className="panelHead">
                <div>
                  <h2>Active Run</h2>
                  <p className="panelCopy">The monitor updates every {Math.round(POLL_INTERVAL_MS / 1000)} seconds with summary, metrics, samples, checkpoints, and logs.</p>
                </div>
                {activeRun ? (
                  <span className={`pillBadge ${statusTone(activeRun.status)}`}>{formatStatusLabel(activeRun.status)}</span>
                ) : null}
              </div>

              {activeRun ? (
                <>
                  <div className="trainingProgress">
                    <div className="trainingSectionHeader">
                      <h3>{activeRun.name}</h3>
                      <span className="pillBadge tone-neutral">{activeRun.stage}</span>
                    </div>
                    <div className="trainingProgressBar">
                      <span style={{ width: `${Math.max(0, Math.min(activeRun.progress, 1)) * 100}%` }} />
                    </div>
                    <div className="trainingInlineMeta">
                      <span>Run identifier: {activeRun.id.slice(0, 8)}</span>
                      <span>Created {formatDate(activeRun.created_at)}</span>
                      <span>Started {activeRun.started_at ? formatDate(activeRun.started_at) : "waiting"}</span>
                      <span>Estimated time remaining: {formatDuration((activeRun as unknown as { eta_seconds?: number | null }).eta_seconds ?? null)}</span>
                    </div>
                  </div>

                  <div className="statusGrid">
                    <div className="statusCard">
                      <div className="statusCardIcon"><FiActivity /></div>
                      <div>
                        <div className="statusCardTitle">Training step</div>
                        <div className="statusCardValue">
                          {formatInteger(activeRun.last_step)} / {formatInteger(activeRun.max_steps)}
                        </div>
                        <div className="statusCardDetail">Training progress: {Math.round(activeRun.progress * 100)}%</div>
                      </div>
                    </div>
                    <div className="statusCard">
                      <div className="statusCardIcon"><FiCheckCircle /></div>
                      <div>
                        <div className="statusCardTitle">Loss</div>
                        <div className="statusCardValue">{formatMetric(activeRun.latest_loss, 4)}</div>
                        <div className="statusCardDetail">Gradient norm: {formatMetric(activeRun.latest_grad_norm, 3)}</div>
                      </div>
                    </div>
                    <div className="statusCard">
                      <div className="statusCardIcon"><FiRefreshCw /></div>
                      <div>
                        <div className="statusCardTitle">Learning Rate</div>
                        <div className="statusCardValue">{formatMetric(activeRun.latest_lr, 6)}</div>
                        <div className="statusCardDetail">Tokens per second: {formatInteger(activeRun.latest_tokens_per_sec)}</div>
                      </div>
                    </div>
                    <div className="statusCard">
                      <div className="statusCardIcon"><FiArchive /></div>
                      <div>
                        <div className="statusCardTitle">Saved artifacts</div>
                        <div className="statusCardValue">
                          {formatInteger(activeRun.checkpoint_count)} checkpoints
                        </div>
                        <div className="statusCardDetail">
                          {formatInteger(activeRun.sample_count)} sample groups • {formatBytes(activeRun.output_size_bytes)}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="trainingChartGrid">
                    <MetricChart
                      title="Loss"
                      metricKey="loss"
                      metrics={metrics}
                      latestValue={formatMetric(activeRun.latest_loss, 4)}
                      stroke="var(--brand)"
                      digits={4}
                    />
                    <MetricChart
                      title="Learning Rate"
                      metricKey="lr"
                      metrics={metrics}
                      latestValue={formatMetric(activeRun.latest_lr, 6)}
                      stroke="var(--ok)"
                      digits={6}
                    />
                    <MetricChart
                      title="Gradient Norm"
                      metricKey="norm"
                      metrics={metrics}
                      latestValue={formatMetric(activeRun.latest_grad_norm, 3)}
                      stroke="var(--warn)"
                      digits={3}
                    />
                    <MetricChart
                      title="Throughput"
                      metricKey="tok_per_sec"
                      metrics={metrics}
                      latestValue={formatInteger(activeRun.latest_tokens_per_sec)}
                      stroke="var(--danger)"
                      digits={1}
                    />
                  </div>

                  <details className="sectionDisclosure" open>
                    <summary className="sectionDisclosureSummary">Samples</summary>
                    <div className="trainingSampleList">
                      {samples.length ? (
                        samples.slice().reverse().map((entry, entryIndex) => {
                          const sampleCount = entry.samples.length;
                          const totalChars = entry.samples.reduce(
                            (sum, sample) => sum + sample.text.length + (sample.prompt?.length ?? 0),
                            0
                          );

                          return (
                            <details
                              key={`sample-${entry.step}`}
                              className="trainingSampleCard trainingSampleStepDisclosure"
                              open={entryIndex === 0}
                            >
                              <summary className="trainingSampleStepSummary">
                                <span>
                                  <span className="trainingSampleTitle">Step {entry.step}</span>
                                  <span className="trainingSampleMeta">
                                    {sampleCount} prompt{sampleCount === 1 ? "" : "s"} generated
                                    {" - "}
                                    {formatInteger(totalChars)} characters
                                  </span>
                                </span>
                              </summary>

                              <div className="trainingSampleStepBody">
                                {entry.samples.map((sample) => (
                                  <details
                                    key={`${entry.step}-${sample.index}`}
                                    className="trainingSampleTextDisclosure"
                                  >
                                    <summary className="trainingSampleTextSummary">
                                      <span>Prompt {sample.index + 1}</span>
                                      <span className="trainingSampleMeta">
                                        {formatInteger(sample.text.length)} generated characters
                                      </span>
                                    </summary>
                                    {sample.prompt ? (
                                      <div className="trainingSamplePromptBlock">
                                        <span>Prompt</span>
                                        <pre>{sample.prompt}</pre>
                                      </div>
                                    ) : null}
                                    <pre className="trainingCodeBlock trainingSampleCodeBlock">{sample.text}</pre>
                                  </details>
                                ))}
                              </div>
                            </details>
                          );
                        })
                      ) : (
                        <div className="trainingEmpty">No samples have been recorded yet.</div>
                      )}
                    </div>
                  </details>

                  <details className="sectionDisclosure" open>
                    <summary className="sectionDisclosureSummary">Checkpoints</summary>
                    <div className="trainingCheckpointList">
                      {checkpoints.length ? (
                        checkpoints.map((checkpoint) => (
                          <div key={checkpoint.directory} className="trainingCheckpointCard">
                            <div className="trainingCheckpointTitle">Step {checkpoint.step}</div>
                            <div className="trainingCheckpointMeta">
                              {checkpoint.created_at ? formatDate(checkpoint.created_at) : "Created time unavailable"} • {formatBytes(checkpoint.size_bytes)}
                            </div>
                            <div className="trainingCheckpointMeta">{checkpoint.files.join(", ")}</div>
                          </div>
                        ))
                      ) : (
                        <div className="trainingEmpty">No checkpoints are available yet.</div>
                      )}
                    </div>
                  </details>

                  <details className="sectionDisclosure" open>
                    <summary className="sectionDisclosureSummary">Logs</summary>
                    <div className="trainingDualLog">
                      <div className="trainingLogBox">{logs.stdout.join("\n") || "stdout is quiet so far."}</div>
                      <div className="trainingLogBox">{logs.stderr.join("\n") || "stderr is clear."}</div>
                    </div>
                  </details>

                  <details className="sectionDisclosure">
                    <summary className="sectionDisclosureSummary">Resolved runtime and configurations</summary>
                    <div className="trainingJsonGrid">
                      <pre className="trainingCodeBlock">{prettyJson(activeRun.resolved_runtime)}</pre>
                      <pre className="trainingCodeBlock">{prettyJson(activeRun.memory_estimate)}</pre>
                    </div>
                  </details>
                </>
              ) : (
                <div className="trainingEmpty">No active run selected. Launch a new run or choose one from the recent runs column.</div>
              )}
          </section>
        </div>

        <div className="trainingPanelStack">
          <section className="panelCard">
              <div className="panelHead">
                <div>
                  <h2>Recent Runs</h2>
                  <p className="panelCopy">Recent jobs stay navigable after refresh so you can jump between current and past runs quickly.</p>
                </div>
                <button type="button" className="buttonGhost buttonSmall" onClick={() => void refreshRecentRuns()}>
                  Refresh
                </button>
              </div>
              <div className="trainingRecentList">
                {recentRuns.length ? (
                  recentRuns.map((job) => (
                    <div key={job.id} className={`trainingRecentCard ${activeRunId === job.id ? "is-active" : ""}`}>
                      <button
                        type="button"
                        className="trainingRecentSelect"
                        onClick={() => setActiveRunId(job.id)}
                      >
                        <div>
                          <strong className="trainingRecentTitle">{job.name}</strong>
                          <p>{job.project_name} / {job.tokenizer_name}</p>
                        </div>
                        <div className="trainingRecentRowMeta">
                          <span className={`pillBadge ${statusTone(job.status)}`}>{formatStatusLabel(job.status)}</span>
                          <p>{formatDate(job.created_at)}</p>
                        </div>
                      </button>
                      <div className="trainingRecentIconActions">
                        <button
                          type="button"
                          className="trainingRecentIconButton trainingRecentIconButton-danger"
                          onClick={() => void handleDeleteRun(job.id)}
                          disabled={job.status === "running" || job.status === "pending"}
                          aria-label={`Delete ${job.name}`}
                          title="Delete run"
                        >
                          <FiTrash2 aria-hidden="true" />
                        </button>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="trainingEmpty">
                    No training runs yet.
                  </div>
                )}
              </div>
          </section>
        </div>
      </section>

      <section className="panelCard trainingSettingsStudio">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">Settings</p>
            <h2>Configuration Studio</h2>
            <p className="panelCopy">
              Core launch fields stay visible. Dataset, prompts, and runtime controls live in
              foldable sections so the page stays compact like the tokenizer workflow.
            </p>
          </div>
          <div className="heroMetaPills">
            <span className="pillBadge tone-neutral">{datasetEntries.length} dataset source{datasetEntries.length === 1 ? "" : "s"}</span>
            <span className="pillBadge tone-neutral">{promptEntries.length} prompt{promptEntries.length === 1 ? "" : "s"}</span>
          </div>
        </div>

        {trainingConfig && dataloaderConfig ? (
          <div className="trainingSettingsStack">
            <details className="settingsPanel" open ref={trainingPlanPanelRef}>
              <summary>Training plan</summary>
              <div className="settingsGrid">
                <div
                  id="settings-training"
                  ref={trainingSettingsRef}
                  className={`settingsGroup settingsCategoryAnchor ${
                    highlightedWorkflowTarget === "training"
                      ? "settingsCategoryAnchor-highlight"
                      : ""
                  }`}
                >
                  <div className="settingsGroupHeader">
                    <h3>Core launch knobs</h3>
                    <p className="settingsGroupHint">
                      Tune the values you are most likely to touch between runs before opening the
                      deeper runtime controls.
                    </p>
                  </div>
                  <div className="fieldGrid trainingSettingsCompactGrid">
                    <label className="fieldLabel">
                      <span>Sequence length</span>
                      <ConfigNumberInput
                        value={asNumber(trainingConfig.seq_len, 128)}
                        onCommit={(value) => handleTrainingField(["seq_len"], value)}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Maximum training steps</span>
                      <ConfigNumberInput
                        value={asNumber(trainingConfig.max_steps, 0)}
                        onCommit={(value) => handleTrainingField(["max_steps"], value)}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Total batch size</span>
                      <ConfigNumberInput
                        value={asNumber(trainingConfig.total_batch_size, 0)}
                        onCommit={(value) => handleTrainingField(["total_batch_size"], value)}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Micro batch size <small>optional</small></span>
                      <OptionalConfigNumberInput
                        value={
                          typeof trainingConfig.micro_batch_size === "number"
                            ? trainingConfig.micro_batch_size
                            : null
                        }
                        onCommit={(value) =>
                          handleOptionalTrainingField(["micro_batch_size"], value)
                        }
                        placeholder="Auto"
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Learning rate</span>
                      <ConfigNumberInput
                        mode="decimal"
                        step="0.000001"
                        value={asNumber(asRecord(trainingConfig.optimizer).lr, 0.0003)}
                        onCommit={(value) => handleTrainingField(["optimizer", "lr"], value)}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Weight decay</span>
                      <ConfigNumberInput
                        mode="decimal"
                        step="0.0001"
                        value={asNumber(asRecord(trainingConfig.optimizer).weight_decay, 0.1)}
                        onCommit={(value) =>
                          handleTrainingField(
                            ["optimizer", "weight_decay"],
                            value
                          )
                        }
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Save checkpoint every</span>
                      <ConfigNumberInput
                        value={asNumber(trainingConfig.save_every, 0)}
                        onCommit={(value) => handleTrainingField(["save_every"], value)}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Generate samples every</span>
                      <ConfigNumberInput
                        value={asNumber(trainingConfig.sample_every, 0)}
                        onCommit={(value) => handleTrainingField(["sample_every"], value)}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Dataset shuffle buffer</span>
                      <ConfigNumberInput
                        value={asNumber(asRecord(dataloaderConfig.shuffle).buffer_size, 1000)}
                        onCommit={(value) =>
                          handleDataloaderField(
                            ["shuffle", "buffer_size"],
                            value
                          )
                        }
                      />
                    </label>
                  </div>
                </div>
              </div>
            </details>

            <details className="settingsPanel" open ref={datasetPanelRef}>
              <summary>Core dataset settings</summary>
              <div className="settingsGrid">
                <div
                  id="settings-dataset"
                  ref={datasetSettingsRef}
                  className={`settingsGroup settingsCategoryAnchor ${
                    highlightedWorkflowTarget === "dataset"
                      ? "settingsCategoryAnchor-highlight"
                      : ""
                  }`}
                >
                  <div className="settingsGroupHeader">
                    <h3>Dataset sources</h3>
                    <p className="settingsGroupHint">
                      Match the tokenizer trainer: choose one source mode and configure the full
                      dataset stack here.
                    </p>
                  </div>

                  <div className="sourceModeRow trainingTokenizerDatasetSection">
                    <span>Dataset source</span>
                    <div className="modeSwitch">
                      <button
                        type="button"
                        className={`modeSwitchButton ${
                          datasetSourceMode === "local_file" ? "modeSwitchButton-active" : ""
                        }`}
                        onClick={() => setDatasetSourceMode("local_file")}
                      >
                        Local files
                      </button>
                      <button
                        type="button"
                        className={`modeSwitchButton ${
                          datasetSourceMode === "streaming_hf" ? "modeSwitchButton-active" : ""
                        }`}
                        onClick={() => {
                          setDatasetSourceMode("streaming_hf");
                          setStreamingDatasets((previous) =>
                            normalizeStreamingDatasetWeights(previous)
                          );
                        }}
                      >
                        Streaming Hugging Face datasets
                      </button>
                    </div>
                  </div>

                  {datasetSourceMode === "local_file" ? (
                    <div className="datasetConfigurator trainingTokenizerDatasetSection">
                      <div
                        className={`localFileManager ${
                          isDraggingTrainFiles ? "localFileManager-dragging" : ""
                        }`}
                        onDragEnter={handleLocalTrainFilesDragEnter}
                        onDragOver={handleLocalTrainFilesDragOver}
                        onDragLeave={handleLocalTrainFilesDragLeave}
                        onDrop={handleLocalTrainFilesDrop}
                      >
                        <div className="localFileManagerHeader">
                          <div>
                            <strong>Local training files</strong>
                            <p>Training and evaluation use this same file set.</p>
                          </div>
                          <div className="localFileHeaderActions">
                            <div className="localFileHeaderButtons">
                              <label
                                className={`secondaryButton localFileUploadButton localFileHeaderButton ${
                                  isUploadingTrainFile ? "localFileUploadButton-disabled" : ""
                                }`}
                                aria-disabled={isUploadingTrainFile}
                              >
                                {isUploadingTrainFile ? "Uploading..." : "Add files"}
                                <input
                                  type="file"
                                  multiple
                                  onChange={handleTrainFilesSelected}
                                  disabled={isUploadingTrainFile}
                                />
                              </label>
                              <button
                                type="button"
                                className="textButton localFileHeaderButton"
                                onClick={clearLocalTrainFiles}
                                disabled={localTrainFiles.length === 0}
                              >
                                Remove all
                              </button>
                            </div>
                            <span className="localFileCount">
                              {localTrainFiles.length} file
                              {localTrainFiles.length === 1 ? "" : "s"}
                            </span>
                          </div>
                        </div>

                        {localTrainFiles.length === 0 ? (
                          <p className="filterEmpty">
                            No local files added yet. Add one or more files to train.
                          </p>
                        ) : (
                          <ul className="localFileList">
                            {localTrainFiles.map((entry) => {
                              const fileCharLabel = formatCharCount(entry.sizeChars);
                              return (
                                <li key={entry.id} className="localFileItem">
                                  <strong className="localFileName" title={entry.fileName}>
                                    {entry.fileName}
                                  </strong>
                                  <div className="localFileActions">
                                    <span className="localFileStat">
                                      {fileCharLabel
                                        ? `${fileCharLabel} characters`
                                        : "Character count pending"}
                                    </span>
                                    <button
                                      type="button"
                                      className="textButton localFileRemoveIconButton"
                                      onClick={() => removeLocalTrainFile(entry.id)}
                                      aria-label={`Remove ${entry.fileName}`}
                                      title={`Remove ${entry.fileName}`}
                                    >
                                      <FiX aria-hidden="true" />
                                    </button>
                                  </div>
                                </li>
                              );
                            })}
                          </ul>
                        )}

                        <span className="fieldNote">Files are deduplicated by stored path.</span>
                      </div>
                    </div>
                  ) : (
                    <div className="datasetConfigurator trainingTokenizerDatasetSection">
                      <label className="fieldLabel fullWidthField">
                        <span>Hugging Face access token <small>optional</small></span>
                        <input
                          type="password"
                          value={hfToken}
                          onChange={(event) => setHfToken(event.target.value)}
                          autoComplete="off"
                          placeholder="hf_..."
                        />
                        <span className="fieldNote">Required for gated/private datasets.</span>
                      </label>

                      <div className="actionRow">
                        <button
                          type="button"
                          className="secondaryButton"
                          onClick={addStreamingDataset}
                        >
                          Add dataset
                        </button>
                        <button
                          type="button"
                          className="secondaryButton"
                          onClick={() => {
                            void handleLoadStreamingTemplate();
                          }}
                          disabled={isLoadingDatasetTemplate}
                        >
                          {isLoadingDatasetTemplate
                            ? "Loading template..."
                            : "Load streaming template"}
                        </button>
                      </div>

                      <div className="datasetList">
                        {streamingDatasets.map((entry, index) => (
                          <div key={entry.id} className="datasetCard">
                            <div className="datasetCardHeader">
                              <strong>Streaming dataset {index + 1}</strong>
                              <button
                                type="button"
                                className="textButton datasetRemoveButton"
                                onClick={() => removeStreamingDataset(entry.id)}
                                disabled={streamingDatasets.length <= 1}
                                aria-label={`Remove streaming dataset ${index + 1}`}
                                title={`Remove streaming dataset ${index + 1}`}
                              >
                                <FiTrash2 aria-hidden="true" />
                              </button>
                            </div>

                            <div className="fieldGrid">
                              <label className="fieldLabel">
                                <span>Hugging Face dataset name</span>
                                <input
                                  value={entry.name}
                                  onChange={(event) =>
                                    updateStreamingDataset(entry.id, {
                                      name: event.target.value,
                                    })
                                  }
                                  placeholder="HuggingFaceFW/fineweb-edu"
                                />
                              </label>

                              <label className="fieldLabel">
                                <span>Split</span>
                                <input
                                  value={entry.split}
                                  onChange={(event) =>
                                    updateStreamingDataset(entry.id, {
                                      split: event.target.value,
                                    })
                                  }
                                  placeholder="train"
                                />
                              </label>

                              <label className="fieldLabel">
                                <span>Weight</span>
                                <input
                                  inputMode="decimal"
                                  pattern="[0-9]*[.]?[0-9]*"
                                  min="0"
                                  max="1"
                                  step="0.000001"
                                  value={entry.weight}
                                  onChange={(event) =>
                                    updateStreamingWeight(entry.id, event.target.value)
                                  }
                                  placeholder="1.0"
                                />
                              </label>

                              <label className="fieldLabel fullWidthField">
                                <span>Text columns</span>
                                <input
                                  value={entry.textColumns}
                                  onChange={(event) =>
                                    updateStreamingDataset(entry.id, {
                                      textColumns: event.target.value,
                                    })
                                  }
                                  placeholder="text"
                                />
                              </label>
                            </div>

                            <details className="subPanel">
                              <summary>Advanced source options</summary>
                              <div className="fieldGrid">
                                <label className="fieldLabel">
                                  <span>Dataset config <small>optional</small></span>
                                  <input
                                    value={entry.config}
                                    onChange={(event) =>
                                      updateStreamingDataset(entry.id, {
                                        config: event.target.value,
                                      })
                                    }
                                  />
                                </label>

                                <div className="fullWidthField filterBuilder">
                                  <div className="filterBuilderHeader">
                                    <span className="filterBuilderTitle">Filters (optional)</span>
                                    <button
                                      type="button"
                                      className="secondaryButton"
                                      onClick={() => addStreamingFilter(entry.id)}
                                    >
                                      Add filter
                                    </button>
                                  </div>

                                  {entry.filters.length === 0 ? (
                                    <p className="filterEmpty">No filters yet.</p>
                                  ) : (
                                    <div className="filterList">
                                      {entry.filters.map((filter) => (
                                        <div key={filter.id} className="filterRow">
                                          <label className="fieldLabel">
                                            <span>Column</span>
                                            <input
                                              value={filter.column}
                                              onChange={(event) =>
                                                updateStreamingFilter(entry.id, filter.id, {
                                                  column: event.target.value,
                                                })
                                              }
                                              placeholder="language_score"
                                            />
                                          </label>

                                          <label className="fieldLabel">
                                            <span>Operator</span>
                                            <select
                                              value={filter.operator}
                                              onChange={(event) =>
                                                updateStreamingFilter(entry.id, filter.id, {
                                                  operator: event.target
                                                    .value as FilterOperator,
                                                })
                                              }
                                            >
                                              {FILTER_OPERATORS.map((operator) => (
                                                <option key={operator} value={operator}>
                                                  {operator}
                                                </option>
                                              ))}
                                            </select>
                                          </label>

                                          <label className="fieldLabel">
                                            <span>Value</span>
                                            <input
                                              value={filter.value}
                                              onChange={(event) =>
                                                updateStreamingFilter(entry.id, filter.id, {
                                                  value: event.target.value,
                                                })
                                              }
                                              placeholder={
                                                filter.operator === "in" ||
                                                filter.operator === "not in"
                                                  ? '["en", "de"] or en,de'
                                                  : 'en, true, 0.95, {"k":1}'
                                              }
                                            />
                                          </label>

                                          <button
                                            type="button"
                                            className="textButton filterRemoveButton"
                                            onClick={() =>
                                              removeStreamingFilter(entry.id, filter.id)
                                            }
                                          >
                                            Remove
                                          </button>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                  <p className="fieldNote">
                                    Values are inferred automatically. For `in`/`not in`, use a
                                    JSON array or comma-separated values.
                                  </p>
                                </div>
                              </div>
                            </details>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </details>

            <details className="settingsPanel" open>
              <summary>Sampling prompts</summary>
              <div className="settingsGrid">
                <div className="settingsGroup">
                  <div className="trainingSettingsPanelHead">
                    <div className="settingsGroupHeader">
                      <h3>Prompt presets</h3>
                      <p className="settingsGroupHint">
                        These prompts power the live sample viewer during the run.
                      </p>
                    </div>
                    <div className="trainingPromptToolbar">
                      <span className="pillBadge tone-neutral">
                        {promptEntries.length} preset{promptEntries.length === 1 ? "" : "s"}
                      </span>
                      <button
                        type="button"
                        className="buttonGhost buttonSmall"
                        onClick={handleAddPrompt}
                      >
                        <FiPlus /> Add prompt
                      </button>
                    </div>
                  </div>
                  <p className="trainingPromptHintLine">
                    Keep a small mix of short, medium, and edge-case prompts so the live samples
                    reveal regressions quickly.
                  </p>
                  <div className="trainingPromptGrid">
                    {promptEntries.map((prompt, index) => (
                      <article key={`prompt-${index}`} className="trainingPromptCard">
                        <div className="trainingPromptCardHead">
                          <div className="trainingPromptTitleGroup">
                            <div className="trainingPromptTitle">Prompt {index + 1}</div>
                            <p className="trainingPromptMeta">
                              {Math.max(0, asString(prompt.prompt).trim().length)} characters
                            </p>
                          </div>
                          <button
                            type="button"
                            className="textButton trainingPromptRemoveButton"
                            onClick={() => handleRemovePrompt(index)}
                            aria-label={`Remove prompt ${index + 1}`}
                            title={`Remove prompt ${index + 1}`}
                          >
                            <FiTrash2 aria-hidden="true" />
                          </button>
                        </div>

                        <label className="fieldLabel trainingPromptEditor">
                          <span>Prompt text</span>
                          <textarea
                            rows={4}
                            value={asString(prompt.prompt)}
                            onChange={(event) =>
                              handlePromptChange(index, "prompt", event.target.value)
                            }
                            placeholder="Write a short evaluation prompt..."
                          />
                        </label>

                        <div className="trainingPromptFields">
                          <label className="fieldLabel">
                            <span>Maximum generated tokens</span>
                            <ConfigNumberInput
                              value={asNumber(prompt.max_tokens, 64)}
                              onCommit={(value) =>
                                handlePromptChange(
                                  index,
                                  "max_tokens",
                                  value
                                )
                              }
                            />
                          </label>
                          <label className="fieldLabel">
                            <span>Temperature</span>
                            <ConfigNumberInput
                              mode="decimal"
                              step="0.05"
                              value={asNumber(prompt.temperature, 0.7)}
                              onCommit={(value) =>
                                handlePromptChange(
                                  index,
                                  "temperature",
                                  value
                                )
                              }
                            />
                          </label>
                          <label className="fieldLabel">
                            <span>Top-k sampling</span>
                            <ConfigNumberInput
                              value={asNumber(prompt.top_k, 40)}
                              onCommit={(value) => handlePromptChange(index, "top_k", value)}
                            />
                          </label>
                        </div>
                      </article>
                    ))}
                  </div>
                </div>
              </div>
            </details>

            <details className="settingsPanel">
              <summary>Advanced runtime controls</summary>
              <div className="settingsGrid">
                <div className="settingsGroup">
                  <div className="settingsGroupHeader">
                    <h3>Deeper runtime options</h3>
                    <p className="settingsGroupHint">
                      Token formatting, optimizer internals, and multi-node controls stay available
                      without taking over the main workflow.
                    </p>
                  </div>
                  <div className="fieldGrid trainingSettingsCompactGrid">
                    <label className="fieldLabel">
                      <span>Beginning-of-sequence token</span>
                      <input
                        value={asString(dataloaderConfig.bos_token)}
                        onChange={(event) =>
                          handleDataloaderField(["bos_token"], event.target.value)
                        }
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>End-of-sequence token</span>
                      <input
                        value={asString(dataloaderConfig.eos_token)}
                        onChange={(event) =>
                          handleDataloaderField(["eos_token"], event.target.value)
                        }
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Padding token</span>
                      <input
                        value={asString(dataloaderConfig.pad_token)}
                        onChange={(event) =>
                          handleDataloaderField(["pad_token"], event.target.value)
                        }
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Token data type</span>
                      <select
                        value={asString(dataloaderConfig.token_dtype, "int64")}
                        onChange={(event) =>
                          handleDataloaderField(["token_dtype"], event.target.value)
                        }
                      >
                        <option value="int64">int64</option>
                        <option value="int32">int32</option>
                        <option value="int16">int16</option>
                        <option value="uint8">uint8</option>
                      </select>
                    </label>
                    <label className="fieldLabel">
                      <span>Pretokenize batch size</span>
                      <ConfigNumberInput
                        value={asNumber(dataloaderConfig.pretokenize_batch_size, 1000)}
                        onCommit={(value) =>
                          handleDataloaderField(
                            ["pretokenize_batch_size"],
                            value
                          )
                        }
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Cache directory</span>
                      <input
                        value={asString(dataloaderConfig.cache_dir)}
                        onChange={(event) =>
                          handleDataloaderField(["cache_dir"], event.target.value)
                        }
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Optimizer betas</span>
                      <input
                        value={
                          Array.isArray(asRecord(trainingConfig.optimizer).betas)
                            ? (asRecord(trainingConfig.optimizer).betas as unknown[])
                                .map(String)
                                .join(", ")
                            : "0.9, 0.95"
                        }
                        onChange={(event) =>
                          handleTrainingField(
                            ["optimizer", "betas"],
                            event.target.value
                              .split(",")
                              .map((item) => Number(item.trim()))
                              .filter((value) => Number.isFinite(value))
                          )
                        }
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Optimizer epsilon</span>
                      <ConfigNumberInput
                        mode="decimal"
                        step="0.00000001"
                        value={asNumber(asRecord(trainingConfig.optimizer).eps, 1e-8)}
                        onCommit={(value) => handleTrainingField(["optimizer", "eps"], value)}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Distributed node split</span>
                      <select
                        value={String(Boolean(dataloaderConfig.node_split))}
                        onChange={(event) =>
                          handleDataloaderField(
                            ["node_split"],
                            event.target.value === "true"
                          )
                        }
                      >
                        <option value="false">Disabled</option>
                        <option value="true">Enabled</option>
                      </select>
                    </label>
                    <label className="fieldLabel">
                      <span>Distributed node rank</span>
                      <ConfigNumberInput
                        value={asNumber(dataloaderConfig.node_rank, 0)}
                        onCommit={(value) => handleDataloaderField(["node_rank"], value)}
                      />
                    </label>
                    <label className="fieldLabel">
                      <span>Distributed node world size</span>
                      <ConfigNumberInput
                        value={asNumber(dataloaderConfig.node_world_size, 1)}
                        onCommit={(value) => handleDataloaderField(["node_world_size"], value)}
                      />
                    </label>
                  </div>
                </div>
              </div>
            </details>

            <details className="settingsPanel">
              <summary>Generated configuration JSON</summary>
              <div className="settingsGrid">
                <div className="trainingJsonGrid">
                  <pre className="trainingCodeBlock">{prettyJson(trainingConfig)}</pre>
                  <pre className="trainingCodeBlock">{prettyJson(dataloaderConfig)}</pre>
                </div>
              </div>
            </details>
          </div>
        ) : (
          <div className="trainingEmpty">Loading starter templates…</div>
        )}
      </section>

      {pickerKind ? (
        <div
          className="trainingAssetPickerOverlay"
          onClick={closePicker}
          role="presentation"
        >
          <section
            id="training-asset-picker"
            className="panelCard trainingAssetPickerDialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="training-asset-picker-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="trainingAssetPickerHeader">
              <div>
                <h2 id="training-asset-picker-title">
                  {pickerKind === "project" ? "Choose model config" : "Choose tokenizer artifact"}
                </h2>
                <p className="panelCopy">
                  {pickerKind === "project"
                    ? "Select a saved model project from the workspace to pair with this run."
                    : "Select a completed tokenizer artifact. Only completed tokenizer jobs are shown here."}
                </p>
              </div>
              <button
                type="button"
                className="buttonGhost iconOnly"
                onClick={closePicker}
                aria-label="Close asset picker"
              >
                <FiXCircle />
              </button>
            </div>

            <div className="trainingAssetPickerControls">
              <label className="trainingAssetPickerSearch">
                <FiSearch />
                <input
                  value={pickerQuery}
                  onChange={(event) => setPickerQuery(event.target.value)}
                  placeholder={
                    pickerKind === "project"
                      ? "Search model configurations by name, identifier, or file"
                      : "Search tokenizers by name, identifier, or artifact"
                  }
                />
              </label>
              <button
                type="button"
                className="buttonGhost"
                onClick={() => {
                  void openPicker(pickerKind);
                }}
              >
                <FiRefreshCw /> Refresh
              </button>
            </div>

            <div className="trainingAssetPickerResults">
              {pickerLoading ? <div className="trainingEmpty">Loading workspace assets…</div> : null}

              {!pickerLoading && pickerError ? (
                <div className="inlineNotice tone-info">{pickerError}</div>
              ) : null}

              {!pickerLoading && !pickerError && pickerKind === "project" && visiblePickerProjects.length === 0 ? (
                <div className="trainingAssetPickerEmpty">
                  <h3>No saved model configurations found.</h3>
                  <p className="panelCopy">
                    Create or save a model config from the Home workspace, then reopen the picker.
                  </p>
                  <Link className="buttonGhost" href="/">
                    <FiLayers /> Open Workspace Assets
                  </Link>
                </div>
              ) : null}

              {!pickerLoading && !pickerError && pickerKind === "tokenizer" && visiblePickerTokenizerJobs.length === 0 ? (
                <div className="trainingAssetPickerEmpty">
                  <h3>No completed tokenizer artifacts found.</h3>
                  <p className="panelCopy">
                    Finish a tokenizer training job first, then reopen the picker to pair it with this run.
                  </p>
                  <Link className="buttonGhost" href="/tokenizer">
                    <FiCpu /> Open Tokenizer Studio
                  </Link>
                </div>
              ) : null}

              {!pickerLoading && !pickerError && pickerKind === "project"
                ? visiblePickerProjects.map((project) => (
                    <button
                      key={project.id}
                      type="button"
                      className={`trainingAssetPickerOption ${
                        selectedProjectId === project.id ? "is-selected" : ""
                      }`}
                      onClick={() => {
                        setSelectedProjectId(project.id);
                        closePicker();
                      }}
                      >
                      <div className="trainingAssetPickerOptionHead">
                        <div>
                          <div className="trainingAssetName">
                            {project.name ?? project.id}
                          </div>
                        </div>
                        <span
                          className={`pillBadge ${
                            selectedProjectId === project.id ? "tone-good" : "tone-neutral"
                          }`}
                        >
                          {selectedProjectId === project.id ? "Selected" : "Use model"}
                        </span>
                      </div>
                      <div className="trainingAssetPickerOptionMeta">
                        {formatDate(project.created_at)} • {formatBytes(project.size_bytes)}
                      </div>
                      <div className="trainingAssetPickerOptionMeta">{project.artifact_file}</div>
                    </button>
                  ))
                : null}

              {!pickerLoading && !pickerError && pickerKind === "tokenizer"
                ? visiblePickerTokenizerJobs.map((job) => (
                    <button
                      key={job.id}
                      type="button"
                      className={`trainingAssetPickerOption ${
                        selectedTokenizerJobId === job.id ? "is-selected" : ""
                      }`}
                      onClick={() => {
                        setSelectedTokenizerJobId(job.id);
                        closePicker();
                      }}
                      >
                      <div className="trainingAssetPickerOptionHead">
                        <div>
                          <div className="trainingAssetName">
                            {asString(job.tokenizer_config.name, job.id)}
                          </div>
                        </div>
                        <span
                          className={`pillBadge ${
                            selectedTokenizerJobId === job.id ? "tone-good" : "tone-neutral"
                          }`}
                        >
                          {selectedTokenizerJobId === job.id ? "Selected" : "Use tokenizer"}
                        </span>
                      </div>
                      <div className="trainingAssetPickerOptionMeta">
                        {formatDate(job.created_at)}
                        {job.stats?.vocab_size ? ` • vocabulary size ${formatInteger(job.stats.vocab_size)}` : ""}
                      </div>
                      <div className="trainingAssetPickerOptionMeta">
                        {job.artifact_file ?? job.artifact_path ?? "Tokenizer artifact path unavailable"}
                      </div>
                    </button>
                  ))
                : null}
            </div>
          </section>
        </div>
      ) : null}

      <div className="trainingToastStack" aria-live="polite">
        {toasts.map((toast) => (
          <div key={toast.id} className={`trainingToast tone-${toast.level === "info" ? "success" : toast.level}`}>
            <div className="trainingToastTitle">{toast.title}</div>
            <div className="trainingToastBody">{toast.body}</div>
          </div>
        ))}
      </div>
    </main>
  );
}

export default function TrainingPage() {
  return (
    <Suspense fallback={null}>
      <TrainingPageContent />
    </Suspense>
  );
}
