"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  Suspense,
  startTransition,
  type ChangeEvent,
  type DragEvent,
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
const HIDDEN_RUNS_STORAGE_KEY = "llm-training-hidden-runs-v1";
const POLL_INTERVAL_MS = 1800;

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
              weight: sanitizeWeightInput(asString(datasetRecord.weight, "1")),
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

function metricSeries(metrics: TrainingMetricPoint[], key: keyof TrainingMetricPoint): number[] {
  return metrics
    .map((item) => item[key])
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
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

function buildPointList(values: number[]): string {
  if (values.length === 0) {
    return "";
  }
  const width = 280;
  const height = 112;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  return values
    .map((value, index) => {
      const x = values.length === 1 ? width / 2 : (index / (values.length - 1)) * width;
      const normalized = (value - min) / range;
      const y = height - normalized * (height - 12) - 6;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function Sparkline({
  title,
  values,
  latestValue,
  stroke,
}: {
  title: string;
  values: number[];
  latestValue: string;
  stroke: string;
}) {
  const points = buildPointList(values);

  return (
    <div className="trainingChartCard">
      <div className="trainingChartHead">
        <strong>{title}</strong>
        <span>{latestValue}</span>
      </div>
      {points ? (
        <svg className="trainingChartSvg" viewBox="0 0 280 112" preserveAspectRatio="none">
          <polyline
            fill="none"
            stroke={stroke}
            strokeWidth="3"
            strokeLinejoin="round"
            strokeLinecap="round"
            points={points}
          />
        </svg>
      ) : (
        <div className="trainingEmpty">Metrics will appear after the first logged steps.</div>
      )}
    </div>
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
  const [hiddenRunIds, setHiddenRunIds] = useState<string[]>([]);
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
  const initializedRef = useRef(false);
  const pickerRequestIdRef = useRef(0);
  const datasetUiHydratedRef = useRef(false);
  const localFileDragDepthRef = useRef(0);
  const localTrainFileStatsPendingIdsRef = useRef(new Set<string>());
  const localTrainFileStatsFailedIdsRef = useRef(new Set<string>());

  const deferredTrainingConfig = useDeferredValue(trainingConfig);
  const deferredDataloaderConfig = useDeferredValue(dataloaderConfig);

  const notify = useCallback((level: ToastLevel, title: string, body: string) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setToasts((current) => [...current, { id, level, title, body }]);
    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id));
    }, 3600);
  }, []);

  const refreshRecentRuns = useCallback(async () => {
    try {
      const jobs = await fetchTrainingJobs();
      startTransition(() => {
        setRecentRuns(jobs);
        if (!activeRunId) {
          const nextVisible = jobs.find((job) => !hiddenRunIds.includes(job.id)) ?? jobs[0] ?? null;
          if (nextVisible) {
            setActiveRunId(nextVisible.id);
          }
        }
      });
    } catch (error) {
      notify("error", "Recent runs unavailable", error instanceof Error ? error.message : "Failed to load training jobs.");
    }
  }, [activeRunId, hiddenRunIds, notify]);

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
    const storedHiddenRuns = readStoredJson<string[]>(HIDDEN_RUNS_STORAGE_KEY, []);

    setHiddenRunIds(storedHiddenRuns);
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
            const nextVisible = jobs.find((job) => !storedHiddenRuns.includes(job.id)) ?? jobs[0] ?? null;
            if (nextVisible) {
              setActiveRunId(nextVisible.id);
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
    writeStoredJson(HIDDEN_RUNS_STORAGE_KEY, hiddenRunIds);
  }, [hiddenRunIds]);

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
          setRecentRuns((current) => {
            const next = [job, ...current.filter((item) => item.id !== job.id)];
            return next.slice(0, 12);
          });
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

  const visibleRecentRuns = useMemo(
    () => recentRuns.filter((job) => !hiddenRunIds.includes(job.id)),
    [hiddenRunIds, recentRuns]
  );

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
      const hydrated = hydrateDatasetUiFromConfig(
        asRecord(templates.dataloader_config_template)
      );
      setDatasetSourceMode("streaming_hf");
      setHfToken(hydrated.hfToken);
      setStreamingDatasets(
        normalizeStreamingDatasetWeights(hydrated.streamingDatasets)
      );
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
        setRecentRuns((current) => [job, ...current.filter((item) => item.id !== job.id)]);
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

  const lossValues = metricSeries(metrics, "loss");
  const lrValues = metricSeries(metrics, "lr");
  const normValues = metricSeries(metrics, "norm");
  const throughputValues = metricSeries(metrics, "tok_per_sec");

  const startReady = Boolean(preflight?.valid && selectedProjectId && selectedTokenizerJobId && !launching);
  const workflowSteps = [
    {
      title: "Choose saved model config",
      ready: Boolean(selectedProject),
      body: selectedProject ? selectedProject.name ?? selectedProject.id : "Pick a saved model config from the home workspace or query parameters.",
    },
    {
      title: "Choose completed tokenizer artifact",
      ready: Boolean(selectedTokenizer && selectedTokenizer.status === "completed"),
      body:
        selectedTokenizer && selectedTokenizer.status === "completed"
          ? asString(selectedTokenizer.tokenizer_config.name, selectedTokenizer.id)
          : "Select a completed tokenizer artifact to ensure vocab compatibility.",
    },
    {
      title: "Configure dataset + run settings",
      ready: Boolean(trainingConfig && dataloaderConfig),
      body: "Tune sequence length, effective batch size, save cadence, prompts, and dataset sources.",
    },
    {
      title: "Validate compatibility + memory",
      ready: Boolean(preflight?.valid),
      body: preflightError ?? preflight?.errors[0]?.message ?? "Automatic preflight checks run after every settings change.",
    },
    {
      title: "Start training",
      ready: startReady,
      body: startReady ? "The run is clear to launch." : "The start action unlocks only after a passing preflight.",
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
            <div className="trainingAssetCard">
              <span className="trainingAssetLabel">Model Config</span>
              <span className="trainingAssetName">
                {selectedProject?.name ?? (selectedProjectId ? selectedProjectId : "No model selected")}
              </span>
              <span className="trainingAssetMeta">
                {selectedProject
                  ? `${formatDate(selectedProject.created_at)} • vocab ${selectedProject.model_config.vocab_size}`
                  : "Choose a saved model config from Home or pass ?project=..."}
              </span>
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
            <div className="trainingAssetCard">
              <span className="trainingAssetLabel">Tokenizer Artifact</span>
              <span className="trainingAssetName">
                {selectedTokenizer
                  ? asString(selectedTokenizer.tokenizer_config.name, selectedTokenizer.id)
                  : selectedTokenizerJobId ?? "No tokenizer selected"}
              </span>
              <span className="trainingAssetMeta">
                {selectedTokenizer
                  ? `${formatDate(selectedTokenizer.created_at)} • ${selectedTokenizer.status.toUpperCase()}`
                  : "Choose a completed tokenizer artifact from Home or pass ?tokenizerJob=..."}
              </span>
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
              <span>Run Name</span>
              <input value={runName} onChange={(event) => setRunName(event.target.value)} placeholder="optional training run name" />
            </label>
          </div>
        </div>
      </section>

      <section className="panelCard">
        <div className="panelHead">
          <div>
            <h2>Workflow</h2>
            <p className="panelCopy">The page keeps the start path explicit: asset pair, runtime config, preflight, then launch.</p>
          </div>
        </div>
        <div className="trainingWorkflowGrid">
          {workflowSteps.map((step, index) => (
            <article
              key={step.title}
              className={`trainingWorkflowCard ${step.ready ? "tone-good" : "tone-warn"}`}
            >
              <div className="trainingWorkflowTop">
                <span className="trainingWorkflowStep">Step {index + 1}</span>
                <span className={`pillBadge ${step.ready ? "tone-good" : "tone-warn"}`}>
                  {step.ready ? "Ready" : "Waiting"}
                </span>
              </div>
              <div className="trainingWorkflowTitle">{step.title}</div>
              <div className="trainingWorkflowMeta">{step.body}</div>
            </article>
          ))}
        </div>
      </section>

      <section className="trainingResultsGrid">
        <div className="trainingPanelStack">
          <section className="panelCard">
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
                      <div className="statusCardTitle">Tokenizer Vocab</div>
                      <div className="statusCardValue">{formatInteger(preflight.compatibility.tokenizer_vocab_size)}</div>
                      <div className="statusCardDetail">Model expects {formatInteger(preflight.compatibility.model_vocab_size)}</div>
                    </div>
                  </div>
                  <div className="statusCard">
                    <div className="statusCardIcon">
                      <FiLayers />
                    </div>
                    <div>
                      <div className="statusCardTitle">Sequence Length</div>
                      <div className="statusCardValue">{formatInteger(preflight.compatibility.seq_len)}</div>
                      <div className="statusCardDetail">Context max {formatInteger(preflight.compatibility.model_context_length)}</div>
                    </div>
                  </div>
                  <div className="statusCard">
                    <div className="statusCardIcon">
                      <FiBarChart2 />
                    </div>
                    <div>
                      <div className="statusCardTitle">Micro Batch</div>
                      <div className="statusCardValue">
                        {formatInteger(preflight.derived_runtime?.micro_batch_size ?? null)}
                      </div>
                      <div className="statusCardDetail">
                        Grad accum {formatInteger(preflight.derived_runtime?.grad_accum_steps ?? null)}
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
                        Memory-estimated max batch {formatInteger((preflight.memory_estimate?.max_batch_size as number | undefined) ?? null)}
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
                  <span className={`pillBadge ${statusTone(activeRun.status)}`}>{activeRun.status.toUpperCase()}</span>
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
                      <span>Run ID {activeRun.id.slice(0, 8)}</span>
                      <span>Created {formatDate(activeRun.created_at)}</span>
                      <span>Started {activeRun.started_at ? formatDate(activeRun.started_at) : "waiting"}</span>
                      <span>ETA {formatDuration((activeRun as unknown as { eta_seconds?: number | null }).eta_seconds ?? null)}</span>
                    </div>
                  </div>

                  <div className="statusGrid">
                    <div className="statusCard">
                      <div className="statusCardIcon"><FiActivity /></div>
                      <div>
                        <div className="statusCardTitle">Step</div>
                        <div className="statusCardValue">
                          {formatInteger(activeRun.last_step)} / {formatInteger(activeRun.max_steps)}
                        </div>
                        <div className="statusCardDetail">Progress {Math.round(activeRun.progress * 100)}%</div>
                      </div>
                    </div>
                    <div className="statusCard">
                      <div className="statusCardIcon"><FiCheckCircle /></div>
                      <div>
                        <div className="statusCardTitle">Loss</div>
                        <div className="statusCardValue">{formatMetric(activeRun.latest_loss, 4)}</div>
                        <div className="statusCardDetail">Grad norm {formatMetric(activeRun.latest_grad_norm, 3)}</div>
                      </div>
                    </div>
                    <div className="statusCard">
                      <div className="statusCardIcon"><FiRefreshCw /></div>
                      <div>
                        <div className="statusCardTitle">Learning Rate</div>
                        <div className="statusCardValue">{formatMetric(activeRun.latest_lr, 6)}</div>
                        <div className="statusCardDetail">Tokens/sec {formatInteger(activeRun.latest_tokens_per_sec)}</div>
                      </div>
                    </div>
                    <div className="statusCard">
                      <div className="statusCardIcon"><FiArchive /></div>
                      <div>
                        <div className="statusCardTitle">Artifacts</div>
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
                    <Sparkline title="Loss" values={lossValues} latestValue={formatMetric(activeRun.latest_loss, 4)} stroke="var(--brand)" />
                    <Sparkline title="Learning Rate" values={lrValues} latestValue={formatMetric(activeRun.latest_lr, 6)} stroke="var(--ok)" />
                    <Sparkline title="Grad Norm" values={normValues} latestValue={formatMetric(activeRun.latest_grad_norm, 3)} stroke="var(--warn)" />
                    <Sparkline title="Throughput" values={throughputValues} latestValue={formatInteger(activeRun.latest_tokens_per_sec)} stroke="var(--danger)" />
                  </div>

                  <details className="sectionDisclosure" open>
                    <summary className="sectionDisclosureSummary">Samples</summary>
                    <div className="trainingSampleList">
                      {samples.length ? (
                        samples.slice().reverse().map((entry) => (
                          <div key={`sample-${entry.step}`} className="trainingSampleCard">
                            <div className="trainingSampleTitle">Step {entry.step}</div>
                            {entry.samples.map((sample) => (
                              <div key={`${entry.step}-${sample.index}`} className="trainingCodeBlock">
                                {sample.prompt ? `Prompt ${sample.index + 1}: ${sample.prompt}\n\n` : ""}
                                {sample.text}
                              </div>
                            ))}
                          </div>
                        ))
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
                    <summary className="sectionDisclosureSummary">Resolved runtime and configs</summary>
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
                {visibleRecentRuns.length ? (
                  visibleRecentRuns.map((job) => (
                    <div key={job.id} className={`trainingRecentCard ${activeRunId === job.id ? "is-active" : ""}`}>
                      <div className="trainingSectionHeader">
                        <div>
                          <div className="trainingRecentTitle">{job.name}</div>
                          <div className="trainingRecentMeta">
                            {job.project_name} • {job.tokenizer_name}
                          </div>
                        </div>
                        <span className={`pillBadge ${statusTone(job.status)}`}>{job.status}</span>
                      </div>
                      <div className="trainingRecentMeta">
                        {formatDate(job.created_at)} • Step {formatInteger(job.last_step)} / {formatInteger(job.max_steps)}
                      </div>
                      <div className="trainingRecentActions">
                        <button type="button" className="buttonGhost buttonSmall" onClick={() => setActiveRunId(job.id)}>
                          Open
                        </button>
                        <button
                          type="button"
                          className="buttonGhost buttonSmall"
                          onClick={() => setHiddenRunIds((current) => [...current, job.id])}
                        >
                          Hide
                        </button>
                        <button
                          type="button"
                          className="buttonDanger buttonSmall"
                          onClick={() => void handleDeleteRun(job.id)}
                          disabled={job.status === "running" || job.status === "pending"}
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="trainingEmpty">
                    No visible training runs yet.
                    {hiddenRunIds.length ? (
                      <button
                        type="button"
                        className="buttonGhost buttonSmall trainingEmptyAction"
                        onClick={() => setHiddenRunIds([])}
                      >
                        Restore hidden runs
                      </button>
                    ) : null}
                  </div>
                )}
              </div>
          </section>
        </div>
      </section>

      <section className="panelCard trainingSettingsStudio">
        <div className="panelHead">
          <div>
            <h2>Training Settings Studio</h2>
            <p className="panelCopy">High-value controls stay front and center; advanced runtime knobs live behind disclosures instead of blocking the first screen.</p>
          </div>
          <div className="heroMetaPills">
            <span className="pillBadge tone-neutral">{datasetEntries.length} dataset source{datasetEntries.length === 1 ? "" : "s"}</span>
            <span className="pillBadge tone-neutral">{promptEntries.length} prompt{promptEntries.length === 1 ? "" : "s"}</span>
          </div>
        </div>

        {trainingConfig && dataloaderConfig ? (
          <div className="trainingSettingsStack">
            <section className="trainingSettingsSection">
              <div className="trainingSettingsSectionHead">
                <div>
                  <h3>Core launch knobs</h3>
                  <p className="trainingSettingsSectionCopy">
                    Tune the values you are most likely to touch between runs before opening the deeper runtime controls.
                  </p>
                </div>
              </div>
              <div className="fieldGrid">
                <label className="fieldLabel">
                  <span>Sequence length</span>
                  <input
                    type="number"
                    value={asNumber(trainingConfig.seq_len, 128)}
                    onChange={(event) => handleTrainingField(["seq_len"], Number(event.target.value))}
                  />
                </label>
                <label className="fieldLabel">
                  <span>Max steps</span>
                  <input
                    type="number"
                    value={asNumber(trainingConfig.max_steps, 0)}
                    onChange={(event) => handleTrainingField(["max_steps"], Number(event.target.value))}
                  />
                </label>
                <label className="fieldLabel">
                  <span>Total batch size</span>
                  <input
                    type="number"
                    value={asNumber(trainingConfig.total_batch_size, 0)}
                    onChange={(event) => handleTrainingField(["total_batch_size"], Number(event.target.value))}
                  />
                </label>
                <label className="fieldLabel">
                  <span>Learning rate</span>
                  <input
                    type="number"
                    step="0.000001"
                    value={asNumber(asRecord(trainingConfig.optimizer).lr, 0.0003)}
                    onChange={(event) => handleTrainingField(["optimizer", "lr"], Number(event.target.value))}
                  />
                </label>
                <label className="fieldLabel">
                  <span>Weight decay</span>
                  <input
                    type="number"
                    step="0.0001"
                    value={asNumber(asRecord(trainingConfig.optimizer).weight_decay, 0.1)}
                    onChange={(event) => handleTrainingField(["optimizer", "weight_decay"], Number(event.target.value))}
                  />
                </label>
                <label className="fieldLabel">
                  <span>Save every</span>
                  <input
                    type="number"
                    value={asNumber(trainingConfig.save_every, 0)}
                    onChange={(event) => handleTrainingField(["save_every"], Number(event.target.value))}
                  />
                </label>
                <label className="fieldLabel">
                  <span>Sample every</span>
                  <input
                    type="number"
                    value={asNumber(trainingConfig.sample_every, 0)}
                    onChange={(event) => handleTrainingField(["sample_every"], Number(event.target.value))}
                  />
                </label>
                <label className="fieldLabel">
                  <span>Dataset shuffle buffer</span>
                  <input
                    type="number"
                    value={asNumber(asRecord(dataloaderConfig.shuffle).buffer_size, 1000)}
                    onChange={(event) => handleDataloaderField(["shuffle", "buffer_size"], Number(event.target.value))}
                  />
                </label>
              </div>
            </section>

            <section className="trainingSettingsSection">
              <div className="trainingSettingsSectionHead">
                <div>
                  <h3>Dataset Sources</h3>
                  <p className="trainingSettingsSectionCopy">
                    Match the tokenizer trainer exactly: choose one source mode and configure the full dataset stack here.
                  </p>
                </div>
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
                    Streaming HF datasets
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
                          {localTrainFiles.length} file{localTrainFiles.length === 1 ? "" : "s"}
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
                                  {fileCharLabel ? `${fileCharLabel} chars` : "char count pending"}
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
                    <span>HF access token (optional)</span>
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
                    <button type="button" className="secondaryButton" onClick={addStreamingDataset}>
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
                      {isLoadingDatasetTemplate ? "Loading template..." : "Load streaming template"}
                    </button>
                  </div>

                  <div className="datasetList">
                    {streamingDatasets.map((entry, index) => (
                      <div key={entry.id} className="datasetCard">
                        <div className="datasetCardHeader">
                          <strong>Streaming dataset {index + 1}</strong>
                          <button
                            type="button"
                            className="textButton"
                            onClick={() => removeStreamingDataset(entry.id)}
                            disabled={streamingDatasets.length <= 1}
                          >
                            Remove
                          </button>
                        </div>

                        <div className="fieldGrid">
                          <label className="fieldLabel">
                            <span>HF dataset name</span>
                            <input
                              value={entry.name}
                              onChange={(event) =>
                                updateStreamingDataset(entry.id, { name: event.target.value })
                              }
                              placeholder="HuggingFaceFW/fineweb-edu"
                            />
                          </label>

                          <label className="fieldLabel">
                            <span>Split</span>
                            <input
                              value={entry.split}
                              onChange={(event) =>
                                updateStreamingDataset(entry.id, { split: event.target.value })
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
                              <span>Config (optional)</span>
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
                                              operator: event.target.value as FilterOperator,
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
                                        onClick={() => removeStreamingFilter(entry.id, filter.id)}
                                      >
                                        Remove
                                      </button>
                                    </div>
                                  ))}
                                </div>
                              )}
                              <p className="fieldNote">
                                Values are inferred automatically. For `in`/`not in`, use a JSON
                                array or comma-separated values.
                              </p>
                            </div>
                          </div>
                        </details>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>

            <section className="trainingSettingsSection">
              <div className="trainingSettingsSectionHead">
                <div>
                  <h3>Sampling Prompts</h3>
                  <p className="trainingSettingsSectionCopy">These prompts power the live sample viewer during the run.</p>
                </div>
                <button type="button" className="buttonGhost buttonSmall" onClick={handleAddPrompt}>
                  <FiPlus /> Add prompt
                </button>
              </div>
              <div className="trainingPromptList">
                {promptEntries.map((prompt, index) => (
                  <div key={`prompt-${index}`} className="trainingPromptCard">
                    <div className="trainingSectionHeader">
                      <div className="trainingPromptTitle">Prompt {index + 1}</div>
                      <button type="button" className="buttonDanger buttonSmall" onClick={() => handleRemovePrompt(index)}>
                        <FiTrash2 /> Remove
                      </button>
                    </div>
                    <div className="fieldGrid compact">
                      <label className="fieldLabel fullWidthField">
                        <span>Prompt text</span>
                        <textarea
                          value={asString(prompt.prompt)}
                          onChange={(event) => handlePromptChange(index, "prompt", event.target.value)}
                        />
                      </label>
                      <label className="fieldLabel">
                        <span>Max tokens</span>
                        <input type="number" value={asNumber(prompt.max_tokens, 64)} onChange={(event) => handlePromptChange(index, "max_tokens", Number(event.target.value))} />
                      </label>
                      <label className="fieldLabel">
                        <span>Temperature</span>
                        <input type="number" step="0.05" value={asNumber(prompt.temperature, 0.7)} onChange={(event) => handlePromptChange(index, "temperature", Number(event.target.value))} />
                      </label>
                      <label className="fieldLabel">
                        <span>Top-k</span>
                        <input type="number" value={asNumber(prompt.top_k, 40)} onChange={(event) => handlePromptChange(index, "top_k", Number(event.target.value))} />
                      </label>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <details className="sectionDisclosure trainingAdvancedDisclosure" open>
              <summary className="sectionDisclosureSummary">Advanced runtime controls</summary>
              <div className="fieldGrid">
                  <label className="fieldLabel">
                    <span>BOS token</span>
                    <input value={asString(dataloaderConfig.bos_token)} onChange={(event) => handleDataloaderField(["bos_token"], event.target.value)} />
                  </label>
                  <label className="fieldLabel">
                    <span>EOS token</span>
                    <input value={asString(dataloaderConfig.eos_token)} onChange={(event) => handleDataloaderField(["eos_token"], event.target.value)} />
                  </label>
                  <label className="fieldLabel">
                    <span>PAD token</span>
                    <input value={asString(dataloaderConfig.pad_token)} onChange={(event) => handleDataloaderField(["pad_token"], event.target.value)} />
                  </label>
                  <label className="fieldLabel">
                    <span>Token dtype</span>
                    <select value={asString(dataloaderConfig.token_dtype, "int64")} onChange={(event) => handleDataloaderField(["token_dtype"], event.target.value)}>
                      <option value="int64">int64</option>
                      <option value="int32">int32</option>
                      <option value="int16">int16</option>
                      <option value="uint8">uint8</option>
                    </select>
                  </label>
                  <label className="fieldLabel">
                    <span>Pretokenize batch size</span>
                    <input type="number" value={asNumber(dataloaderConfig.pretokenize_batch_size, 1000)} onChange={(event) => handleDataloaderField(["pretokenize_batch_size"], Number(event.target.value))} />
                  </label>
                  <label className="fieldLabel">
                    <span>Cache dir</span>
                    <input value={asString(dataloaderConfig.cache_dir)} onChange={(event) => handleDataloaderField(["cache_dir"], event.target.value)} />
                  </label>
                  <label className="fieldLabel">
                    <span>Optimizer betas</span>
                    <input
                      value={Array.isArray(asRecord(trainingConfig.optimizer).betas) ? (asRecord(trainingConfig.optimizer).betas as unknown[]).map(String).join(", ") : "0.9, 0.95"}
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
                    <span>Optimizer eps</span>
                    <input type="number" step="0.00000001" value={asNumber(asRecord(trainingConfig.optimizer).eps, 1e-8)} onChange={(event) => handleTrainingField(["optimizer", "eps"], Number(event.target.value))} />
                  </label>
                  <label className="fieldLabel">
                    <span>Node split</span>
                    <select value={String(Boolean(dataloaderConfig.node_split))} onChange={(event) => handleDataloaderField(["node_split"], event.target.value === "true")}>
                      <option value="false">Disabled</option>
                      <option value="true">Enabled</option>
                    </select>
                  </label>
                  <label className="fieldLabel">
                    <span>Node rank</span>
                    <input type="number" value={asNumber(dataloaderConfig.node_rank, 0)} onChange={(event) => handleDataloaderField(["node_rank"], Number(event.target.value))} />
                  </label>
                  <label className="fieldLabel">
                    <span>Node world size</span>
                    <input type="number" value={asNumber(dataloaderConfig.node_world_size, 1)} onChange={(event) => handleDataloaderField(["node_world_size"], Number(event.target.value))} />
                  </label>
              </div>
            </details>
          </div>
        ) : (
          <div className="trainingEmpty">Loading starter templates…</div>
        )}
      </section>

      <section className="panelCard">
        <div className="panelHead">
          <div>
            <h2>Generated JSON</h2>
            <p className="panelCopy">The page stays form-first, but the underlying resolved JSON remains visible for debugging and advanced inspection.</p>
          </div>
        </div>
        <div className="trainingJsonGrid">
          <pre className="trainingCodeBlock">{prettyJson(trainingConfig)}</pre>
          <pre className="trainingCodeBlock">{prettyJson(dataloaderConfig)}</pre>
        </div>
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
                      ? "Search model configs by name, id, or file"
                      : "Search tokenizers by name, id, or artifact"
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
                  <h3>No saved model configs found.</h3>
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
                        {job.stats?.vocab_size ? ` • vocab ${formatInteger(job.stats.vocab_size)}` : ""}
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
