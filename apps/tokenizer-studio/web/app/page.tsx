"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import { FiMoon, FiSun } from "react-icons/fi";

import {
  apiBaseUrl,
  artifactDownloadUrl,
  createTrainingJob,
  fetchConfigTemplates,
  fetchTrainingJob,
  fetchTrainingJobs,
  type TrainingJob,
  uploadTrainFile,
  uploadValidationFile,
  validateDataloaderConfig,
  validateTokenizerConfig,
} from "../lib/api";
import {
  defaultDataloaderConfig,
  defaultTokenizerConfig,
} from "../lib/defaults";

type TokenizerType = "bpe" | "wordpiece" | "unigram";
type PreTokenizerType = "byte_level" | "whitespace" | "metaspace";
type DecoderType = "byte_level" | "wordpiece" | "metaspace";
type BudgetUnit = "chars" | "bytes";
type BudgetBehavior = "stop" | "truncate";
type DatasetSourceMode = "local_file" | "streaming_hf";
type FilterOperator = "==" | "!=" | ">" | ">=" | "<" | "<=" | "in" | "not in";
type ThemeMode = "white" | "dark";

const FILTER_OPERATORS: FilterOperator[] = [
  "==",
  "!=",
  ">",
  ">=",
  "<",
  "<=",
  "in",
  "not in",
];
const THEME_STORAGE_KEY = "tokenizer-studio-theme";
const WEIGHT_SUM_EPSILON = 1e-9;
const WEIGHT_SCALE = 1_000_000;
const MIN_STRICT_POSITIVE_WEIGHT = 1e-6;

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

interface TokenizerFormState {
  name: string;
  tokenizerType: TokenizerType;
  vocabSize: string;
  minFrequency: string;
  specialTokens: string;
  byteFallback: boolean;
  unkToken: string;
  preTokenizer: PreTokenizerType;
  decoder: DecoderType;
}

interface DatasetFormState {
  sourceMode: DatasetSourceMode;
  name: string;
  config: string;
  split: string;
  textColumns: string;
  trainFilePath: string;
  trainFileName: string;
  streamingDatasets: StreamingDatasetFormState[];
}

interface TrainingFormState {
  budgetLimit: string;
  budgetUnit: BudgetUnit;
  budgetBehavior: BudgetBehavior;
  evaluationThresholds: string;
  evaluationTextPath: string;
  evaluationFileName: string;
}

interface BuildResult {
  value: Record<string, unknown> | null;
  error: string | null;
}

type ToastLevel = "info" | "success" | "error";

interface ToastState {
  id: string;
  level: ToastLevel;
  message: string;
  durationMs: number;
}

function asRecord(value: unknown): Record<string, unknown> {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

function asTokenizerType(value: unknown): TokenizerType {
  if (value === "bpe" || value === "wordpiece" || value === "unigram") {
    return value;
  }
  return "bpe";
}

function asPreTokenizerType(value: unknown): PreTokenizerType {
  if (value === "byte_level" || value === "whitespace" || value === "metaspace") {
    return value;
  }
  return "byte_level";
}

function asDecoderType(value: unknown): DecoderType {
  if (value === "byte_level" || value === "wordpiece" || value === "metaspace") {
    return value;
  }
  return "byte_level";
}

function asBudgetUnit(value: unknown): BudgetUnit {
  if (value === "chars" || value === "bytes") {
    return value;
  }
  return "chars";
}

function asBudgetBehavior(value: unknown): BudgetBehavior {
  if (value === "stop" || value === "truncate") {
    return value;
  }
  return "truncate";
}

function asThemeMode(value: unknown): ThemeMode {
  return value === "dark" ? "dark" : "white";
}

function makeStreamingDatasetEntry(
  value?: Partial<Omit<StreamingDatasetFormState, "id">>
): StreamingDatasetFormState {
  return {
    id: `dataset-${Math.random().toString(36).slice(2, 10)}`,
    name: value?.name ?? "",
    config: value?.config ?? "",
    split: value?.split ?? "train",
    textColumns: value?.textColumns ?? "text",
    weight: value?.weight ?? "1",
    filters: value?.filters ?? [],
  };
}

function makeStreamingFilterEntry(
  value?: Partial<Omit<StreamingFilterFormState, "id">>
): StreamingFilterFormState {
  return {
    id: `filter-${Math.random().toString(36).slice(2, 10)}`,
    column: value?.column ?? "",
    operator: value?.operator ?? "==",
    value: value?.value ?? "",
  };
}

function splitTokens(value: string): string[] {
  return value
    .split(/[\n,]/)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
}

function fileNameFromPath(value: string): string {
  const normalized = value.replaceAll("\\", "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] ?? "";
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

function parseInteger(raw: string, label: string, min?: number): number {
  const trimmed = raw.trim();
  if (!/^-?\d+$/.test(trimmed)) {
    throw new Error(`${label} must be an integer`);
  }
  const value = Number(trimmed);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`${label} is out of range`);
  }
  if (typeof min === "number" && value < min) {
    throw new Error(`${label} must be greater than or equal to ${min}`);
  }
  return value;
}

function parsePositiveInt(raw: string, label: string): number {
  return parseInteger(raw, label, 1);
}

function parseNonNegativeNumber(raw: string, label: string): number {
  const trimmed = raw.trim();
  if (trimmed === "") {
    throw new Error(`${label} is required`);
  }
  const value = Number(trimmed);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${label} must be a non-negative number`);
  }
  return value;
}

function sanitizePositiveIntegerInput(value: string): string {
  return value.replace(/\D+/g, "");
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

function sanitizeThresholdsInput(value: string): string {
  return value.replace(/[^\d,\s]/g, "");
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

  const safeWeights = weights.map((weight) =>
    Number.isFinite(weight) ? Math.max(0, weight) : 0
  );
  const total = safeWeights.reduce((sum, weight) => sum + weight, 0);

  if (total <= WEIGHT_SUM_EPSILON) {
    const shared = 1 / safeWeights.length;
    return safeWeights.map(() => shared);
  }

  return safeWeights.map((weight) => weight / total);
}

function normalizeWeightsWithLockedIndex(
  weights: number[],
  lockedIndex: number
): number[] {
  if (weights.length === 0) {
    return [];
  }
  if (weights.length === 1) {
    return [1];
  }

  const safeWeights = weights.map((weight) =>
    Number.isFinite(weight) ? Math.max(0, weight) : 0
  );
  const normalized = new Array(safeWeights.length).fill(0);
  const lockedWeight = clamp(safeWeights[lockedIndex] ?? 0, 0, 1);
  normalized[lockedIndex] = lockedWeight;

  const otherIndexes = safeWeights
    .map((_, index) => index)
    .filter((index) => index !== lockedIndex);
  const remaining = 1 - lockedWeight;

  if (remaining <= WEIGHT_SUM_EPSILON) {
    otherIndexes.forEach((index) => {
      normalized[index] = 0;
    });
    return normalized;
  }

  const totalOthers = otherIndexes.reduce(
    (sum, index) => sum + safeWeights[index],
    0
  );

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
    typeof lockedId === "string"
      ? datasets.findIndex((entry) => entry.id === lockedId)
      : -1;
  const weights = datasets.map((entry) =>
    Math.max(0, parseWeightInput(entry.weight) ?? 0)
  );

  if (lockedIndex >= 0 && typeof lockedRawWeight === "string") {
    weights[lockedIndex] = Math.max(0, parseWeightInput(lockedRawWeight) ?? 0);
  }

  const lockedRawParsed =
    typeof lockedRawWeight === "string" ? parseWeightInput(lockedRawWeight) : null;
  const shouldLock =
    lockedIndex >= 0 && lockedRawParsed !== null && lockedRawParsed <= 1;
  const normalizedWeights =
    shouldLock
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
  roundedWeights[adjustmentIndex] = clamp(
    roundedWeights[adjustmentIndex] + drift,
    0,
    1
  );

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

function parseFilterValue(
  value: string,
  operator: FilterOperator,
  label: string
): unknown {
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
      // Fallback to comma-separated inference.
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

function buildFiltersFromForm(
  filters: StreamingFilterFormState[],
  label: string
): unknown[][] | undefined {
  if (filters.length === 0) {
    return undefined;
  }

  return filters.map((filter, index) => {
    const filterLabel = `${label} filter ${index + 1}`;
    const column = filter.column.trim();
    if (column === "") {
      throw new Error(`${filterLabel}: column is required`);
    }
    const parsedValue = parseFilterValue(filter.value, filter.operator, filterLabel);
    return [column, filter.operator, parsedValue];
  });
}

function parseThresholds(raw: string): number[] {
  const values = raw
    .split(/[\n, ]+/)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0)
    .map((entry) => parseInteger(entry, "Evaluation threshold", 1));

  if (values.length === 0) {
    throw new Error("At least one evaluation threshold is required");
  }

  return Array.from(new Set(values)).sort((a, b) => a - b);
}

function resolveEvaluationTextPath(raw: string): string {
  const value = raw.trim();
  if (value === "") {
    throw new Error("Validation file is required");
  }
  return value;
}

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function tokenizerFormFromConfig(config: Record<string, unknown>): TokenizerFormState {
  const specialTokensRaw = config.special_tokens;
  const specialTokens = Array.isArray(specialTokensRaw)
    ? specialTokensRaw.map((entry) => String(entry)).join(", ")
    : "<|endoftext|>, <|pad|>";

  return {
    name: String(config.name ?? "tokenizer"),
    tokenizerType: asTokenizerType(config.tokenizer_type),
    vocabSize: String(config.vocab_size ?? 1000),
    minFrequency: String(config.min_frequency ?? 2),
    specialTokens,
    byteFallback:
      typeof config.byte_fallback === "boolean" ? config.byte_fallback : true,
    unkToken: String(config.unk_token ?? "[UNK]"),
    preTokenizer: asPreTokenizerType(config.pre_tokenizer),
    decoder: asDecoderType(config.decoder),
  };
}

function datasetFormFromConfig(config: Record<string, unknown>): DatasetFormState {
  const datasetsRaw = Array.isArray(config.datasets) ? config.datasets : [];
  const firstDataset = asRecord(datasetsRaw[0]);
  const firstDataFiles = firstDataset.data_files;
  const sourceMode: DatasetSourceMode =
    datasetsRaw.length === 1 &&
    (typeof firstDataFiles === "string" ||
      (typeof firstDataFiles === "object" &&
        firstDataFiles !== null &&
        typeof asRecord(firstDataFiles).train === "string"))
      ? "local_file"
      : "streaming_hf";

  const textColumnsRaw = firstDataset.text_columns;
  const textColumns = Array.isArray(textColumnsRaw)
    ? textColumnsRaw.map((entry) => String(entry)).join(", ")
    : "text";

  let trainFilePath = "datasets/shake.txt";
  const dataFiles = firstDataset.data_files;
  if (typeof dataFiles === "string") {
    trainFilePath = dataFiles;
  } else if (typeof dataFiles === "object" && dataFiles !== null) {
    const record = asRecord(dataFiles);
    if (typeof record.train === "string") {
      trainFilePath = record.train;
    }
  }

  const defaultStreamingDatasets = [
    makeStreamingDatasetEntry({
      name: "HuggingFaceFW/fineweb-edu",
      split: "train",
      textColumns: "text",
      weight: "1",
    }),
  ];

  const streamingDatasets =
    sourceMode === "local_file"
      ? defaultStreamingDatasets
      : datasetsRaw.length > 0
        ? datasetsRaw.map((entry) => {
            const datasetRecord = asRecord(entry);
            const datasetTextColumnsRaw = datasetRecord.text_columns;
            const datasetTextColumns = Array.isArray(datasetTextColumnsRaw)
              ? datasetTextColumnsRaw.map((item) => String(item)).join(", ")
              : "text";
            const datasetFiltersRaw = datasetRecord.filters;
            const datasetFilters = Array.isArray(datasetFiltersRaw)
              ? datasetFiltersRaw
                  .filter((item) => Array.isArray(item) && item.length === 3)
                  .map((item) => {
                    const filterEntry = item as [unknown, unknown, unknown];
                    return makeStreamingFilterEntry({
                      column: String(filterEntry[0] ?? ""),
                      operator: asFilterOperator(filterEntry[1]),
                      value: stringifyFilterValue(filterEntry[2]),
                    });
                  })
              : [];
            return makeStreamingDatasetEntry({
              name: String(datasetRecord.name ?? ""),
              config: String(datasetRecord.config ?? ""),
              split: String(datasetRecord.split ?? "train"),
              textColumns: datasetTextColumns,
              weight: String(datasetRecord.weight ?? 1),
              filters: datasetFilters,
            });
          })
        : defaultStreamingDatasets;

  return {
    name: String(firstDataset.name ?? "text"),
    config: String(firstDataset.config ?? ""),
    split: String(firstDataset.split ?? "train"),
    textColumns,
    trainFilePath,
    trainFileName: fileNameFromPath(trainFilePath),
    sourceMode,
    streamingDatasets,
  };
}

function trainingFormFromConfig(config: Record<string, unknown>): TrainingFormState {
  const budget = asRecord(config.budget);
  const evaluationTextPath = "datasets/shake.txt";
  return {
    budgetLimit: String(budget.limit ?? 250000),
    budgetUnit: asBudgetUnit(budget.unit),
    budgetBehavior: asBudgetBehavior(budget.behavior),
    evaluationThresholds: "5,10,25",
    evaluationTextPath,
    evaluationFileName: fileNameFromPath(evaluationTextPath),
  };
}

function buildTokenizerConfigFromForm(
  form: TokenizerFormState
): Record<string, unknown> {
  const name = form.name.trim();
  if (name === "") {
    throw new Error("Tokenizer name is required");
  }

  const specialTokens = splitTokens(form.specialTokens);
  if (specialTokens.length === 0) {
    throw new Error("At least one special token is required");
  }

  const config: Record<string, unknown> = {
    name,
    tokenizer_type: form.tokenizerType,
    vocab_size: parsePositiveInt(form.vocabSize, "Vocab size"),
    min_frequency: parsePositiveInt(form.minFrequency, "Min frequency"),
    special_tokens: specialTokens,
    pre_tokenizer: form.preTokenizer,
    decoder: form.decoder,
  };

  if (form.tokenizerType === "bpe") {
    config.byte_fallback = form.byteFallback;
    if (!form.byteFallback) {
      const token = form.unkToken.trim();
      if (token === "") {
        throw new Error("Unknown token is required when BPE byte fallback is disabled");
      }
      config.unk_token = token;
    }
  }

  if (form.tokenizerType === "wordpiece") {
    const token = form.unkToken.trim();
    if (token === "") {
      throw new Error("Unknown token is required for WordPiece");
    }
    config.unk_token = token;
  }

  return config;
}

function buildDataloaderConfigFromForm(
  dataset: DatasetFormState,
  training: TrainingFormState
): Record<string, unknown> {
  let datasets: Record<string, unknown>[] = [];

  if (dataset.sourceMode === "local_file") {
    const datasetName = dataset.name.trim();
    if (datasetName === "") {
      throw new Error("Dataset name is required");
    }

    const textColumns = splitTokens(dataset.textColumns);
    if (textColumns.length === 0) {
      throw new Error("At least one text column is required");
    }

    const datasetConfig: Record<string, unknown> = {
      name: datasetName,
      split: dataset.split.trim() || "train",
      text_columns: textColumns,
      weight: 1,
    };

    const datasetConfigName = dataset.config.trim();
    if (datasetConfigName !== "") {
      datasetConfig.config = datasetConfigName;
    }

    const trainFilePath = dataset.trainFilePath.trim();
    if (trainFilePath === "") {
      throw new Error("Local train file is required");
    }
    datasetConfig.data_files = { train: trainFilePath };

    datasets = [datasetConfig];
  } else {
    if (dataset.streamingDatasets.length === 0) {
      throw new Error("At least one streaming dataset is required");
    }

    const streamingDatasetConfigs = dataset.streamingDatasets.map((entry, index) => {
      const datasetName = entry.name.trim();
      if (datasetName === "") {
        throw new Error(`Streaming dataset ${index + 1}: dataset name is required`);
      }

      const textColumns = splitTokens(entry.textColumns);
      if (textColumns.length === 0) {
        throw new Error(`Streaming dataset ${index + 1}: text columns are required`);
      }

      const parsedWeight = parseNonNegativeNumber(
        entry.weight,
        `Streaming dataset ${index + 1}: weight`
      );

      const datasetConfig: Record<string, unknown> = {
        name: datasetName,
        split: entry.split.trim() || "train",
        text_columns: textColumns,
        weight: parsedWeight,
      };

      const datasetConfigName = entry.config.trim();
      if (datasetConfigName !== "") {
        datasetConfig.config = datasetConfigName;
      }

      const parsedFilters = buildFiltersFromForm(
        entry.filters,
        `Streaming dataset ${index + 1}`
      );
      if (parsedFilters) {
        datasetConfig.filters = parsedFilters;
      }

      return {
        datasetConfig,
        parsedWeight,
      };
    });

    const totalWeight = streamingDatasetConfigs.reduce(
      (sum, entry) => sum + entry.parsedWeight,
      0
    );

    if (Math.abs(totalWeight - 1) > WEIGHT_SUM_EPSILON) {
      throw new Error(
        `Streaming dataset weights must sum to exactly 1. Current total: ${Number(
          totalWeight.toFixed(6)
        )}`
      );
    }

    const strictlyPositiveWeights = normalizeWeights(
      streamingDatasetConfigs.map((entry) =>
        entry.parsedWeight <= WEIGHT_SUM_EPSILON
          ? MIN_STRICT_POSITIVE_WEIGHT
          : entry.parsedWeight
      )
    );

    datasets = streamingDatasetConfigs.map((entry, index) => ({
      ...entry.datasetConfig,
      weight: strictlyPositiveWeights[index],
    }));
  }

  return {
    datasets,
    budget: {
      limit: parsePositiveInt(training.budgetLimit, "Text budget limit"),
      unit: training.budgetUnit,
      behavior: training.budgetBehavior,
    },
  };
}

function buildResult(factory: () => Record<string, unknown>): BuildResult {
  try {
    return {
      value: factory(),
      error: null,
    };
  } catch (error) {
    return {
      value: null,
      error: error instanceof Error ? error.message : "Unknown form error",
    };
  }
}

function formatDate(value: string | null): string {
  if (!value) {
    return "-";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function JobBadge({ status }: { status: TrainingJob["status"] }) {
  return <span className={`jobBadge jobBadge-${status}`}>{status}</span>;
}

export default function Home() {
  const [tokenizerForm, setTokenizerForm] = useState<TokenizerFormState>(() =>
    tokenizerFormFromConfig(defaultTokenizerConfig)
  );
  const [datasetForm, setDatasetForm] = useState<DatasetFormState>(() =>
    datasetFormFromConfig(defaultDataloaderConfig)
  );
  const [trainingForm, setTrainingForm] = useState<TrainingFormState>(() =>
    trainingFormFromConfig(defaultDataloaderConfig)
  );
  const [themeMode, setThemeMode] = useState<ThemeMode>("white");

  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<TrainingJob | null>(null);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [isLoadingTemplate, setIsLoadingTemplate] = useState(false);
  const [isUploadingTrainFile, setIsUploadingTrainFile] = useState(false);
  const [isUploadingValidationFile, setIsUploadingValidationFile] = useState(false);
  const [toasts, setToasts] = useState<ToastState[]>([]);
  const toastTimeoutsRef = useRef<Record<string, number>>({});
  const jobNotificationKeysRef = useRef<Set<string>>(new Set());
  const controlsDisabled =
    isSubmitting ||
    isValidating ||
    isLoadingTemplate ||
    isUploadingTrainFile ||
    isUploadingValidationFile;

  const removeToast = useCallback((toastId: string) => {
    setToasts((previous) => previous.filter((toast) => toast.id !== toastId));
    const timeoutId = toastTimeoutsRef.current[toastId];
    if (typeof timeoutId === "number") {
      window.clearTimeout(timeoutId);
      delete toastTimeoutsRef.current[toastId];
    }
  }, []);

  const notify = useCallback(
    (level: ToastLevel, message: string, durationMs = 4500) => {
      const id = `toast-${Math.random().toString(36).slice(2, 10)}`;
      const toast: ToastState = { id, level, message, durationMs };
      setToasts((previous) => [...previous, toast]);

      toastTimeoutsRef.current[id] = window.setTimeout(() => {
        removeToast(id);
      }, durationMs);
    },
    [removeToast]
  );

  useEffect(() => {
    return () => {
      Object.values(toastTimeoutsRef.current).forEach((timeoutId) => {
        window.clearTimeout(timeoutId);
      });
      toastTimeoutsRef.current = {};
    };
  }, []);

  useEffect(() => {
    const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (storedTheme) {
      setThemeMode(asThemeMode(storedTheme));
    }
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = themeMode;
    window.localStorage.setItem(THEME_STORAGE_KEY, themeMode);
  }, [themeMode]);

  const tokenizerBuild = useMemo(
    () => buildResult(() => buildTokenizerConfigFromForm(tokenizerForm)),
    [tokenizerForm]
  );

  const dataloaderBuild = useMemo(
    () => buildResult(() => buildDataloaderConfigFromForm(datasetForm, trainingForm)),
    [datasetForm, trainingForm]
  );

  const updateStreamingDataset = useCallback(
    (
      datasetId: string,
      updates: Partial<Omit<StreamingDatasetFormState, "id">>
    ) => {
      setDatasetForm((previous) => ({
        ...previous,
        streamingDatasets: previous.streamingDatasets.map((entry) =>
          entry.id === datasetId ? { ...entry, ...updates } : entry
        ),
      }));
    },
    []
  );

  const updateStreamingWeight = useCallback(
    (datasetId: string, rawWeight: string) => {
      const sanitizedWeight = sanitizeWeightInput(rawWeight);
      setDatasetForm((previous) => ({
        ...previous,
        streamingDatasets: normalizeStreamingDatasetWeights(
          previous.streamingDatasets,
          datasetId,
          sanitizedWeight
        ),
      }));
    },
    []
  );

  const addStreamingDataset = useCallback(() => {
    setDatasetForm((previous) => {
      const nextDatasets = normalizeStreamingDatasetWeights([
        ...previous.streamingDatasets,
        makeStreamingDatasetEntry(),
      ]);
      return {
        ...previous,
        streamingDatasets: nextDatasets,
      };
    });
  }, []);

  const removeStreamingDataset = useCallback((datasetId: string) => {
    setDatasetForm((previous) => {
      const nextDatasets = previous.streamingDatasets.filter(
        (entry) => entry.id !== datasetId
      );
      return {
        ...previous,
        streamingDatasets: normalizeStreamingDatasetWeights(
          nextDatasets.length > 0 ? nextDatasets : [makeStreamingDatasetEntry()]
        ),
      };
    });
  }, []);

  const updateStreamingFilter = useCallback(
    (
      datasetId: string,
      filterId: string,
      updates: Partial<Omit<StreamingFilterFormState, "id">>
    ) => {
      setDatasetForm((previous) => ({
        ...previous,
        streamingDatasets: previous.streamingDatasets.map((entry) => {
          if (entry.id !== datasetId) {
            return entry;
          }
          return {
            ...entry,
            filters: entry.filters.map((filter) =>
              filter.id === filterId ? { ...filter, ...updates } : filter
            ),
          };
        }),
      }));
    },
    []
  );

  const addStreamingFilter = useCallback((datasetId: string) => {
    setDatasetForm((previous) => ({
      ...previous,
      streamingDatasets: previous.streamingDatasets.map((entry) =>
        entry.id === datasetId
          ? { ...entry, filters: [...entry.filters, makeStreamingFilterEntry()] }
          : entry
      ),
    }));
  }, []);

  const removeStreamingFilter = useCallback((datasetId: string, filterId: string) => {
    setDatasetForm((previous) => ({
      ...previous,
      streamingDatasets: previous.streamingDatasets.map((entry) => {
        if (entry.id !== datasetId) {
          return entry;
        }
        return {
          ...entry,
          filters: entry.filters.filter((filter) => filter.id !== filterId),
        };
      }),
    }));
  }, []);

  const refreshJobs = useCallback(async () => {
    try {
      const latest = await fetchTrainingJobs();
      setJobs(latest);

      if (latest.length === 0) {
        setActiveJobId(null);
        setActiveJob(null);
        return;
      }

      if (!activeJobId) {
        setActiveJobId(latest[0].id);
        setActiveJob(latest[0]);
        return;
      }

      const selected = latest.find((job) => job.id === activeJobId);
      if (selected) {
        setActiveJob(selected);
      } else {
        setActiveJobId(latest[0].id);
        setActiveJob(latest[0]);
      }
    } catch {
      // Non-blocking background refresh failure.
    }
  }, [activeJobId]);

  useEffect(() => {
    void refreshJobs();
    const timer = window.setInterval(() => {
      void refreshJobs();
    }, 4000);

    return () => window.clearInterval(timer);
  }, [refreshJobs]);

  useEffect(() => {
    if (!activeJobId) {
      return;
    }

    let cancelled = false;

    const poll = async () => {
      try {
        const job = await fetchTrainingJob(activeJobId);
        if (cancelled) {
          return;
        }

        setActiveJob(job);

        if (job.status === "completed") {
          const completedKey = `${job.id}:completed`;
          if (!jobNotificationKeysRef.current.has(completedKey)) {
            jobNotificationKeysRef.current.add(completedKey);
            notify(
              "success",
              `Training job ${job.id.slice(0, 8)} completed. Artifact is ready.`,
              6000
            );
          }
          void refreshJobs();
        }

        if (job.status === "failed") {
          const failedKey = `${job.id}:failed`;
          if (!jobNotificationKeysRef.current.has(failedKey)) {
            jobNotificationKeysRef.current.add(failedKey);
            notify("error", job.error ?? "Training job failed", 7000);
          }
          void refreshJobs();
        }
      } catch (error) {
        if (!cancelled) {
          const message =
            error instanceof Error ? error.message : "Failed to poll job status";
          notify("error", message, 7000);
        }
      }
    };

    void poll();
    const timer = window.setInterval(() => {
      void poll();
    }, 1500);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeJobId, notify, refreshJobs]);

  const handleTrainFileSelected = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    notify("info", `Uploading local train file: ${file.name}`, 2500);
    setIsUploadingTrainFile(true);

    try {
      const uploadedFile = await uploadTrainFile(file);
      setDatasetForm((previous) => ({
        ...previous,
        trainFilePath: uploadedFile.file_path,
        trainFileName: uploadedFile.file_name,
      }));
      notify("success", `Uploaded ${uploadedFile.file_name}. Ready to validate.`);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to upload local train file";
      notify("error", `Could not upload local train file. ${message}`, 7000);
    } finally {
      setIsUploadingTrainFile(false);
      event.target.value = "";
    }
  };

  const handleValidationFileSelected = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    notify("info", `Uploading validation file: ${file.name}`, 2500);
    setIsUploadingValidationFile(true);

    try {
      const uploadedFile = await uploadValidationFile(file);
      setTrainingForm((previous) => ({
        ...previous,
        evaluationTextPath: uploadedFile.file_path,
        evaluationFileName: uploadedFile.file_name,
      }));
      notify("success", `Uploaded ${uploadedFile.file_name}. Ready to validate.`);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to upload validation file";
      notify("error", `Could not upload validation file. ${message}`, 7000);
    } finally {
      setIsUploadingValidationFile(false);
      event.target.value = "";
    }
  };

  const handleLoadStreamingTemplate = async () => {
    notify("info", "Loading streaming dataset template...", 2500);
    setIsLoadingTemplate(true);

    try {
      const templates = await fetchConfigTemplates();
      const templateConfig = asRecord(templates.dataloader_config_template);
      const templateDatasetForm = datasetFormFromConfig(templateConfig);
      const templateTrainingForm = trainingFormFromConfig(templateConfig);

      setDatasetForm((previous) => ({
        ...previous,
        sourceMode: "streaming_hf",
        streamingDatasets: normalizeStreamingDatasetWeights(
          templateDatasetForm.streamingDatasets
        ),
      }));
      setTrainingForm((previous) => ({
        ...previous,
        budgetLimit: templateTrainingForm.budgetLimit,
        budgetUnit: templateTrainingForm.budgetUnit,
        budgetBehavior: templateTrainingForm.budgetBehavior,
      }));
      notify("success", "Loaded streaming template datasets.");
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Failed to load streaming template";
      notify("error", `Could not load streaming template. ${message}`, 7000);
    } finally {
      setIsLoadingTemplate(false);
    }
  };

  const handleValidate = async () => {
    notify("info", "Validating configs with API...", 2500);
    setIsValidating(true);

    try {
      if (!tokenizerBuild.value) {
        throw new Error(tokenizerBuild.error ?? "Tokenizer config is invalid");
      }
      if (!dataloaderBuild.value) {
        throw new Error(dataloaderBuild.error ?? "Dataloader config is invalid");
      }
      resolveEvaluationTextPath(trainingForm.evaluationTextPath);

      await Promise.all([
        validateTokenizerConfig(tokenizerBuild.value),
        validateDataloaderConfig(dataloaderBuild.value),
      ]);

      notify("success", "Validation passed. You can start training.");
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Validation failed unexpectedly";
      notify("error", `Validation failed. ${message}`, 7000);
    } finally {
      setIsValidating(false);
    }
  };

  const handleTrain = async () => {
    notify("info", "Submitting training job...", 2500);
    setIsSubmitting(true);

    try {
      if (!tokenizerBuild.value) {
        throw new Error(tokenizerBuild.error ?? "Tokenizer config is invalid");
      }
      if (!dataloaderBuild.value) {
        throw new Error(dataloaderBuild.error ?? "Dataloader config is invalid");
      }

      const thresholds = parseThresholds(trainingForm.evaluationThresholds);
      const evaluationTextPath = resolveEvaluationTextPath(
        trainingForm.evaluationTextPath
      );

      const job = await createTrainingJob({
        tokenizer_config: tokenizerBuild.value,
        dataloader_config: dataloaderBuild.value,
        evaluation_thresholds: thresholds,
        evaluation_text_path: evaluationTextPath,
      });

      setActiveJobId(job.id);
      setActiveJob(job);
      notify("success", `Training job ${job.id.slice(0, 8)} started.`);
      await refreshJobs();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to create training job";
      notify("error", `Could not start training. ${message}`, 7000);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="studioPage">
      <header className="heroCard">
        <div className="heroTopRow">
          <p className="heroTag">Tokenizer Studio</p>
          <button
            type="button"
            className="themeToggle"
            onClick={() =>
              setThemeMode((previous) =>
                previous === "white" ? "dark" : "white"
              )
            }
            aria-label={
              themeMode === "dark"
                ? "Switch to white theme"
                : "Switch to dark theme"
            }
            title={
              themeMode === "dark"
                ? "Switch to white theme"
                : "Switch to dark theme"
            }
          >
            {themeMode === "dark" ? <FiSun aria-hidden="true" /> : <FiMoon aria-hidden="true" />}
          </button>
        </div>
        <h1>Train a tokenizer in 3 steps</h1>
        <p>
          Set basic tokenizer fields, point to your dataset, validate once, and run.
          The interface keeps advanced options hidden by default.
        </p>
        <div className="heroMeta">API: {apiBaseUrl()}</div>
      </header>

      <section className="card sectionBlock">
        <details className="sectionCollapse" open>
          <summary className="sectionCollapseSummary">
            <span className="sectionCollapseHeading">1. Quick Setup</span>
          </summary>
          <div className="setupGrid">
            <article className="miniPanel">
            <h3>Tokenizer</h3>
            <div className="fieldGrid">
              <label>
                Name
                <input
                  value={tokenizerForm.name}
                  onChange={(event) =>
                    setTokenizerForm((previous) => ({
                      ...previous,
                      name: event.target.value,
                    }))
                  }
                />
              </label>

              <label>
                Type
                <select
                  value={tokenizerForm.tokenizerType}
                  onChange={(event) =>
                    setTokenizerForm((previous) => ({
                      ...previous,
                      tokenizerType: event.target.value as TokenizerType,
                    }))
                  }
                >
                  <option value="bpe">BPE</option>
                  <option value="wordpiece">WordPiece</option>
                  <option value="unigram">Unigram</option>
                </select>
              </label>

              <label>
                Vocab size
                <input
                  inputMode="numeric"
                  pattern="[0-9]*"
                  value={tokenizerForm.vocabSize}
                  onChange={(event) =>
                    setTokenizerForm((previous) => ({
                      ...previous,
                      vocabSize: sanitizePositiveIntegerInput(event.target.value),
                    }))
                  }
                />
              </label>

              <label>
                Min frequency
                <input
                  inputMode="numeric"
                  pattern="[0-9]*"
                  value={tokenizerForm.minFrequency}
                  onChange={(event) =>
                    setTokenizerForm((previous) => ({
                      ...previous,
                      minFrequency: sanitizePositiveIntegerInput(event.target.value),
                    }))
                  }
                />
              </label>
            </div>

            <label>
              Special tokens (comma or new line)
              <textarea
                rows={2}
                value={tokenizerForm.specialTokens}
                onChange={(event) =>
                  setTokenizerForm((previous) => ({
                    ...previous,
                    specialTokens: event.target.value,
                  }))
                }
              />
            </label>

            {tokenizerForm.tokenizerType === "bpe" ? (
              <div className="inlineRow">
                <label className="checkLabel">
                  <input
                    type="checkbox"
                    checked={tokenizerForm.byteFallback}
                    onChange={(event) =>
                      setTokenizerForm((previous) => ({
                        ...previous,
                        byteFallback: event.target.checked,
                      }))
                    }
                  />
                  Byte fallback
                </label>

                {!tokenizerForm.byteFallback ? (
                  <label className="inlineField">
                    Unknown token
                    <input
                      value={tokenizerForm.unkToken}
                      onChange={(event) =>
                        setTokenizerForm((previous) => ({
                          ...previous,
                          unkToken: event.target.value,
                        }))
                      }
                    />
                  </label>
                ) : null}
              </div>
            ) : null}

            {tokenizerForm.tokenizerType === "wordpiece" ? (
              <label>
                Unknown token
                <input
                  value={tokenizerForm.unkToken}
                  onChange={(event) =>
                    setTokenizerForm((previous) => ({
                      ...previous,
                      unkToken: event.target.value,
                    }))
                  }
                />
              </label>
            ) : null}

            <details className="advancedPanel">
              <summary>Advanced tokenizer fields</summary>
              <div className="fieldGrid">
                <label>
                  Pre-tokenizer
                  <select
                    value={tokenizerForm.preTokenizer}
                    onChange={(event) =>
                      setTokenizerForm((previous) => ({
                        ...previous,
                        preTokenizer: event.target.value as PreTokenizerType,
                      }))
                    }
                  >
                    <option value="byte_level">byte_level</option>
                    <option value="whitespace">whitespace</option>
                    <option value="metaspace">metaspace</option>
                  </select>
                </label>

                <label>
                  Decoder
                  <select
                    value={tokenizerForm.decoder}
                    onChange={(event) =>
                      setTokenizerForm((previous) => ({
                        ...previous,
                        decoder: event.target.value as DecoderType,
                      }))
                    }
                  >
                    <option value="byte_level">byte_level</option>
                    <option value="wordpiece">wordpiece</option>
                    <option value="metaspace">metaspace</option>
                  </select>
                </label>
              </div>
            </details>

            <p className={`hint ${tokenizerBuild.error ? "hint-error" : "hint-ok"}`}>
              {tokenizerBuild.error ?? "Tokenizer config looks valid."}
            </p>
          </article>

          <article className="miniPanel">
            <h3>Dataset + Budget</h3>
            <div className="sourceModeRow">
              <span>Dataset source</span>
              <div className="modeSwitch">
                <button
                  type="button"
                  className={`modeSwitchButton ${
                    datasetForm.sourceMode === "local_file"
                      ? "modeSwitchButton-active"
                      : ""
                  }`}
                  disabled={controlsDisabled}
                  onClick={() =>
                    setDatasetForm((previous) => ({
                      ...previous,
                      sourceMode: "local_file",
                    }))
                  }
                >
                  Local file
                </button>
                <button
                  type="button"
                  className={`modeSwitchButton ${
                    datasetForm.sourceMode === "streaming_hf"
                      ? "modeSwitchButton-active"
                      : ""
                  }`}
                  disabled={controlsDisabled}
                  onClick={() =>
                    setDatasetForm((previous) => ({
                      ...previous,
                      sourceMode: "streaming_hf",
                      streamingDatasets: normalizeStreamingDatasetWeights(
                        previous.streamingDatasets
                      ),
                    }))
                  }
                >
                  Streaming HF datasets
                </button>
              </div>
            </div>
            <p className="fieldNote">
              Switch between local data files and streaming Hugging Face datasets.
            </p>

            {datasetForm.sourceMode === "local_file" ? (
              <div className="fieldGrid">
                <label className="fullWidthField">
                  Local train file
                  <input
                    type="file"
                    onChange={handleTrainFileSelected}
                    disabled={controlsDisabled}
                  />
                  <span className="fieldNote">
                    {datasetForm.trainFilePath === ""
                      ? "Choose a local file to continue."
                      : `Using: ${datasetForm.trainFileName} (${datasetForm.trainFilePath})`}
                  </span>
                </label>

                <label>
                  Dataset builder
                  <input
                    value={datasetForm.name}
                    onChange={(event) =>
                      setDatasetForm((previous) => ({
                        ...previous,
                        name: event.target.value,
                      }))
                    }
                    placeholder="text"
                  />
                </label>

                <label>
                  Dataset config (optional)
                  <input
                    value={datasetForm.config}
                    onChange={(event) =>
                      setDatasetForm((previous) => ({
                        ...previous,
                        config: event.target.value,
                      }))
                    }
                  />
                </label>

                <label>
                  Split
                  <input
                    value={datasetForm.split}
                    onChange={(event) =>
                      setDatasetForm((previous) => ({
                        ...previous,
                        split: event.target.value,
                      }))
                    }
                    placeholder="train"
                  />
                </label>

                <label>
                  Text columns
                  <input
                    value={datasetForm.textColumns}
                    onChange={(event) =>
                      setDatasetForm((previous) => ({
                        ...previous,
                        textColumns: event.target.value,
                      }))
                    }
                    placeholder="text"
                  />
                </label>
              </div>
            ) : (
              <div className="datasetConfigurator">
                <div className="actionRow compactActionRow">
                  <button
                    type="button"
                    className="secondaryButton"
                    onClick={addStreamingDataset}
                    disabled={controlsDisabled}
                  >
                    Add dataset
                  </button>
                  <button
                    type="button"
                    className="secondaryButton"
                    onClick={handleLoadStreamingTemplate}
                    disabled={controlsDisabled}
                  >
                    {isLoadingTemplate
                      ? "Loading template..."
                      : "Load streaming template"}
                  </button>
                </div>

                <div className="datasetList">
                  {datasetForm.streamingDatasets.map((entry, index) => (
                    <div key={entry.id} className="datasetCard">
                      <div className="datasetCardHeader">
                        <strong>Streaming dataset {index + 1}</strong>
                        <button
                          type="button"
                          className="textButton"
                          onClick={() => removeStreamingDataset(entry.id)}
                          disabled={controlsDisabled || datasetForm.streamingDatasets.length <= 1}
                        >
                          Remove
                        </button>
                      </div>

                      <div className="fieldGrid">
                        <label>
                          HF dataset name
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

                        <label>
                          Config (optional)
                          <input
                            value={entry.config}
                            onChange={(event) =>
                              updateStreamingDataset(entry.id, {
                                config: event.target.value,
                              })
                            }
                          />
                        </label>

                        <label>
                          Split
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

                        <label>
                          Weight
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

                        <label className="fullWidthField">
                          Text columns
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

                        <div className="fullWidthField filterBuilder">
                          <div className="filterBuilderHeader">
                            <span className="filterBuilderTitle">Filters (optional)</span>
                            <button
                              type="button"
                              className="secondaryButton"
                              onClick={() => addStreamingFilter(entry.id)}
                              disabled={controlsDisabled}
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
                                  <label>
                                    Column
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

                                  <label>
                                    Operator
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

                                  <label>
                                    Value
                                    <input
                                      value={filter.value}
                                      onChange={(event) =>
                                        updateStreamingFilter(entry.id, filter.id, {
                                          value: event.target.value,
                                        })
                                      }
                                      placeholder={
                                        filter.operator === "in" || filter.operator === "not in"
                                          ? '["en", "de"] or en,de'
                                          : "en, true, 0.95, {\"k\":1}"
                                      }
                                    />
                                  </label>

                                  <button
                                    type="button"
                                    className="textButton filterRemoveButton"
                                    onClick={() => removeStreamingFilter(entry.id, filter.id)}
                                    disabled={controlsDisabled}
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
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="fieldGrid">
              <label>
                Budget limit
                <input
                  inputMode="numeric"
                  pattern="[0-9]*"
                  value={trainingForm.budgetLimit}
                  onChange={(event) =>
                    setTrainingForm((previous) => ({
                      ...previous,
                      budgetLimit: sanitizePositiveIntegerInput(event.target.value),
                    }))
                  }
                />
              </label>

              <label>
                Budget unit
                <select
                  value={trainingForm.budgetUnit}
                  onChange={(event) =>
                    setTrainingForm((previous) => ({
                      ...previous,
                      budgetUnit: event.target.value as BudgetUnit,
                    }))
                  }
                >
                  <option value="chars">chars</option>
                  <option value="bytes">bytes</option>
                </select>
              </label>

              <label>
                Budget behavior
                <select
                  value={trainingForm.budgetBehavior}
                  onChange={(event) =>
                    setTrainingForm((previous) => ({
                      ...previous,
                      budgetBehavior: event.target.value as BudgetBehavior,
                    }))
                  }
                >
                  <option value="truncate">truncate</option>
                  <option value="stop">stop</option>
                </select>
              </label>
            </div>

            <p className={`hint ${dataloaderBuild.error ? "hint-error" : "hint-ok"}`}>
              {dataloaderBuild.error ?? "Dataloader config looks valid."}
            </p>
          </article>
          </div>
        </details>
      </section>

      <section className="card sectionBlock">
        <h2>2. Validate and Run</h2>
        <label className="fullWidthField">
          Validation file
          <input
            type="file"
            onChange={handleValidationFileSelected}
            disabled={controlsDisabled}
          />
          <span className="fieldNote">
            {trainingForm.evaluationTextPath === ""
              ? "Choose a validation file to continue."
              : `Using: ${trainingForm.evaluationFileName} (${trainingForm.evaluationTextPath})`}
          </span>
        </label>

        <div className="fieldGrid">
          <label>
            Evaluation thresholds
            <input
              value={trainingForm.evaluationThresholds}
              onChange={(event) =>
                setTrainingForm((previous) => ({
                  ...previous,
                  evaluationThresholds: sanitizeThresholdsInput(event.target.value),
                }))
              }
              placeholder="5,10,25"
            />
          </label>
        </div>

        <div className="actionRow">
          <button
            type="button"
            className="secondaryButton"
            onClick={handleValidate}
            disabled={
              isSubmitting ||
              isValidating ||
              isUploadingTrainFile ||
              isUploadingValidationFile
            }
          >
            {isValidating ? "Validating..." : "Validate"}
          </button>
          <button
            type="button"
            className="primaryButton"
            onClick={handleTrain}
            disabled={
              isSubmitting ||
              isValidating ||
              isUploadingTrainFile ||
              isUploadingValidationFile
            }
          >
            {isSubmitting ? "Starting..." : "Start Training"}
          </button>
        </div>
      </section>

      <section className="card sectionBlock">
        <details className="advancedPanel">
          <summary>3. Preview generated JSON</summary>
          <div className="jsonGrid">
            <label>
              Tokenizer JSON
              <pre>{tokenizerBuild.value ? prettyJson(tokenizerBuild.value) : "Invalid config"}</pre>
            </label>

            <label>
              Dataloader JSON
              <pre>{dataloaderBuild.value ? prettyJson(dataloaderBuild.value) : "Invalid config"}</pre>
            </label>
          </div>
        </details>
      </section>

      <section className="jobsGrid">
        <article className="card sectionBlock">
          <div className="sectionHeader">
            <h2>Active Job</h2>
            {activeJob ? <JobBadge status={activeJob.status} /> : null}
          </div>

          {activeJob ? (
            <>
              <p className="metaLine">
                <strong>ID:</strong> {activeJob.id}
              </p>
              <p className="metaLine">
                <strong>Stage:</strong> {activeJob.stage}
              </p>
              <p className="metaLine">
                <strong>Progress:</strong> {formatPercent(activeJob.progress)}
              </p>

              <div className="progressTrack">
                <div
                  className="progressFill"
                  style={{ width: `${Math.max(0, Math.min(activeJob.progress * 100, 100))}%` }}
                />
              </div>

              <p className="metaLine">
                <strong>Created:</strong> {formatDate(activeJob.created_at)}
              </p>
              <p className="metaLine">
                <strong>Started:</strong> {formatDate(activeJob.started_at)}
              </p>
              <p className="metaLine">
                <strong>Finished:</strong> {formatDate(activeJob.finished_at)}
              </p>

              {activeJob.status === "completed" && activeJob.stats ? (
                <div className="statsGrid">
                  <div>
                    <span>Tokens / Char</span>
                    <strong>{activeJob.stats.token_per_char.toFixed(4)}</strong>
                  </div>
                  <div>
                    <span>Vocab</span>
                    <strong>{activeJob.stats.vocab_size}</strong>
                  </div>
                  <div>
                    <span>Used tokens</span>
                    <strong>{activeJob.stats.num_used_tokens}</strong>
                  </div>
                  <div>
                    <span>Unused tokens</span>
                    <strong>{activeJob.stats.num_unused_tokens}</strong>
                  </div>
                </div>
              ) : null}

              {activeJob.artifact_file ? (
                <a
                  className="downloadLink"
                  href={artifactDownloadUrl(activeJob.id)}
                  target="_blank"
                  rel="noreferrer"
                >
                  Download {activeJob.artifact_file}
                </a>
              ) : null}

              {activeJob.error ? <p className="hint hint-error">{activeJob.error}</p> : null}
            </>
          ) : (
            <p className="metaLine">No job selected yet.</p>
          )}
        </article>

        <article className="card sectionBlock">
          <div className="sectionHeader">
            <h2>Recent Jobs</h2>
            <button type="button" className="textButton" onClick={() => void refreshJobs()}>
              Refresh
            </button>
          </div>

          {jobs.length === 0 ? (
            <p className="metaLine">No jobs yet.</p>
          ) : (
            <div className="jobsList">
              {jobs.map((job) => (
                <button
                  key={job.id}
                  type="button"
                  className={`jobRow ${activeJobId === job.id ? "jobRow-active" : ""}`}
                  onClick={() => {
                    setActiveJobId(job.id);
                    setActiveJob(job);
                  }}
                >
                  <div>
                    <strong>{String(job.tokenizer_config.name ?? "tokenizer")}</strong>
                    <p>{job.id.slice(0, 8)}</p>
                  </div>
                  <div>
                    <JobBadge status={job.status} />
                    <p>{formatPercent(job.progress)}</p>
                  </div>
                </button>
              ))}
            </div>
          )}
        </article>
      </section>

      <aside className="toastViewport" aria-live="polite" aria-atomic="false">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast toast-${toast.level}`}>
            <div className="toastContent">
              <p>{toast.message}</p>
              <button
                type="button"
                className="toastClose"
                onClick={() => removeToast(toast.id)}
                aria-label="Dismiss notification"
              >
                ×
              </button>
            </div>
            <div
              className="toastProgress"
              style={{ animationDuration: `${toast.durationMs}ms` }}
            />
          </div>
        ))}
      </aside>
    </main>
  );
}
