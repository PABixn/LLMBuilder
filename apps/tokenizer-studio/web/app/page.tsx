"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type MouseEvent,
} from "react";
import { FiMoon, FiSun, FiX } from "react-icons/fi";

import {
  artifactDownloadUrl,
  createTrainingJob,
  downloadJobArtifact,
  fetchConfigTemplates,
  fetchLocalTrainFileStats,
  fetchTrainingJob,
  fetchTrainingJobs,
  previewJobTokenizer,
  type TrainingJob,
  type TokenizerPreviewResult,
  type TokenizerPreviewToken,
  uploadTrainFile,
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
type SettingsCategory = "tokenizer" | "dataset" | "training";

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
const TOKENIZER_FORM_STORAGE_KEY = "tokenizer-studio-tokenizer-form";
const DATASET_FORM_STORAGE_KEY = "tokenizer-studio-dataset-form";
const TRAINING_FORM_STORAGE_KEY = "tokenizer-studio-training-form";
const ACTIVE_JOB_STORAGE_KEY = "tokenizer-studio-active-job-id";
const PREVIEW_TEXT_STORAGE_KEY = "tokenizer-studio-preview-text";
const SETTINGS_CATEGORY_HASH_MAP: Record<SettingsCategory, string> = {
  tokenizer: "#settings-tokenizer",
  dataset: "#settings-dataset",
  training: "#settings-training",
};
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

interface LocalTrainFileFormState {
  id: string;
  fileName: string;
  filePath: string;
  sizeBytes: number | null;
  sizeChars: number | null;
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
  localTrainFiles: LocalTrainFileFormState[];
  hfToken: string;
  streamingDatasets: StreamingDatasetFormState[];
}

interface TrainingFormState {
  budgetLimit: string;
  budgetUnit: BudgetUnit;
  budgetBehavior: BudgetBehavior;
  evaluationThresholds: string;
}

interface BuildResult {
  value: Record<string, unknown> | null;
  error: string | null;
}

interface PreviewSegment {
  kind: "plain" | "token";
  text: string;
  token?: TokenizerPreviewToken;
}

type ToastLevel = "info" | "success" | "error";

interface ToastState {
  id: string;
  level: ToastLevel;
  message: string;
  durationMs: number;
}

type TauriInvokeFn = (
  command: string,
  args?: Record<string, unknown>
) => Promise<unknown>;

function getTauriInvoke(): TauriInvokeFn | null {
  if (typeof window === "undefined") {
    return null;
  }

  const tauriWindow = window as Window & {
    __TAURI__?: {
      core?: {
        invoke?: TauriInvokeFn;
      };
    };
    __TAURI_INTERNALS__?: {
      invoke?: TauriInvokeFn;
    };
  };

  return (
    tauriWindow.__TAURI__?.core?.invoke ??
    tauriWindow.__TAURI_INTERNALS__?.invoke ??
    null
  );
}

function triggerBlobDownload(blob: Blob, fileName: string): void {
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = fileName;
  link.rel = "noreferrer";
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => {
    URL.revokeObjectURL(objectUrl);
  }, 1_000);
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

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function readStoredValue(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function readStoredJson(key: string): unknown | null {
  const raw = readStoredValue(key);
  if (raw === null || raw.trim() === "") {
    return null;
  }
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function writeStoredValue(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Ignore quota/unavailable storage failures in local mode.
  }
}

function writeStoredJson(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Ignore quota/unavailable storage failures in local mode.
  }
}

function removeStoredValue(key: string): void {
  try {
    window.localStorage.removeItem(key);
  } catch {
    // Ignore quota/unavailable storage failures in local mode.
  }
}

function hydrateStreamingFilter(value: unknown): StreamingFilterFormState {
  const record = asRecord(value);
  const storedId = asString(record.id, "").trim();

  return makeStreamingFilterEntry({
    id: storedId === "" ? undefined : storedId,
    column: asString(record.column, ""),
    operator: asFilterOperator(record.operator),
    value: asString(record.value, ""),
  });
}

function hydrateStreamingDataset(value: unknown): StreamingDatasetFormState {
  const record = asRecord(value);
  const storedId = asString(record.id, "").trim();
  const filtersRaw = Array.isArray(record.filters) ? record.filters : [];

  return makeStreamingDatasetEntry({
    id: storedId === "" ? undefined : storedId,
    name: asString(record.name, ""),
    config: asString(record.config, ""),
    split: asString(record.split, "train"),
    textColumns: asString(record.textColumns, "text"),
    weight: sanitizeWeightInput(asString(record.weight, "1")),
    filters: filtersRaw.map((entry) => hydrateStreamingFilter(entry)),
  });
}

function hydrateLocalTrainFile(value: unknown): LocalTrainFileFormState | null {
  const record = asRecord(value);
  const filePath = asString(record.filePath, "").trim();
  if (filePath === "") {
    return null;
  }

  const fileName = asString(record.fileName, "").trim() || fileNameFromPath(filePath);
  const sizeBytesRaw = record.sizeBytes;
  const sizeBytes =
    typeof sizeBytesRaw === "number" && Number.isFinite(sizeBytesRaw) && sizeBytesRaw >= 0
      ? sizeBytesRaw
      : null;
  const sizeCharsRaw = record.sizeChars;
  const sizeChars =
    typeof sizeCharsRaw === "number" && Number.isFinite(sizeCharsRaw) && sizeCharsRaw >= 0
      ? sizeCharsRaw
      : sizeBytes;

  return makeLocalTrainFileEntry({
    id: asString(record.id, "").trim() || undefined,
    fileName,
    filePath,
    sizeBytes,
    sizeChars,
  });
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
        typeof entry.sizeBytes === "number" &&
        Number.isFinite(entry.sizeBytes) &&
        entry.sizeBytes >= 0
          ? entry.sizeBytes
          : null,
      sizeChars:
        typeof entry.sizeChars === "number" &&
        Number.isFinite(entry.sizeChars) &&
        entry.sizeChars >= 0
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

function hydrateTokenizerForm(
  value: unknown,
  fallback: TokenizerFormState
): TokenizerFormState {
  const record = asRecord(value);

  return {
    name: asString(record.name, fallback.name),
    tokenizerType: asTokenizerType(record.tokenizerType),
    vocabSize: sanitizePositiveIntegerInput(
      asString(record.vocabSize, fallback.vocabSize)
    ),
    minFrequency: sanitizePositiveIntegerInput(
      asString(record.minFrequency, fallback.minFrequency)
    ),
    specialTokens: asString(record.specialTokens, fallback.specialTokens),
    byteFallback:
      typeof record.byteFallback === "boolean"
        ? record.byteFallback
        : fallback.byteFallback,
    unkToken: asString(record.unkToken, fallback.unkToken),
    preTokenizer: asPreTokenizerType(record.preTokenizer),
    decoder: asDecoderType(record.decoder),
  };
}

function hydrateDatasetForm(value: unknown, fallback: DatasetFormState): DatasetFormState {
  const record = asRecord(value);
  const sourceMode: DatasetSourceMode =
    record.sourceMode === "local_file" || record.sourceMode === "streaming_hf"
      ? record.sourceMode
      : fallback.sourceMode;
  const streamingDatasetsRaw = Array.isArray(record.streamingDatasets)
    ? record.streamingDatasets
    : [];
  const streamingDatasets =
    streamingDatasetsRaw.length > 0
      ? streamingDatasetsRaw.map((entry) => hydrateStreamingDataset(entry))
      : fallback.streamingDatasets.map((entry) => makeStreamingDatasetEntry(entry));
  const hasStoredLocalTrainFiles = Array.isArray(record.localTrainFiles);
  const localTrainFilesRaw: unknown[] = hasStoredLocalTrainFiles
    ? (record.localTrainFiles as unknown[])
    : [];
  const localTrainFilesFromStorage = normalizeLocalTrainFiles(
    localTrainFilesRaw
      .map((entry: unknown) => hydrateLocalTrainFile(entry))
      .filter((entry: LocalTrainFileFormState | null): entry is LocalTrainFileFormState =>
        entry !== null
      )
  );
  const legacyTrainFilePath = asString(record.trainFilePath, "").trim();
  const legacyTrainFileName = asString(record.trainFileName, "").trim();
  const legacyLocalTrainFile =
    legacyTrainFilePath === ""
      ? []
      : [
          makeLocalTrainFileEntry({
            filePath: legacyTrainFilePath,
            fileName: legacyTrainFileName || fileNameFromPath(legacyTrainFilePath),
          }),
        ];
  const localTrainFiles =
    localTrainFilesFromStorage.length > 0
      ? localTrainFilesFromStorage
      : hasStoredLocalTrainFiles
        ? []
        : legacyLocalTrainFile.length > 0
          ? legacyLocalTrainFile
          : fallback.localTrainFiles.map((entry) => makeLocalTrainFileEntry(entry));

  return {
    sourceMode,
    localTrainFiles: normalizeLocalTrainFiles(localTrainFiles),
    hfToken: asString(record.hfToken, fallback.hfToken),
    streamingDatasets: normalizeStreamingDatasetWeights(
      streamingDatasets.length > 0 ? streamingDatasets : [makeStreamingDatasetEntry()]
    ),
  };
}

function hydrateTrainingForm(
  value: unknown,
  fallback: TrainingFormState
): TrainingFormState {
  const record = asRecord(value);

  return {
    budgetLimit: sanitizePositiveIntegerInput(
      asString(record.budgetLimit, fallback.budgetLimit)
    ),
    budgetUnit: asBudgetUnit(record.budgetUnit),
    budgetBehavior: asBudgetBehavior(record.budgetBehavior),
    evaluationThresholds: sanitizeThresholdsInput(
      asString(record.evaluationThresholds, fallback.evaluationThresholds)
    ),
  };
}

function hydratePreviewText(value: unknown, fallback: string): string {
  if (typeof value !== "string") {
    return fallback;
  }
  return value.slice(0, 50_000);
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

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function makePreviewSegments(text: string, tokens: TokenizerPreviewToken[]): PreviewSegment[] {
  if (text.length === 0) {
    return [];
  }

  const segments: PreviewSegment[] = [];
  let cursor = 0;

  for (const token of tokens) {
    const start = Math.max(0, Math.min(text.length, Math.trunc(token.start)));
    const end = Math.max(0, Math.min(text.length, Math.trunc(token.end)));

    if (end <= start || start < cursor) {
      continue;
    }

    if (start > cursor) {
      segments.push({
        kind: "plain",
        text: text.slice(cursor, start),
      });
    }

    segments.push({
      kind: "token",
      text: text.slice(start, end),
      token,
    });
    cursor = end;
  }

  if (cursor < text.length) {
    segments.push({
      kind: "plain",
      text: text.slice(cursor),
    });
  }

  return segments;
}

function displayTokenLabel(value: string): string {
  return value
    .replaceAll("Ġ", "[space]")
    .replaceAll("▁", "[space]")
    .replaceAll("Ċ", "[\\n]")
    .replaceAll("\r\n", "[\\r\\n]\n")
    .replaceAll(" ", "[space]")
    .replaceAll("\n", "[\\n]\n")
    .replaceAll("\t", "[\\t]");
}

function tokenHue(index: number): number {
  return (index * 37) % 360;
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
  const firstHfToken =
    typeof firstDataset.hf_token === "string" ? firstDataset.hf_token : "";
  const localTrainPaths = extractTrainFilePaths(firstDataset.data_files);
  const sourceMode: DatasetSourceMode =
    datasetsRaw.length === 1 && localTrainPaths.length > 0
      ? "local_file"
      : "streaming_hf";
  const localTrainFiles = normalizeLocalTrainFiles(
    localTrainPaths.map((filePath) =>
      makeLocalTrainFileEntry({
        filePath,
        fileName: fileNameFromPath(filePath),
      })
    )
  );

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
    localTrainFiles,
    hfToken: firstHfToken,
    sourceMode,
    streamingDatasets,
  };
}

function trainingFormFromConfig(config: Record<string, unknown>): TrainingFormState {
  const budget = asRecord(config.budget);
  return {
    budgetLimit: String(budget.limit ?? 250000),
    budgetUnit: asBudgetUnit(budget.unit),
    budgetBehavior: asBudgetBehavior(budget.behavior),
    evaluationThresholds: "5,10,25",
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
  const hfToken = dataset.hfToken.trim();

  if (dataset.sourceMode === "local_file") {
    const localTrainPaths = normalizeLocalTrainFiles(dataset.localTrainFiles)
      .map((entry) => entry.filePath.trim())
      .filter((entry) => entry !== "");
    if (localTrainPaths.length === 0) {
      throw new Error("Add at least one local train file");
    }
    const trainDataFiles =
      localTrainPaths.length === 1 ? localTrainPaths[0] : localTrainPaths;

    const datasetConfig: Record<string, unknown> = {
      name: "text",
      split: "train",
      text_columns: ["text"],
      weight: 1,
      data_files: { train: trainDataFiles },
    };

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

      if (hfToken !== "") {
        datasetConfig.hf_token = hfToken;
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

function buildDatasetSelectionFromForm(dataset: DatasetFormState): Record<string, unknown> {
  if (dataset.sourceMode === "local_file") {
    const localTrainPaths = normalizeLocalTrainFiles(dataset.localTrainFiles)
      .map((entry) => entry.filePath.trim())
      .filter((entry) => entry !== "");
    if (localTrainPaths.length === 0) {
      throw new Error("Add at least one local train file");
    }
    return { source_mode: "local_file", file_count: localTrainPaths.length };
  }

  if (dataset.streamingDatasets.length === 0) {
    throw new Error("At least one streaming dataset is required");
  }

  const parsedWeights = dataset.streamingDatasets.map((entry, index) => {
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

    buildFiltersFromForm(entry.filters, `Streaming dataset ${index + 1}`);
    return parsedWeight;
  });

  const totalWeight = parsedWeights.reduce((sum, weight) => sum + weight, 0);
  if (Math.abs(totalWeight - 1) > WEIGHT_SUM_EPSILON) {
    throw new Error(
      `Streaming dataset weights must sum to exactly 1. Current total: ${Number(
        totalWeight.toFixed(6)
      )}`
    );
  }

  return { source_mode: "streaming_hf", dataset_count: dataset.streamingDatasets.length };
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

type JobBadgeTone =
  | "pending"
  | "setup"
  | "training"
  | "saving"
  | "evaluating"
  | "running"
  | "completed"
  | "failed";

function describeJobState(state: TrainingJob["state"]): string {
  switch (state) {
    case "queued":
      return "Queued";
    case "initializing":
      return "Initializing";
    case "preparing_dataset":
      return "Preparing data";
    case "training":
      return "Training";
    case "saving_artifact":
      return "Saving artifact";
    case "evaluating":
      return "Evaluating";
    case "running":
      return "Running";
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    default:
      return "Running";
  }
}

function jobBadgeTone(state: TrainingJob["state"]): JobBadgeTone {
  switch (state) {
    case "queued":
      return "pending";
    case "initializing":
    case "preparing_dataset":
      return "setup";
    case "training":
      return "training";
    case "saving_artifact":
      return "saving";
    case "evaluating":
      return "evaluating";
    case "running":
      return "running";
    case "completed":
      return "completed";
    case "failed":
      return "failed";
    default:
      return "running";
  }
}

function evaluationSourceLabel(source: TrainingJob["evaluation_source"]): string {
  return source === "training_dataset"
    ? "Training dataset (same config)"
    : "Legacy external file";
}

function JobBadge({ job }: { job: TrainingJob }) {
  const tone = jobBadgeTone(job.state);
  const label = describeJobState(job.state);
  return <span className={`jobBadge jobBadge-${tone}`}>{label}</span>;
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
  const [previewText, setPreviewText] = useState(
    "The quick brown fox jumps over the lazy dog."
  );
  const [previewResult, setPreviewResult] = useState<TokenizerPreviewResult | null>(
    null
  );
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [isDownloadingArtifact, setIsDownloadingArtifact] = useState(false);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [hasValidationPassed, setHasValidationPassed] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isLoadingTemplate, setIsLoadingTemplate] = useState(false);
  const [isUploadingTrainFile, setIsUploadingTrainFile] = useState(false);
  const [isDraggingTrainFiles, setIsDraggingTrainFiles] = useState(false);
  const [hasHydratedLocalState, setHasHydratedLocalState] = useState(false);
  const [highlightedSettingsCategory, setHighlightedSettingsCategory] =
    useState<SettingsCategory | null>(null);
  const [toasts, setToasts] = useState<ToastState[]>([]);
  const toastTimeoutsRef = useRef<Record<string, number>>({});
  const jobNotificationKeysRef = useRef<Set<string>>(new Set());
  const locallyStartedJobIdsRef = useRef<Set<string>>(new Set());
  const validationRequestRef = useRef(0);
  const previewRequestRef = useRef(0);
  const localFileDragDepthRef = useRef(0);
  const localTrainFileStatsPendingIdsRef = useRef<Set<string>>(new Set());
  const localTrainFileStatsFailedIdsRef = useRef<Set<string>>(new Set());
  const hasHydratedLocalStateRef = useRef(false);
  const settingsCategoryHighlightTimeoutRef = useRef<number | null>(null);
  const tokenizerAndTrainingPanelRef = useRef<HTMLDetailsElement | null>(null);
  const datasetPanelRef = useRef<HTMLDetailsElement | null>(null);
  const tokenizerCategoryRef = useRef<HTMLDivElement | null>(null);
  const datasetCategoryRef = useRef<HTMLDivElement | null>(null);
  const trainingCategoryRef = useRef<HTMLDivElement | null>(null);
  const controlsDisabled =
    isSubmitting ||
    isLoadingTemplate ||
    isUploadingTrainFile;

  useEffect(() => {
    if (!controlsDisabled) {
      return;
    }
    localFileDragDepthRef.current = 0;
    setIsDraggingTrainFiles(false);
  }, [controlsDisabled]);

  useEffect(() => {
    return () => {
      if (settingsCategoryHighlightTimeoutRef.current !== null) {
        window.clearTimeout(settingsCategoryHighlightTimeoutRef.current);
      }
    };
  }, []);

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
    const storedTheme = readStoredValue(THEME_STORAGE_KEY);
    if (storedTheme) {
      setThemeMode(asThemeMode(storedTheme));
    }

    const storedTokenizerForm = readStoredJson(TOKENIZER_FORM_STORAGE_KEY);
    if (storedTokenizerForm !== null) {
      setTokenizerForm((previous) =>
        hydrateTokenizerForm(storedTokenizerForm, previous)
      );
    }

    const storedDatasetForm = readStoredJson(DATASET_FORM_STORAGE_KEY);
    if (storedDatasetForm !== null) {
      setDatasetForm((previous) => hydrateDatasetForm(storedDatasetForm, previous));
    }

    const storedTrainingForm = readStoredJson(TRAINING_FORM_STORAGE_KEY);
    if (storedTrainingForm !== null) {
      setTrainingForm((previous) =>
        hydrateTrainingForm(storedTrainingForm, previous)
      );
    }

    const storedPreviewText = readStoredValue(PREVIEW_TEXT_STORAGE_KEY);
    if (storedPreviewText !== null) {
      setPreviewText((previous) => hydratePreviewText(storedPreviewText, previous));
    }

    const storedActiveJobId = readStoredValue(ACTIVE_JOB_STORAGE_KEY);
    if (storedActiveJobId !== null && storedActiveJobId.trim() !== "") {
      setActiveJobId(storedActiveJobId.trim());
    }

    hasHydratedLocalStateRef.current = true;
    setHasHydratedLocalState(true);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = themeMode;
    writeStoredValue(THEME_STORAGE_KEY, themeMode);
  }, [themeMode]);

  useEffect(() => {
    if (!hasHydratedLocalState) {
      return;
    }
    writeStoredJson(TOKENIZER_FORM_STORAGE_KEY, tokenizerForm);
  }, [hasHydratedLocalState, tokenizerForm]);

  useEffect(() => {
    if (!hasHydratedLocalState) {
      return;
    }
    writeStoredJson(DATASET_FORM_STORAGE_KEY, datasetForm);
  }, [datasetForm, hasHydratedLocalState]);

  useEffect(() => {
    if (!hasHydratedLocalState) {
      return;
    }
    writeStoredJson(TRAINING_FORM_STORAGE_KEY, trainingForm);
  }, [hasHydratedLocalState, trainingForm]);

  useEffect(() => {
    if (!hasHydratedLocalState) {
      return;
    }
    writeStoredValue(PREVIEW_TEXT_STORAGE_KEY, previewText);
  }, [hasHydratedLocalState, previewText]);

  useEffect(() => {
    if (!hasHydratedLocalState) {
      return;
    }

    const currentIds = new Set(datasetForm.localTrainFiles.map((entry) => entry.id));
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

    const entriesNeedingStats = datasetForm.localTrainFiles.filter((entry) => {
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

      setDatasetForm((previous) => {
        let changed = false;
        const nextLocalTrainFiles = previous.localTrainFiles.map((entry) => {
          const stats = updatesById.get(entry.id);
          if (!stats) {
            return entry;
          }
          if (entry.sizeBytes === stats.sizeBytes && entry.sizeChars === stats.sizeChars) {
            return entry;
          }
          changed = true;
          return {
            ...entry,
            sizeBytes: stats.sizeBytes,
            sizeChars: stats.sizeChars,
          };
        });

        if (!changed) {
          return previous;
        }

        return {
          ...previous,
          localTrainFiles: normalizeLocalTrainFiles(nextLocalTrainFiles),
        };
      });
    });

    return () => {
      cancelled = true;
    };
  }, [datasetForm.localTrainFiles, hasHydratedLocalState]);

  useEffect(() => {
    if (!hasHydratedLocalState) {
      return;
    }
    if (activeJobId === null) {
      removeStoredValue(ACTIVE_JOB_STORAGE_KEY);
      return;
    }
    writeStoredValue(ACTIVE_JOB_STORAGE_KEY, activeJobId);
  }, [activeJobId, hasHydratedLocalState]);

  const tokenizerBuild = useMemo(
    () => buildResult(() => buildTokenizerConfigFromForm(tokenizerForm)),
    [tokenizerForm]
  );

  const dataloaderBuild = useMemo(
    () => buildResult(() => buildDataloaderConfigFromForm(datasetForm, trainingForm)),
    [datasetForm, trainingForm]
  );
  const datasetBuild = useMemo(
    () => buildResult(() => buildDatasetSelectionFromForm(datasetForm)),
    [datasetForm]
  );
  const trainingRuntimeBuild = useMemo(
    () =>
      buildResult(() => ({
        budget_limit: parsePositiveInt(trainingForm.budgetLimit, "Text budget limit"),
        thresholds: parseThresholds(trainingForm.evaluationThresholds),
      })),
    [trainingForm.budgetLimit, trainingForm.evaluationThresholds]
  );

  useEffect(() => {
    setHasValidationPassed(false);
    setValidationError(null);
  }, [tokenizerForm, datasetForm, trainingForm]);

  const openSettingsCategory = useCallback((category: SettingsCategory) => {
    if (tokenizerAndTrainingPanelRef.current) {
      tokenizerAndTrainingPanelRef.current.open = true;
    }
    if (datasetPanelRef.current) {
      datasetPanelRef.current.open = true;
    }

    const targetRef =
      category === "tokenizer"
        ? tokenizerCategoryRef
        : category === "dataset"
          ? datasetCategoryRef
          : trainingCategoryRef;

    const hash = SETTINGS_CATEGORY_HASH_MAP[category];
    if (window.location.hash !== hash) {
      window.history.replaceState(null, "", hash);
    }

    targetRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
      inline: "nearest",
    });

    setHighlightedSettingsCategory(category);
    if (settingsCategoryHighlightTimeoutRef.current !== null) {
      window.clearTimeout(settingsCategoryHighlightTimeoutRef.current);
    }
    settingsCategoryHighlightTimeoutRef.current = window.setTimeout(() => {
      setHighlightedSettingsCategory((previous) =>
        previous === category ? null : previous
      );
    }, 1800);
  }, []);

  const handleSettingsCategoryNavigation = useCallback(
    (category: SettingsCategory) => {
      openSettingsCategory(category);
    },
    [openSettingsCategory]
  );

  useEffect(() => {
    const hash = window.location.hash;
    if (hash === SETTINGS_CATEGORY_HASH_MAP.tokenizer) {
      openSettingsCategory("tokenizer");
      return;
    }
    if (hash === SETTINGS_CATEGORY_HASH_MAP.dataset) {
      openSettingsCategory("dataset");
      return;
    }
    if (hash === SETTINGS_CATEGORY_HASH_MAP.training) {
      openSettingsCategory("training");
    }
  }, [openSettingsCategory]);

  useEffect(() => {
    if (!hasHydratedLocalStateRef.current) {
      return;
    }
    if (
      tokenizerBuild.error !== null ||
      dataloaderBuild.error !== null ||
      !tokenizerBuild.value ||
      !dataloaderBuild.value
    ) {
      setIsValidating(false);
      return;
    }

    const requestId = ++validationRequestRef.current;
    setIsValidating(true);
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          await Promise.all([
            validateTokenizerConfig(tokenizerBuild.value as Record<string, unknown>),
            validateDataloaderConfig(dataloaderBuild.value as Record<string, unknown>),
          ]);
          if (validationRequestRef.current !== requestId) {
            return;
          }
          setHasValidationPassed(true);
          setValidationError(null);
        } catch (error) {
          if (validationRequestRef.current !== requestId) {
            return;
          }
          const message =
            error instanceof Error ? error.message : "Validation failed unexpectedly";
          setHasValidationPassed(false);
          setValidationError(message);
        } finally {
          if (validationRequestRef.current === requestId) {
            setIsValidating(false);
          }
        }
      })();
    }, 350);

    return () => {
      window.clearTimeout(timer);
    };
  }, [
    dataloaderBuild.error,
    dataloaderBuild.value,
    tokenizerBuild.error,
    tokenizerBuild.value,
  ]);

  const handleManualValidate = useCallback(async () => {
    if (!tokenizerBuild.value) {
      const message = tokenizerBuild.error ?? "Tokenizer config is invalid";
      setHasValidationPassed(false);
      setValidationError(message);
      notify("error", message, 5000);
      return;
    }
    if (!dataloaderBuild.value) {
      const message = dataloaderBuild.error ?? "Dataloader config is invalid";
      setHasValidationPassed(false);
      setValidationError(message);
      notify("error", message, 5000);
      return;
    }

    const requestId = ++validationRequestRef.current;
    setIsValidating(true);

    try {
      await Promise.all([
        validateTokenizerConfig(tokenizerBuild.value),
        validateDataloaderConfig(dataloaderBuild.value),
      ]);
      if (validationRequestRef.current !== requestId) {
        return;
      }
      setHasValidationPassed(true);
      setValidationError(null);
      notify("success", "Validation passed.", 3500);
    } catch (error) {
      if (validationRequestRef.current !== requestId) {
        return;
      }
      const message =
        error instanceof Error ? error.message : "Validation failed unexpectedly";
      setHasValidationPassed(false);
      setValidationError(message);
      notify("error", message, 6000);
    } finally {
      if (validationRequestRef.current === requestId) {
        setIsValidating(false);
      }
    }
  }, [
    dataloaderBuild.error,
    dataloaderBuild.value,
    notify,
    tokenizerBuild.error,
    tokenizerBuild.value,
  ]);

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

  const removeLocalTrainFile = useCallback((localFileId: string) => {
    setDatasetForm((previous) => ({
      ...previous,
      localTrainFiles: previous.localTrainFiles.filter(
        (entry) => entry.id !== localFileId
      ),
    }));
  }, []);

  const clearLocalTrainFiles = useCallback(() => {
    setDatasetForm((previous) => ({
      ...previous,
      localTrainFiles: [],
    }));
  }, []);

  const runPreview = useCallback(async (jobId: string, text: string) => {
    const requestId = ++previewRequestRef.current;
    setIsPreviewing(true);
    setPreviewError(null);
    setPreviewResult((previous) =>
      previous && previous.job_id === jobId && previous.text === text ? previous : null
    );

    try {
      const result = await previewJobTokenizer(jobId, { text });
      if (previewRequestRef.current !== requestId) {
        return;
      }
      setPreviewResult(result);
    } catch (error) {
      if (previewRequestRef.current !== requestId) {
        return;
      }
      const message =
        error instanceof Error ? error.message : "Failed to preview tokenizer output";
      setPreviewResult(null);
      setPreviewError(message);
    } finally {
      if (previewRequestRef.current === requestId) {
        setIsPreviewing(false);
      }
    }
  }, []);

  const handleDownloadArtifact = useCallback(
    async (event: MouseEvent<HTMLAnchorElement>) => {
      event.preventDefault();

      if (!activeJob?.artifact_file || isDownloadingArtifact) {
        return;
      }

      const fileName = activeJob.artifact_file;
      setIsDownloadingArtifact(true);

      try {
        const blob = await downloadJobArtifact(activeJob.id);
        const tauriInvoke = getTauriInvoke();

        if (tauriInvoke) {
          try {
            const bytes = Array.from(new Uint8Array(await blob.arrayBuffer()));
            const result = await tauriInvoke("save_tokenizer_artifact", {
              file_name: fileName,
              bytes,
            });
            const savedPath =
              typeof result === "string" && result.trim() !== "" ? result : null;

            if (savedPath) {
              notify("success", `Saved ${fileName}`, 4500);
            }
            return;
          } catch {
            // Fall back to browser-style download if the desktop command is unavailable.
          }
        }

        triggerBlobDownload(blob, fileName);
        notify("success", `Downloaded ${fileName}`, 4500);
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Failed to download tokenizer";
        notify("error", message, 6500);
      } finally {
        setIsDownloadingArtifact(false);
      }
    },
    [activeJob, isDownloadingArtifact, notify]
  );

  const previewReadyJobId =
    activeJob && activeJob.status === "completed" && activeJob.artifact_file
      ? activeJob.id
      : null;

  const previewSegments = useMemo(() => {
    if (!previewResult) {
      return [];
    }
    return makePreviewSegments(previewResult.text, previewResult.tokens);
  }, [previewResult]);

  useEffect(() => {
    if (!previewReadyJobId) {
      setPreviewResult(null);
      setPreviewError(null);
      setIsPreviewing(false);
      return;
    }

    const timer = window.setTimeout(() => {
      void runPreview(previewReadyJobId, previewText);
    }, 280);

    return () => {
      window.clearTimeout(timer);
    };
  }, [previewReadyJobId, previewText, runPreview]);

  const refreshJobs = useCallback(async () => {
    if (!hasHydratedLocalStateRef.current) {
      return;
    }

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
    if (!hasHydratedLocalStateRef.current || !activeJobId) {
      return;
    }

    let cancelled = false;
    let isPolling = false;
    let hasCapturedInitialSnapshot = false;

    const poll = async () => {
      if (isPolling) {
        return;
      }
      isPolling = true;

      try {
        const job = await fetchTrainingJob(activeJobId);
        if (cancelled) {
          return;
        }

        const isInitialSnapshot = !hasCapturedInitialSnapshot;
        hasCapturedInitialSnapshot = true;
        const suppressTerminalToast =
          isInitialSnapshot && !locallyStartedJobIdsRef.current.has(job.id);

        setActiveJob(job);

        if (job.status === "completed") {
          const completedKey = `${job.id}:completed`;
          if (!jobNotificationKeysRef.current.has(completedKey)) {
            jobNotificationKeysRef.current.add(completedKey);
            if (!suppressTerminalToast) {
              notify(
                "success",
                `Training job ${job.id.slice(0, 8)} completed. Artifact is ready.`,
                6000
              );
            }
          }
          void refreshJobs();
        }

        if (job.status === "failed") {
          const failedKey = `${job.id}:failed`;
          if (!jobNotificationKeysRef.current.has(failedKey)) {
            jobNotificationKeysRef.current.add(failedKey);
            if (!suppressTerminalToast) {
              notify("error", job.error ?? "Training job failed", 7000);
            }
          }
          void refreshJobs();
        }
      } catch (error) {
        if (!cancelled) {
          const message =
            error instanceof Error ? error.message : "Failed to poll job status";
          notify("error", message, 7000);
        }
      } finally {
        isPolling = false;
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

  const uploadLocalTrainFiles = useCallback(
    async (selectedFiles: File[]) => {
      if (selectedFiles.length === 0) {
        return;
      }

      notify(
        "info",
        selectedFiles.length === 1
          ? `Uploading local train file: ${selectedFiles[0].name}`
          : `Uploading ${selectedFiles.length} local train files...`,
        2500
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
          setDatasetForm((previous) => ({
            ...previous,
            localTrainFiles: normalizeLocalTrainFiles([
              ...previous.localTrainFiles,
              ...successfulUploads.map((uploadedFile) =>
                makeLocalTrainFileEntry({
                  fileName: uploadedFile.file_name,
                  filePath: uploadedFile.file_path,
                  sizeBytes: uploadedFile.size_bytes,
                  sizeChars: uploadedFile.size_chars,
                })
              ),
            ]),
          }));
        }

        if (successfulUploads.length > 0) {
          notify(
            "success",
            successfulUploads.length === 1
              ? `Added ${stripGeneratedUploadPrefix(successfulUploads[0].file_name)}.`
              : `Added ${successfulUploads.length} local train files.`,
            4500
          );
        }

        const failedUploads = uploadResults.filter(
          (result): result is PromiseRejectedResult => result.status === "rejected"
        );
        if (failedUploads.length > 0) {
          const firstFailure = failedUploads[0];
          const firstFailureMessage =
            firstFailure?.reason instanceof Error
              ? firstFailure.reason.message
              : "Upload failed";
          notify(
            "error",
            `Failed to upload ${failedUploads.length} file(s). ${firstFailureMessage}`,
            8000
          );
        }
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Failed to upload local train files";
        notify("error", `Could not upload local train files. ${message}`, 7000);
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

  const handleLocalTrainFilesDragEnter = useCallback(
    (event: DragEvent<HTMLElement>) => {
      event.preventDefault();
      event.stopPropagation();
      if (controlsDisabled) {
        return;
      }
      if (!Array.from(event.dataTransfer.types).includes("Files")) {
        return;
      }
      localFileDragDepthRef.current += 1;
      setIsDraggingTrainFiles(true);
    },
    [controlsDisabled]
  );

  const handleLocalTrainFilesDragOver = useCallback(
    (event: DragEvent<HTMLElement>) => {
      event.preventDefault();
      event.stopPropagation();
      if (controlsDisabled) {
        return;
      }
      event.dataTransfer.dropEffect = "copy";
      setIsDraggingTrainFiles(true);
    },
    [controlsDisabled]
  );

  const handleLocalTrainFilesDragLeave = useCallback(
    (event: DragEvent<HTMLElement>) => {
      event.preventDefault();
      event.stopPropagation();
      if (controlsDisabled) {
        return;
      }
      localFileDragDepthRef.current = Math.max(0, localFileDragDepthRef.current - 1);
      if (localFileDragDepthRef.current === 0) {
        setIsDraggingTrainFiles(false);
      }
    },
    [controlsDisabled]
  );

  const handleLocalTrainFilesDrop = useCallback(
    async (event: DragEvent<HTMLElement>) => {
      event.preventDefault();
      event.stopPropagation();
      localFileDragDepthRef.current = 0;
      setIsDraggingTrainFiles(false);
      if (controlsDisabled) {
        return;
      }
      const droppedFiles = Array.from(event.dataTransfer.files ?? []);
      await uploadLocalTrainFiles(droppedFiles);
    },
    [controlsDisabled, uploadLocalTrainFiles]
  );

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

  const handleTrain = async () => {
    notify("info", "Submitting training job...", 2500);
    setIsSubmitting(true);

    try {
      if (!hasValidationPassed) {
        throw new Error("Automatic validation must pass before starting training");
      }
      if (!tokenizerBuild.value) {
        throw new Error(tokenizerBuild.error ?? "Tokenizer config is invalid");
      }
      if (!dataloaderBuild.value) {
        throw new Error(dataloaderBuild.error ?? "Dataloader config is invalid");
      }

      const thresholds = parseThresholds(trainingForm.evaluationThresholds);

      const job = await createTrainingJob({
        tokenizer_config: tokenizerBuild.value,
        dataloader_config: dataloaderBuild.value,
        evaluation_thresholds: thresholds,
      });

      locallyStartedJobIdsRef.current.add(job.id);
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

  const localTrainFileCount = datasetForm.localTrainFiles.length;
  const localTrainKnownCharCount = datasetForm.localTrainFiles.reduce(
    (sum, entry) => sum + (entry.sizeChars ?? 0),
    0
  );
  const localTrainKnownCharLabel =
    localTrainKnownCharCount > 0 ? formatCharCount(localTrainKnownCharCount) : null;
  const localTrainFilesHint =
    localTrainFileCount === 0
      ? "Add one or more local files, or switch to streaming datasets in settings."
      : datasetForm.sourceMode === "local_file"
        ? `${localTrainFileCount} file${localTrainFileCount === 1 ? "" : "s"} ready${
            localTrainKnownCharLabel ? ` (${localTrainKnownCharLabel} chars)` : ""
          }.`
        : `${localTrainFileCount} local file${
            localTrainFileCount === 1 ? "" : "s"
          } stored${
            localTrainKnownCharLabel ? ` (${localTrainKnownCharLabel} chars)` : ""
          }. Switch dataset source to Local files to use them.`;
  const tokenizerReady = tokenizerBuild.error === null;
  const datasetReady = datasetBuild.error === null;
  const trainingRuntimeReady = trainingRuntimeBuild.error === null;
  const preflightReady =
    tokenizerBuild.error === null &&
    dataloaderBuild.error === null &&
    trainingRuntimeBuild.error === null;
  const hasTrainingInProgress =
    activeJob !== null &&
    activeJob.status !== "completed" &&
    activeJob.status !== "failed";
  const trainingCompleted = activeJob?.status === "completed";
  const canStartTraining =
    hasValidationPassed &&
    !isValidating &&
    !controlsDisabled &&
    !hasTrainingInProgress;
  const activeThresholds =
    trainingRuntimeBuild.value && Array.isArray(trainingRuntimeBuild.value.thresholds)
      ? (trainingRuntimeBuild.value.thresholds as number[]).join(", ")
      : null;

  return (
    <main className="studioRoot">
      <header className="studioNav" role="navigation" aria-label="Primary">
        <div className="studioNavBrand">
          <span className="studioNavDot" aria-hidden="true" />
          <span>Tokenizer Studio</span>
        </div>
        <nav className="studioNavLinks" aria-label="Sections">
          <a className="studioNavLink" href="#workflow">
            Workflow
          </a>
          <a className="studioNavLink" href="#results">
            Results
          </a>
          <a className="studioNavLink" href="#settings">
            Settings
          </a>
        </nav>
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
          {themeMode === "dark" ? (
            <FiSun aria-hidden="true" />
          ) : (
            <FiMoon aria-hidden="true" />
          )}
        </button>
      </header>

      <section id="workflow" className="panelCard actionDeck">
        <div className="panelHead actionDeckHead">
          <div>
            <p className="panelEyebrow">Top Workflow</p>
            <h2>Steps to train the tokenizer</h2>
            <p className="panelCopy">
              Complete each step in order. A step turns green only when it is ready.
            </p>
          </div>
        </div>

        <div className="workflowStepGrid" role="list" aria-label="Tokenizer training steps">
          <article
            className={`workflowStepTile ${
              tokenizerReady ? "workflowStepTile-ready" : "workflowStepTile-waiting"
            }`}
            role="listitem"
          >
            <p className="workflowStepTitle">Step 1 - Choose tokenizer config</p>
            <strong>{tokenizerReady ? "Ready" : "Waiting for configuration"}</strong>
            <p className="fieldNote">
              {tokenizerBuild.error ??
                `${tokenizerForm.tokenizerType.toUpperCase()} tokenizer configured.`}
            </p>
            <button
              type="button"
              className="workflowStepLink workflowStepAction"
              onClick={() => handleSettingsCategoryNavigation("tokenizer")}
            >
              Open tokenizer settings
            </button>
          </article>

          <article
            className={`workflowStepTile ${
              datasetReady ? "workflowStepTile-ready" : "workflowStepTile-waiting"
            }`}
            role="listitem"
          >
            <p className="workflowStepTitle">Step 2 - Choose dataset</p>
            <strong>{datasetReady ? "Ready" : "Waiting for configuration"}</strong>
            <p className="fieldNote">
              {datasetBuild.error ??
                (datasetForm.sourceMode === "local_file"
                  ? localTrainFilesHint
                  : `${datasetForm.streamingDatasets.length} streaming dataset${
                      datasetForm.streamingDatasets.length === 1 ? "" : "s"
                    } configured.`)}
            </p>
            <button
              type="button"
              className="workflowStepLink workflowStepAction"
              onClick={() => handleSettingsCategoryNavigation("dataset")}
            >
              Open dataset settings
            </button>
          </article>

          <article
            className={`workflowStepTile ${
              trainingRuntimeReady
                ? "workflowStepTile-ready"
                : "workflowStepTile-waiting"
            }`}
            role="listitem"
          >
            <p className="workflowStepTitle">Step 3 - Configure training run</p>
            <strong>{trainingRuntimeReady ? "Ready" : "Waiting for configuration"}</strong>
            <p className="fieldNote">
              {trainingRuntimeBuild.error ??
                `Budget: ${trainingForm.budgetLimit} ${trainingForm.budgetUnit}, thresholds: ${activeThresholds}.`}
            </p>
            <button
              type="button"
              className="workflowStepLink workflowStepAction"
              onClick={() => handleSettingsCategoryNavigation("training")}
            >
              Open training settings
            </button>
          </article>

          <article
            className={`workflowStepTile ${
              hasValidationPassed
                ? "workflowStepTile-ready"
                : isValidating
                  ? "workflowStepTile-inProgress"
                  : "workflowStepTile-waiting"
            }`}
            role="listitem"
          >
            <p className="workflowStepTitle">Step 4 - Validate configs</p>
            <strong>
              {hasValidationPassed
                ? "Ready"
                : isValidating
                  ? "In progress"
                  : "Waiting for configuration"}
            </strong>
            <p className="fieldNote">
              {hasValidationPassed
                ? "Validation passed for tokenizer and dataloader configs."
                : isValidating
                  ? "Validating latest configuration changes automatically..."
                  : preflightReady
                    ? validationError ?? "Waiting for the next validation cycle."
                    : "Complete steps 1-3 first. Validation runs automatically."}
            </p>
            <button
              type="button"
              className="secondaryButton workflowStepAction workflowStepButtonCompact"
              onClick={handleManualValidate}
              disabled={controlsDisabled || !preflightReady || isValidating}
            >
              {isValidating ? "Validating..." : "Validate now"}
            </button>
          </article>

          <article
            className={`workflowStepTile ${
              trainingCompleted
                ? "workflowStepTile-ready"
                : hasTrainingInProgress
                  ? "workflowStepTile-inProgress"
                  : "workflowStepTile-waiting"
            }`}
            role="listitem"
          >
            <p className="workflowStepTitle">Step 5 - Start training</p>
            <strong>
              {trainingCompleted
                ? "Ready (trained)"
                : hasTrainingInProgress
                  ? "In progress"
                  : "Not ready"}
            </strong>
            <p className="fieldNote">
              {trainingCompleted
                ? "Latest training job completed. Artifact is ready."
                : hasTrainingInProgress
                  ? `Current job is ${describeJobState(activeJob?.state ?? "running").toLowerCase()}.`
                  : hasValidationPassed
                    ? "Validation passed. Start training to complete this step."
                    : isValidating
                      ? "Waiting for automatic validation to finish."
                      : "Automatic validation must pass to unlock training."}
            </p>
            <button
              type="button"
              className="primaryButton workflowStepAction"
              onClick={handleTrain}
              disabled={!canStartTraining}
            >
              {hasTrainingInProgress
                ? "Training..."
                : isSubmitting
                  ? "Starting..."
                  : "Start Training"}
            </button>
          </article>
        </div>
      </section>

      <section id="results" className="workspaceGrid">
        <article className="panelCard activeJobCard">
          <div className="sectionHeader">
            <h2>Active Job and Evaluation</h2>
            {activeJob ? <JobBadge job={activeJob} /> : null}
          </div>

          {activeJob ? (
            <>
              <div className="metaList">
                <div className="metaItem">
                  <span className="metaLabel">Job ID</span>
                  <span className="metaValue">{activeJob.id}</span>
                </div>
                <div className="metaItem">
                  <span className="metaLabel">Status</span>
                  <span className="metaValue">{describeJobState(activeJob.state)}</span>
                </div>
                <div className="metaItem">
                  <span className="metaLabel">Stage</span>
                  <span className="metaValue">{activeJob.stage}</span>
                </div>
                <div className="metaItem">
                  <span className="metaLabel">Progress</span>
                  <span className="metaValue">{formatPercent(activeJob.progress)}</span>
                </div>
                <div className="metaItem">
                  <span className="metaLabel">Evaluation</span>
                  <span className="metaValue">
                    {evaluationSourceLabel(activeJob.evaluation_source)}
                  </span>
                </div>
                <div className="metaItem">
                  <span className="metaLabel">Created</span>
                  <span className="metaValue">{formatDate(activeJob.created_at)}</span>
                </div>
                <div className="metaItem">
                  <span className="metaLabel">Started</span>
                  <span className="metaValue">{formatDate(activeJob.started_at)}</span>
                </div>
                <div className="metaItem">
                  <span className="metaLabel">Finished</span>
                  <span className="metaValue">{formatDate(activeJob.finished_at)}</span>
                </div>
              </div>

              <div className="progressTrack">
                <div
                  className="progressFill"
                  style={{
                    width: `${Math.max(0, Math.min(activeJob.progress * 100, 100))}%`,
                  }}
                />
              </div>

              {activeJob.status === "completed" && activeJob.stats ? (
                <>
                  <div className="statsGrid">
                    <div>
                      <span>Tokens / Char</span>
                      <strong>{activeJob.stats.token_per_char.toFixed(4)}</strong>
                    </div>
                    <div>
                      <span>Chars / Token</span>
                      <strong>{activeJob.stats.chars_per_token.toFixed(4)}</strong>
                    </div>
                    <div>
                      <span>Records evaluated</span>
                      <strong>{activeJob.stats.num_records}</strong>
                    </div>
                    <div>
                      <span>Avg chars / record</span>
                      <strong>{activeJob.stats.avg_chars_per_record.toFixed(2)}</strong>
                    </div>
                    <div>
                      <span>Avg tokens / record</span>
                      <strong>{activeJob.stats.avg_tokens_per_record.toFixed(2)}</strong>
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

                  <div className="rareTokenCard">
                    <h3>Rare Token Coverage</h3>
                    <div className="rareTokenTable">
                      {Object.entries(activeJob.stats.rare_tokens)
                        .sort(([left], [right]) => Number(left) - Number(right))
                        .map(([threshold, count]) => {
                          const fraction =
                            activeJob.stats?.rare_token_fraction?.[threshold] ?? 0;
                          return (
                            <div
                              key={`rare-threshold-${threshold}`}
                              className="rareTokenRow"
                            >
                              <span>{`< ${threshold}`}</span>
                              <strong>{count}</strong>
                              <span>{`${(fraction * 100).toFixed(2)}%`}</span>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                </>
              ) : null}

              {activeJob.status === "completed" && activeJob.artifact_file ? (
                <section className="tokenizerPreview">
                  <div className="sectionHeader tokenizerPreviewHeader">
                    <h3>Tokenizer Test Bench</h3>
                    <p className="metaLine">
                      {isPreviewing
                        ? "Tokenizing..."
                        : previewResult
                          ? `${previewResult.num_tokens} token${
                              previewResult.num_tokens === 1 ? "" : "s"
                            }`
                          : "-"}
                    </p>
                  </div>

                  <label className="fullWidthField">
                    Text to tokenize
                    <textarea
                      className="previewInput"
                      value={previewText}
                      onChange={(event) => setPreviewText(event.target.value)}
                      placeholder="Type text to preview tokenization..."
                      rows={4}
                    />
                  </label>

                  {previewError ? <p className="hint hint-error">{previewError}</p> : null}

                  {previewResult && previewResult.text.length > 0 ? (
                    <>
                      <div className="tokenizedText" aria-live="polite">
                        {previewSegments.map((segment, segmentIndex) => {
                          if (segment.kind === "plain") {
                            return (
                              <span key={`plain-${segmentIndex}`} className="plainSegment">
                                {segment.text}
                              </span>
                            );
                          }

                          const token = segment.token;
                          if (!token) {
                            return (
                              <span
                                key={`plain-fallback-${segmentIndex}`}
                                className="plainSegment"
                              >
                                {segment.text}
                              </span>
                            );
                          }

                          const hue = tokenHue(token.index);
                          return (
                            <span
                              key={`token-${token.index}-${token.start}-${token.end}`}
                              className="tokenMark"
                              title={`Token ID: ${token.id}`}
                              style={{
                                backgroundColor: `hsla(${hue}, 95%, 55%, 0.2)`,
                                borderColor: `hsla(${hue}, 70%, 55%, 0.7)`,
                              }}
                            >
                              {segment.text}
                            </span>
                          );
                        })}
                      </div>

                      <div className="tokenChipList" aria-label="Tokenizer output tokens">
                        {previewResult.tokens.map((token) => {
                          const hue = tokenHue(token.index);
                          return (
                            <span
                              key={`chip-${token.index}-${token.id}`}
                              className="tokenChip"
                              title={`Token ID: ${token.id}`}
                              style={{
                                backgroundColor: `hsla(${hue}, 95%, 55%, 0.17)`,
                                borderColor: `hsla(${hue}, 70%, 55%, 0.62)`,
                              }}
                            >
                              <code>{displayTokenLabel(token.token)}</code>
                            </span>
                          );
                        })}
                      </div>
                    </>
                  ) : previewText.trim() === "" ? (
                    <p className="metaLine">Enter text above to preview tokenization.</p>
                  ) : null}
                </section>
              ) : null}

              {activeJob.artifact_file ? (
                <a
                  className="downloadLink"
                  href={artifactDownloadUrl(activeJob.id)}
                  download={activeJob.artifact_file}
                  onClick={(event) => void handleDownloadArtifact(event)}
                  aria-disabled={isDownloadingArtifact}
                  rel="noreferrer"
                >
                  {isDownloadingArtifact
                    ? `Downloading ${activeJob.artifact_file}...`
                    : `Download ${activeJob.artifact_file}`}
                </a>
              ) : null}

              {activeJob.error ? <p className="hint hint-error">{activeJob.error}</p> : null}
            </>
          ) : (
            <p className="metaLine">No job selected yet.</p>
          )}
        </article>

        <article className="panelCard recentJobsCard">
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
                    <p>{job.stage}</p>
                  </div>
                  <div className="jobRowMeta">
                    <JobBadge job={job} />
                    {job.status === "completed" || job.status === "failed" ? null : (
                      <p>{formatPercent(job.progress)}</p>
                    )}
                  </div>
                </button>
              ))}
            </div>
          )}
        </article>
      </section>

      <section id="settings" className="panelCard settingsStudio">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">Settings</p>
            <h2>Configuration Studio</h2>
            <p className="panelCopy">
              Core fields stay visible. Advanced options are collapsed so most
              users can train and test quickly.
            </p>
          </div>
        </div>

        <details className="settingsPanel" open ref={tokenizerAndTrainingPanelRef}>
          <summary>Tokenizer and training budget</summary>
          <div className="settingsGrid">
            <div
              id="settings-tokenizer"
              ref={tokenizerCategoryRef}
              className={`settingsGroup settingsCategoryAnchor ${
                highlightedSettingsCategory === "tokenizer"
                  ? "settingsCategoryAnchor-highlight"
                  : ""
              }`}
            >
              <div className="settingsGroupHeader">
                <h3>Tokenizer core</h3>
                <p className="settingsGroupHint">
                  Set tokenizer identity, model type, and vocabulary behavior.
                </p>
              </div>

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

                <label className="fullWidthField">
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
              </div>

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

              <p className={`hint ${tokenizerBuild.error ? "hint-error" : "hint-ok"}`}>
                {tokenizerBuild.error ?? "Tokenizer config looks valid."}
              </p>
            </div>

            <div
              id="settings-training"
              ref={trainingCategoryRef}
              className={`settingsGroup settingsCategoryAnchor ${
                highlightedSettingsCategory === "training"
                  ? "settingsCategoryAnchor-highlight"
                  : ""
              }`}
            >
              <div className="settingsGroupHeader">
                <h3>Training budget</h3>
                <p className="settingsGroupHint">
                  Control how much dataset text is consumed during tokenizer training.
                </p>
              </div>
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
            </div>
          </div>
        </details>

        <details className="settingsPanel" open ref={datasetPanelRef}>
          <summary>Core dataset settings</summary>
          <div className="settingsGrid">
            <div
              id="settings-dataset"
              ref={datasetCategoryRef}
              className={`settingsCategoryAnchor ${
                highlightedSettingsCategory === "dataset"
                  ? "settingsCategoryAnchor-highlight"
                  : ""
              }`}
            >
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
                  Local files
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

            {datasetForm.sourceMode === "local_file" ? (
              <div className="datasetConfigurator">
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
                            controlsDisabled ? "localFileUploadButton-disabled" : ""
                          }`}
                          aria-disabled={controlsDisabled}
                        >
                          {isUploadingTrainFile ? "Uploading..." : "Add files"}
                          <input
                            type="file"
                            multiple
                            onChange={handleTrainFilesSelected}
                            disabled={controlsDisabled}
                          />
                        </label>
                        <button
                          type="button"
                          className="textButton localFileHeaderButton"
                          onClick={clearLocalTrainFiles}
                          disabled={controlsDisabled || localTrainFileCount === 0}
                        >
                          Remove all
                        </button>
                      </div>
                      <span className="localFileCount">
                        {localTrainFileCount} file{localTrainFileCount === 1 ? "" : "s"}
                      </span>
                    </div>
                  </div>

                  {localTrainFileCount === 0 ? (
                    <p className="filterEmpty">
                      No local files added yet. Add one or more files to train.
                    </p>
                  ) : (
                    <ul className="localFileList">
                      {datasetForm.localTrainFiles.map((entry) => {
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
                                disabled={controlsDisabled}
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

                  <span className="fieldNote">
                    Files are deduplicated by stored path.
                  </span>
                </div>
              </div>
            ) : (
              <div className="datasetConfigurator">
                <label className="fullWidthField">
                  HF access token (optional)
                  <input
                    type="password"
                    value={datasetForm.hfToken}
                    onChange={(event) =>
                      setDatasetForm((previous) => ({
                        ...previous,
                        hfToken: event.target.value,
                      }))
                    }
                    disabled={controlsDisabled}
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
                          disabled={
                            controlsDisabled || datasetForm.streamingDatasets.length <= 1
                          }
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
                      </div>

                      <details className="subPanel">
                        <summary>Advanced source options</summary>
                        <div className="fieldGrid">
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
                                          filter.operator === "in" ||
                                          filter.operator === "not in"
                                            ? '["en", "de"] or en,de'
                                            : "en, true, 0.95, {\"k\":1}"
                                        }
                                      />
                                    </label>

                                    <button
                                      type="button"
                                      className="textButton filterRemoveButton"
                                      onClick={() =>
                                        removeStreamingFilter(entry.id, filter.id)
                                      }
                                      disabled={controlsDisabled}
                                    >
                                      Remove
                                    </button>
                                  </div>
                                ))}
                              </div>
                            )}
                            <p className="fieldNote">
                              Values are inferred automatically. For `in`/`not in`, use
                              a JSON array or comma-separated values.
                            </p>
                          </div>
                        </div>
                      </details>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <p className={`hint ${dataloaderBuild.error ? "hint-error" : "hint-ok"}`}>
              {dataloaderBuild.error ?? "Dataloader config looks valid."}
            </p>
            </div>
          </div>
        </details>

        <details className="settingsPanel">
          <summary>Advanced tokenizer settings</summary>
          <div className="settingsGrid">
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
          </div>
        </details>

        <details className="settingsPanel">
          <summary>Generated config JSON</summary>
          <div className="settingsGrid">
            <div className="jsonGrid">
              <label>
                Tokenizer JSON
                <pre>
                  {tokenizerBuild.value
                    ? prettyJson(tokenizerBuild.value)
                    : "Invalid config"}
                </pre>
              </label>

              <label>
                Dataloader JSON
                <pre>
                  {dataloaderBuild.value
                    ? prettyJson(dataloaderBuild.value)
                    : "Invalid config"}
                </pre>
              </label>
            </div>
          </div>
        </details>
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
