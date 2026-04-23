import {
  FILTER_OPERATORS,
  WEIGHT_SCALE,
  WEIGHT_SUM_EPSILON,
} from "../constants";
import type {
  DatasetSourceMode,
  FilterOperator,
  LocalTrainFileFormState,
  StreamingDatasetFormState,
  StreamingFilterFormState,
} from "../types";
import {
  asRecord,
  asRecordArray,
  asString,
} from "./object";
import {
  fileNameFromPath,
  stripGeneratedUploadPrefix,
} from "./files";

export function makeStreamingFilterEntry(
  value?: Partial<StreamingFilterFormState>
): StreamingFilterFormState {
  return {
    id: value?.id ?? `filter-${Math.random().toString(36).slice(2, 10)}`,
    column: value?.column ?? "",
    operator: value?.operator ?? "==",
    value: value?.value ?? "",
  };
}

export function makeStreamingDatasetEntry(
  value?: Partial<StreamingDatasetFormState>
): StreamingDatasetFormState {
  return {
    id: value?.id ?? `dataset-${Math.random().toString(36).slice(2, 10)}`,
    name: value?.name ?? "",
    config: value?.config ?? "",
    split: value?.split ?? "train",
    textColumns: value?.textColumns ?? "text",
    weight: value?.weight ?? "1",
    filters: (value?.filters ?? []).map((filter) => makeStreamingFilterEntry(filter)),
  };
}

export function makeLocalTrainFileEntry(
  value?: Partial<LocalTrainFileFormState>
): LocalTrainFileFormState {
  const filePath = value?.filePath ?? "";
  const fallbackFileName = stripGeneratedUploadPrefix(fileNameFromPath(filePath));
  const providedFileName = stripGeneratedUploadPrefix(value?.fileName?.trim() ?? "");
  return {
    id: value?.id ?? `local-file-${Math.random().toString(36).slice(2, 10)}`,
    filePath,
    fileName: providedFileName || fallbackFileName || "Training file",
    sizeBytes:
      typeof value?.sizeBytes === "number" && Number.isFinite(value.sizeBytes) && value.sizeBytes >= 0
        ? value.sizeBytes
        : null,
    sizeChars:
      typeof value?.sizeChars === "number" && Number.isFinite(value.sizeChars) && value.sizeChars >= 0
        ? value.sizeChars
        : null,
  };
}

export function normalizeLocalTrainFiles(
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

export function parseWeightInput(value: string): number | null {
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

export function sanitizeWeightInput(value: string): string {
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
  const safeWeights = weights.map((weight) => (Number.isFinite(weight) ? Math.max(0, weight) : 0));
  const normalized = [...safeWeights];
  const lockedWeight = clamp(safeWeights[lockedIndex] ?? 0, 0, 1);
  normalized[lockedIndex] = lockedWeight;
  const remaining = 1 - lockedWeight;
  const otherIndexes = safeWeights
    .map((_, index) => index)
    .filter((index) => index !== lockedIndex);
  if (otherIndexes.length === 0) {
    return normalized;
  }
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

export function normalizeStreamingDatasetWeights(
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

export function hydrateDatasetUiFromConfig(config: Record<string, unknown> | null): {
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

export function buildDatasetsFromUi(
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
