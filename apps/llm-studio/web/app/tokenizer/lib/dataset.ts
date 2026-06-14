import {
  sanitizePositiveDecimalInput,
  sanitizePositiveIntegerInput,
} from "../../shared/lib/configNumber";
import {
  CURRENT_TOKENIZER_API_SEGMENT,
  DEFAULT_SHAKE_DATASET_PATH,
  FILTER_OPERATORS,
  LEGACY_TOKENIZER_API_SEGMENT,
  LEGACY_UPLOADS_SEGMENT,
  MIN_STRICT_POSITIVE_WEIGHT,
  WEIGHT_SCALE,
  WEIGHT_SUM_EPSILON,
} from "../constants";
import type {
  BudgetBehavior,
  BudgetUnit,
  DatasetFormState,
  DatasetSourceMode,
  DecoderType,
  FilterOperator,
  LocalTrainFileFormState,
  PreTokenizerType,
  StreamingDatasetFormState,
  StreamingFilterFormState,
  TokenizerFormState,
  TokenizerType,
  TrainingFormState,
} from "../types";

export function asRecord(value: unknown): Record<string, unknown> {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

export function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

export function asTokenizerType(value: unknown): TokenizerType {
  if (value === "bpe" || value === "wordpiece" || value === "unigram") {
    return value;
  }
  return "bpe";
}

export function asPreTokenizerType(value: unknown): PreTokenizerType {
  if (value === "byte_level" || value === "whitespace" || value === "metaspace") {
    return value;
  }
  return "byte_level";
}

export function asDecoderType(value: unknown): DecoderType {
  if (value === "byte_level" || value === "wordpiece" || value === "metaspace") {
    return value;
  }
  return "byte_level";
}

export function asBudgetUnit(value: unknown): BudgetUnit {
  if (value === "chars" || value === "bytes") {
    return value;
  }
  return "chars";
}

export function asBudgetBehavior(value: unknown): BudgetBehavior {
  if (value === "stop" || value === "truncate") {
    return value;
  }
  return "truncate";
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
    filters: value?.filters ?? [],
  };
}

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

export function stripGeneratedUploadPrefix(value: string): string {
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

export function fileNameFromPath(value: string): string {
  const normalized = value.replaceAll("\\", "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] ?? "";
}

export function makeLocalTrainFileEntry(
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

export function hydrateStreamingFilter(value: unknown): StreamingFilterFormState {
  const record = asRecord(value);
  const storedId = asString(record.id, "").trim();

  return makeStreamingFilterEntry({
    id: storedId === "" ? undefined : storedId,
    column: asString(record.column, ""),
    operator: asFilterOperator(record.operator),
    value: asString(record.value, ""),
  });
}

export function hydrateStreamingDataset(value: unknown): StreamingDatasetFormState {
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

export function migrateLegacyLocalTrainFilePath(value: string): string {
  const trimmed = value.trim();
  if (trimmed === "") {
    return "";
  }

  const normalized = trimmed.replaceAll("\\", "/");
  if (!normalized.includes(LEGACY_TOKENIZER_API_SEGMENT)) {
    return trimmed;
  }

  if (normalized.includes(LEGACY_UPLOADS_SEGMENT)) {
    const normalizedFileName = stripGeneratedUploadPrefix(fileNameFromPath(normalized)).toLowerCase();
    if (normalizedFileName === "shake.txt") {
      return DEFAULT_SHAKE_DATASET_PATH;
    }
    return trimmed;
  }

  const datasetsMarkerIndex = normalized.indexOf("/datasets/");
  if (datasetsMarkerIndex >= 0) {
    return normalized.slice(datasetsMarkerIndex + 1);
  }

  return normalized.replace(LEGACY_TOKENIZER_API_SEGMENT, CURRENT_TOKENIZER_API_SEGMENT);
}

export function hydrateLocalTrainFile(value: unknown): LocalTrainFileFormState | null {
  const record = asRecord(value);
  const filePath = migrateLegacyLocalTrainFilePath(asString(record.filePath, ""));
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

export function normalizeLocalTrainFiles(
  files: LocalTrainFileFormState[]
): LocalTrainFileFormState[] {
  const dedupedByPath = new Map<string, LocalTrainFileFormState>();
  files.forEach((entry) => {
    const filePath = migrateLegacyLocalTrainFilePath(entry.filePath).trim();
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

export function resolveLocalTrainFilePaths(files: LocalTrainFileFormState[]): string[] {
  return normalizeLocalTrainFiles(files)
    .map((entry) => migrateLegacyLocalTrainFilePath(entry.filePath).trim())
    .filter((entry) => entry !== "");
}

export function extractTrainFilePaths(dataFiles: unknown): string[] {
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

export function hydrateTokenizerForm(
  value: unknown,
  fallback: TokenizerFormState
): TokenizerFormState {
  const record = asRecord(value);

  return {
    name: asString(record.name, fallback.name),
    tokenizerType: asTokenizerType(record.tokenizerType),
    vocabSize: sanitizePositiveIntegerInput(asString(record.vocabSize, fallback.vocabSize)),
    minFrequency: sanitizePositiveIntegerInput(
      asString(record.minFrequency, fallback.minFrequency)
    ),
    specialTokens: asString(record.specialTokens, fallback.specialTokens),
    byteFallback:
      typeof record.byteFallback === "boolean" ? record.byteFallback : fallback.byteFallback,
    unkToken: asString(record.unkToken, fallback.unkToken),
    preTokenizer: asPreTokenizerType(record.preTokenizer),
    decoder: asDecoderType(record.decoder),
  };
}

export function hydrateDatasetForm(value: unknown, fallback: DatasetFormState): DatasetFormState {
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
  const hadStoredLocalTrainFileEntries = localTrainFilesRaw.length > 0;
  const localTrainFilesFromStorage = normalizeLocalTrainFiles(
    localTrainFilesRaw
      .map((entry: unknown) => hydrateLocalTrainFile(entry))
      .filter((entry: LocalTrainFileFormState | null): entry is LocalTrainFileFormState => entry !== null)
  );
  const legacyTrainFilePath = migrateLegacyLocalTrainFilePath(asString(record.trainFilePath, ""));
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
        ? hadStoredLocalTrainFileEntries
          ? fallback.localTrainFiles.map((entry) => makeLocalTrainFileEntry(entry))
          : []
        : legacyLocalTrainFile.length > 0
          ? legacyLocalTrainFile
          : fallback.localTrainFiles.map((entry) => makeLocalTrainFileEntry(entry));

  return {
    sourceMode,
    localTrainFiles: normalizeLocalTrainFiles(localTrainFiles),
    hfToken: "",
    streamingDatasets: normalizeStreamingDatasetWeights(
      streamingDatasets.length > 0 ? streamingDatasets : [makeStreamingDatasetEntry()]
    ),
  };
}

export function hydrateTrainingForm(
  value: unknown,
  fallback: TrainingFormState
): TrainingFormState {
  const record = asRecord(value);

  return {
    budgetLimit: sanitizePositiveIntegerInput(asString(record.budgetLimit, fallback.budgetLimit)),
    budgetUnit: asBudgetUnit(record.budgetUnit),
    budgetBehavior: asBudgetBehavior(record.budgetBehavior),
    evaluationThresholds: sanitizeThresholdsInput(
      asString(record.evaluationThresholds, fallback.evaluationThresholds)
    ),
  };
}

export function splitTokens(value: string): string[] {
  return value
    .split(/[\n,]/)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
}

export function formatCharCount(value: number | null): string | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return null;
  }
  return new Intl.NumberFormat().format(Math.trunc(value));
}

export function asFilterOperator(value: unknown): FilterOperator {
  return FILTER_OPERATORS.includes(value as FilterOperator) ? (value as FilterOperator) : "==";
}

export function stringifyFilterValue(value: unknown): string {
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

export function parseInteger(raw: string, label: string, min?: number): number {
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

export function parsePositiveInt(raw: string, label: string): number {
  return parseInteger(raw, label, 1);
}

export function parseThresholds(raw: string): number[] {
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

export function parseNonNegativeNumber(raw: string, label: string): number {
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

export function sanitizeThresholdsInput(value: string): string {
  return value.replace(/[^\d,\s]/g, "");
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

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function formatWeight(value: number): string {
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

export function normalizeWeights(weights: number[]): number[] {
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

export function normalizeWeightsWithLockedIndex(
  weights: number[],
  lockedIndex: number
): number[] {
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
  const shouldLock = lockedIndex >= 0 && lockedRawParsed !== null && lockedRawParsed <= 1;
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
          (bestIndex, weight, index) => (weight > roundedWeights[bestIndex] ? index : bestIndex),
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

export function parseFilterValue(value: string, operator: FilterOperator, label: string): unknown {
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

export function buildFiltersFromForm(
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

export function buildDatasetSelectionFromForm(dataset: DatasetFormState): Record<string, unknown> {
  if (dataset.sourceMode === "local_file") {
    const localTrainPaths = resolveLocalTrainFilePaths(dataset.localTrainFiles);
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

export function tokenizerFormFromConfig(config: Record<string, unknown>): TokenizerFormState {
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
    byteFallback: typeof config.byte_fallback === "boolean" ? config.byte_fallback : true,
    unkToken: String(config.unk_token ?? "[UNK]"),
    preTokenizer: asPreTokenizerType(config.pre_tokenizer),
    decoder: asDecoderType(config.decoder),
  };
}

export function datasetFormFromConfig(config: Record<string, unknown>): DatasetFormState {
  const datasetsRaw = Array.isArray(config.datasets) ? config.datasets : [];
  const firstDataset = asRecord(datasetsRaw[0]);
  const localTrainPaths = extractTrainFilePaths(firstDataset.data_files);
  const sourceMode: DatasetSourceMode =
    datasetsRaw.length === 1 && localTrainPaths.length > 0 ? "local_file" : "streaming_hf";
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
    hfToken: "",
    sourceMode,
    streamingDatasets,
  };
}

export function trainingFormFromConfig(config: Record<string, unknown>): TrainingFormState {
  const budget = asRecord(config.budget);
  return {
    budgetLimit: String(budget.limit ?? 250000),
    budgetUnit: asBudgetUnit(budget.unit),
    budgetBehavior: asBudgetBehavior(budget.behavior),
    evaluationThresholds: "5,10,25",
  };
}

export function buildTokenizerConfigFromForm(form: TokenizerFormState): Record<string, unknown> {
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

export function buildDataloaderConfigFromForm(
  dataset: DatasetFormState,
  training: TrainingFormState
): Record<string, unknown> {
  let datasets: Record<string, unknown>[];

  if (dataset.sourceMode === "local_file") {
    const localTrainPaths = resolveLocalTrainFilePaths(dataset.localTrainFiles);
    if (localTrainPaths.length === 0) {
      throw new Error("Add at least one local train file");
    }
    const trainDataFiles = localTrainPaths.length === 1 ? localTrainPaths[0] : localTrainPaths;

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

    const totalWeight = streamingDatasetConfigs.reduce((sum, entry) => sum + entry.parsedWeight, 0);

    if (Math.abs(totalWeight - 1) > WEIGHT_SUM_EPSILON) {
      throw new Error(
        `Streaming dataset weights must sum to exactly 1. Current total: ${Number(
          totalWeight.toFixed(6)
        )}`
      );
    }

    const strictlyPositiveWeights = normalizeWeights(
      streamingDatasetConfigs.map((entry) =>
        entry.parsedWeight <= WEIGHT_SUM_EPSILON ? MIN_STRICT_POSITIVE_WEIGHT : entry.parsedWeight
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
