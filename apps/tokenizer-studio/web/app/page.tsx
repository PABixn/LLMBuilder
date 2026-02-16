"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  apiBaseUrl,
  artifactDownloadUrl,
  createTrainingJob,
  fetchConfigTemplates,
  fetchTrainingJob,
  fetchTrainingJobs,
  type TrainingJob,
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

interface StreamingDatasetFormState {
  id: string;
  name: string;
  config: string;
  split: string;
  textColumns: string;
  weight: string;
  filters: string;
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
  streamingDatasets: StreamingDatasetFormState[];
}

interface TrainingFormState {
  budgetLimit: string;
  budgetUnit: BudgetUnit;
  budgetBehavior: BudgetBehavior;
  evaluationThresholds: string;
  evaluationTextPath: string;
}

interface BuildResult {
  value: Record<string, unknown> | null;
  error: string | null;
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
    filters: value?.filters ?? "",
  };
}

function splitTokens(value: string): string[] {
  return value
    .split(/[\n,]/)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
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

function parsePositiveNumber(raw: string, label: string): number {
  const trimmed = raw.trim();
  if (trimmed === "") {
    throw new Error(`${label} is required`);
  }
  const value = Number(trimmed);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a positive number`);
  }
  return value;
}

function parseFilters(raw: string, label: string): unknown[][] | undefined {
  const trimmed = raw.trim();
  if (trimmed === "") {
    return undefined;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    throw new Error(`${label} must be valid JSON`);
  }

  if (!Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON array`);
  }

  return parsed.map((entry, index) => {
    if (!Array.isArray(entry) || entry.length !== 3) {
      throw new Error(`${label} item ${index + 1} must be [column, op, value]`);
    }
    return [entry[0], entry[1], entry[2]];
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
            const datasetFilters = Array.isArray(datasetRecord.filters)
              ? JSON.stringify(datasetRecord.filters, null, 2)
              : "";
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
    evaluationTextPath: "datasets/shake.txt",
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
    if (trainFilePath !== "") {
      datasetConfig.data_files = { train: trainFilePath };
    }

    datasets = [datasetConfig];
  } else {
    if (dataset.streamingDatasets.length === 0) {
      throw new Error("At least one streaming dataset is required");
    }

    datasets = dataset.streamingDatasets.map((entry, index) => {
      const datasetName = entry.name.trim();
      if (datasetName === "") {
        throw new Error(`Streaming dataset ${index + 1}: dataset name is required`);
      }

      const textColumns = splitTokens(entry.textColumns);
      if (textColumns.length === 0) {
        throw new Error(`Streaming dataset ${index + 1}: text columns are required`);
      }

      const datasetConfig: Record<string, unknown> = {
        name: datasetName,
        split: entry.split.trim() || "train",
        text_columns: textColumns,
        weight: parsePositiveNumber(
          entry.weight,
          `Streaming dataset ${index + 1}: weight`
        ),
      };

      const datasetConfigName = entry.config.trim();
      if (datasetConfigName !== "") {
        datasetConfig.config = datasetConfigName;
      }

      const parsedFilters = parseFilters(
        entry.filters,
        `Streaming dataset ${index + 1}: filters`
      );
      if (parsedFilters) {
        datasetConfig.filters = parsedFilters;
      }

      return datasetConfig;
    });
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

  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<TrainingJob | null>(null);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [isLoadingTemplate, setIsLoadingTemplate] = useState(false);
  const [statusMessage, setStatusMessage] = useState(
    "Ready. Fill the quick setup and run training."
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

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

  const addStreamingDataset = useCallback(() => {
    setDatasetForm((previous) => ({
      ...previous,
      streamingDatasets: [...previous.streamingDatasets, makeStreamingDatasetEntry()],
    }));
  }, []);

  const removeStreamingDataset = useCallback((datasetId: string) => {
    setDatasetForm((previous) => {
      const nextDatasets = previous.streamingDatasets.filter(
        (entry) => entry.id !== datasetId
      );
      return {
        ...previous,
        streamingDatasets:
          nextDatasets.length > 0 ? nextDatasets : [makeStreamingDatasetEntry()],
      };
    });
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
          setStatusMessage(
            `Training job ${job.id.slice(0, 8)} completed. Artifact is ready.`
          );
          void refreshJobs();
        }

        if (job.status === "failed") {
          setErrorMessage(job.error ?? "Training job failed");
          void refreshJobs();
        }
      } catch (error) {
        if (!cancelled) {
          const message =
            error instanceof Error ? error.message : "Failed to poll job status";
          setErrorMessage(message);
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
  }, [activeJobId, refreshJobs]);

  const handleLoadStreamingTemplate = async () => {
    setErrorMessage(null);
    setStatusMessage("Loading streaming dataset template...");
    setIsLoadingTemplate(true);

    try {
      const templates = await fetchConfigTemplates();
      const templateConfig = asRecord(templates.dataloader_config_template);
      const templateDatasetForm = datasetFormFromConfig(templateConfig);
      const templateTrainingForm = trainingFormFromConfig(templateConfig);

      setDatasetForm((previous) => ({
        ...previous,
        sourceMode: "streaming_hf",
        streamingDatasets: templateDatasetForm.streamingDatasets,
      }));
      setTrainingForm((previous) => ({
        ...previous,
        budgetLimit: templateTrainingForm.budgetLimit,
        budgetUnit: templateTrainingForm.budgetUnit,
        budgetBehavior: templateTrainingForm.budgetBehavior,
      }));
      setStatusMessage("Loaded streaming template datasets.");
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Failed to load streaming template";
      setErrorMessage(message);
      setStatusMessage("Could not load streaming template.");
    } finally {
      setIsLoadingTemplate(false);
    }
  };

  const handleValidate = async () => {
    setErrorMessage(null);
    setStatusMessage("Validating configs with API...");
    setIsValidating(true);

    try {
      if (!tokenizerBuild.value) {
        throw new Error(tokenizerBuild.error ?? "Tokenizer config is invalid");
      }
      if (!dataloaderBuild.value) {
        throw new Error(dataloaderBuild.error ?? "Dataloader config is invalid");
      }

      await Promise.all([
        validateTokenizerConfig(tokenizerBuild.value),
        validateDataloaderConfig(dataloaderBuild.value),
      ]);

      setStatusMessage("Validation passed. You can start training.");
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Validation failed unexpectedly";
      setErrorMessage(message);
      setStatusMessage("Validation failed.");
    } finally {
      setIsValidating(false);
    }
  };

  const handleTrain = async () => {
    setErrorMessage(null);
    setStatusMessage("Submitting training job...");
    setIsSubmitting(true);

    try {
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
        evaluation_text_path: trainingForm.evaluationTextPath.trim(),
      });

      setActiveJobId(job.id);
      setActiveJob(job);
      setStatusMessage(`Training job ${job.id.slice(0, 8)} started.`);
      await refreshJobs();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to create training job";
      setErrorMessage(message);
      setStatusMessage("Could not start training.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="studioPage">
      <header className="heroCard">
        <p className="heroTag">Tokenizer Studio</p>
        <h1>Train a tokenizer in 3 steps</h1>
        <p>
          Set basic tokenizer fields, point to your dataset, validate once, and run.
          The interface keeps advanced options hidden by default.
        </p>
        <div className="heroMeta">API: {apiBaseUrl()}</div>
      </header>

      <section className="messages">
        <div className="message message-info">{statusMessage}</div>
        {errorMessage ? <div className="message message-error">{errorMessage}</div> : null}
      </section>

      <section className="card sectionBlock">
        <h2>1. Quick Setup</h2>
        <p className="metaLine">
          Only required fields are shown. Open advanced sections only when needed.
        </p>
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
                  value={tokenizerForm.vocabSize}
                  onChange={(event) =>
                    setTokenizerForm((previous) => ({
                      ...previous,
                      vocabSize: event.target.value,
                    }))
                  }
                />
              </label>

              <label>
                Min frequency
                <input
                  value={tokenizerForm.minFrequency}
                  onChange={(event) =>
                    setTokenizerForm((previous) => ({
                      ...previous,
                      minFrequency: event.target.value,
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
                  onClick={() =>
                    setDatasetForm((previous) => ({
                      ...previous,
                      sourceMode: "streaming_hf",
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

                <label className="fullWidthField">
                  Local train file path (optional)
                  <input
                    value={datasetForm.trainFilePath}
                    onChange={(event) =>
                      setDatasetForm((previous) => ({
                        ...previous,
                        trainFilePath: event.target.value,
                      }))
                    }
                    placeholder="datasets/shake.txt"
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
                    disabled={isSubmitting || isValidating || isLoadingTemplate}
                  >
                    Add dataset
                  </button>
                  <button
                    type="button"
                    className="secondaryButton"
                    onClick={handleLoadStreamingTemplate}
                    disabled={isSubmitting || isValidating || isLoadingTemplate}
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
                          disabled={datasetForm.streamingDatasets.length <= 1}
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
                            value={entry.weight}
                            onChange={(event) =>
                              updateStreamingDataset(entry.id, {
                                weight: event.target.value,
                              })
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

                        <label className="fullWidthField">
                          Filters JSON (optional)
                          <textarea
                            rows={2}
                            value={entry.filters}
                            onChange={(event) =>
                              updateStreamingDataset(entry.id, {
                                filters: event.target.value,
                              })
                            }
                            placeholder='[["language_score", ">", 0.95]]'
                          />
                        </label>
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
                  value={trainingForm.budgetLimit}
                  onChange={(event) =>
                    setTrainingForm((previous) => ({
                      ...previous,
                      budgetLimit: event.target.value,
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
      </section>

      <section className="card sectionBlock">
        <h2>2. Validate and Run</h2>
        <div className="fieldGrid">
          <label>
            Evaluation thresholds
            <input
              value={trainingForm.evaluationThresholds}
              onChange={(event) =>
                setTrainingForm((previous) => ({
                  ...previous,
                  evaluationThresholds: event.target.value,
                }))
              }
              placeholder="5,10,25"
            />
          </label>

          <label>
            Evaluation text path
            <input
              value={trainingForm.evaluationTextPath}
              onChange={(event) =>
                setTrainingForm((previous) => ({
                  ...previous,
                  evaluationTextPath: event.target.value,
                }))
              }
              placeholder="datasets/shake.txt"
            />
          </label>
        </div>

        <div className="actionRow">
          <button
            type="button"
            className="secondaryButton"
            onClick={handleValidate}
            disabled={isSubmitting || isValidating}
          >
            {isValidating ? "Validating..." : "Validate"}
          </button>
          <button
            type="button"
            className="primaryButton"
            onClick={handleTrain}
            disabled={isSubmitting || isValidating}
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
    </main>
  );
}
