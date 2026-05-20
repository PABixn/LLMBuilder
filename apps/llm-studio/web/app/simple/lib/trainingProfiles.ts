import type {
  TrainingBatchLrRecommendation,
  TrainingBatchLrRecommendationOption,
  TrainingFixSuggestion,
} from "../../../lib/training/types";
import {
  SIMPLE_STARTER_DATASET_PATH,
  SIMPLE_STREAMING_DATASET_FILTERS,
  SIMPLE_STREAMING_DATASET_NAME,
} from "../constants";
import type {
  SimpleDatasetSource,
  SimpleLocalTrainFile,
  SimpleTrainingProfile,
} from "../types";
import { fitSchedulersToMaxSteps } from "../../training/lib/learningRateSchedule";
import {
  asNumber,
  asRecord,
  asRecordArray,
  cloneRecord,
  deleteAtPath,
  updateAtPath,
} from "../../training/lib/object";

export interface SimpleTrainingProfileResult {
  config: Record<string, unknown>;
  note: string;
  appliedRecommendation: TrainingBatchLrRecommendationOption | null;
}

export interface AppliedTrainingFixResult {
  trainingConfig: Record<string, unknown>;
  dataloaderConfig: Record<string, unknown>;
  labels: string[];
}

function clampInteger(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Math.trunc(value)));
}

export function selectRecommendedTrainingOption(
  recommendation: TrainingBatchLrRecommendation | null | undefined
): TrainingBatchLrRecommendationOption | null {
  if (!recommendation || recommendation.options.length === 0) {
    return null;
  }
  return (
    recommendation.options.find((option) => option.key === recommendation.recommended_option_key) ??
    recommendation.options[0]
  );
}

function recommendedStepsForProfile(
  profile: SimpleTrainingProfile,
  option: TrainingBatchLrRecommendationOption | null,
  recommendation: TrainingBatchLrRecommendation | null | undefined
): number {
  const templateFallback =
    profile === "quick" ? 100 : profile === "balanced" ? 500 : 1000;
  const recommendedMaxSteps = option?.recommended_max_steps ?? templateFallback;

  if (profile === "quick") {
    return clampInteger(Math.min(recommendedMaxSteps, 100), 20, 100);
  }

  if (profile === "balanced") {
    return clampInteger(recommendedMaxSteps, 1, 20_000);
  }

  const doubled = recommendedMaxSteps * 2;
  const signals = recommendation?.signals;
  const datasetScale = signals?.dataset_scale ?? "";
  const tinyDataset =
    datasetScale === "tiny" ||
    datasetScale === "tiny_local" ||
    (typeof signals?.approx_local_tokens === "number" && signals.approx_local_tokens < 50_000);
  const smallDataset =
    datasetScale === "small_local" ||
    (typeof signals?.approx_local_tokens === "number" && signals.approx_local_tokens < 500_000);
  const upperCap = tinyDataset ? 300 : smallDataset ? 800 : 2000;
  return clampInteger(Math.min(doubled, upperCap), 20, upperCap);
}

function sequenceLengthForProfile(
  profile: SimpleTrainingProfile,
  modelContextLength: number
): number {
  const safeContextLength = Math.max(1, Math.trunc(modelContextLength));
  const preferred =
    profile === "quick"
      ? 256
      : profile === "balanced"
        ? 512
        : 1024;
  return clampInteger(Math.min(preferred, safeContextLength), 1, safeContextLength);
}

function setOptimizerLearningRate(
  config: Record<string, unknown>,
  learningRate: number
): Record<string, unknown> {
  const optimizer = {
    ...asRecord(config.optimizer),
    lr: learningRate,
  };
  return {
    ...config,
    optimizer,
  };
}

function refitScheduler(
  config: Record<string, unknown>,
  maxSteps: number,
  learningRate: number
): Record<string, unknown> {
  const lrScheduler = asRecord(config.lr_scheduler);
  const schedulers = asRecordArray(lrScheduler.schedulers);
  return {
    ...config,
    lr_scheduler: {
      type: "sequential",
      schedulers: fitSchedulersToMaxSteps(schedulers, maxSteps, learningRate),
    },
  };
}

function clampIntervalField(
  config: Record<string, unknown>,
  key: "sample_every" | "save_every",
  maxSteps: number
): Record<string, unknown> {
  const value = asNumber(config[key], Number.NaN);
  if (!Number.isFinite(value)) {
    return config;
  }
  return {
    ...config,
    [key]: clampInteger(value, 1, maxSteps),
  };
}

export function buildSimpleTrainingConfig(
  template: Record<string, unknown>,
  profile: SimpleTrainingProfile,
  modelContextLength: number,
  recommendation: TrainingBatchLrRecommendation | null | undefined
): SimpleTrainingProfileResult {
  const option = selectRecommendedTrainingOption(recommendation);
  const maxSteps = recommendedStepsForProfile(profile, option, recommendation);
  const seqLen = sequenceLengthForProfile(profile, modelContextLength);

  let config = cloneRecord(template);
  config.seq_len = seqLen;
  config.max_steps = maxSteps;

  if (option) {
    config.total_batch_size = option.total_batch_size;
    if (option.clear_manual_micro_batch) {
      delete config.micro_batch_size;
    }
    config = setOptimizerLearningRate(config, option.learning_rate);
  }

  const learningRate = asNumber(asRecord(config.optimizer).lr, 0.0003);
  config = refitScheduler(config, maxSteps, learningRate);

  if (profile === "quick") {
    config.sample_every = Math.max(10, Math.floor(maxSteps / 4));
    config.save_every = maxSteps;
  } else {
    config = clampIntervalField(config, "sample_every", maxSteps);
    config = clampIntervalField(config, "save_every", maxSteps);
  }

  const profileLabel =
    profile === "quick" ? "Quick check" : profile === "balanced" ? "Balanced" : "Longer run";
  const note = option
    ? `${profileLabel}: ${maxSteps.toLocaleString()} steps at ${seqLen.toLocaleString()} tokens, with backend batch and learning-rate guidance.`
    : `${profileLabel}: ${maxSteps.toLocaleString()} steps at ${seqLen.toLocaleString()} tokens while waiting for backend guidance.`;

  return {
    config,
    note,
    appliedRecommendation: option,
  };
}

export function buildSimpleTrainingDataloaderConfig(
  template: Record<string, unknown>,
  datasetSource: SimpleDatasetSource,
  localTrainFiles: SimpleLocalTrainFile[]
): Record<string, unknown> {
  const config = cloneRecord(template);

  if (datasetSource === "starter") {
    config.datasets = [
      {
        name: "text",
        split: "train",
        text_columns: ["text"],
        weight: 1,
        streaming: true,
        data_files: { train: SIMPLE_STARTER_DATASET_PATH },
      },
    ];
    return config;
  }

  if (datasetSource === "upload") {
    const paths = localTrainFiles
      .map((file) => file.filePath.trim())
      .filter((filePath) => filePath !== "");
    config.datasets = [
      {
        name: "text",
        split: "train",
        text_columns: ["text"],
        weight: 1,
        streaming: true,
        data_files: { train: paths.length <= 1 ? (paths[0] ?? "") : paths },
      },
    ];
    return config;
  }

  config.datasets = [
    {
      name: SIMPLE_STREAMING_DATASET_NAME,
      split: "train",
      text_columns: ["text"],
      weight: 1,
      filters: SIMPLE_STREAMING_DATASET_FILTERS.map((filter) => [...filter]),
      streaming: true,
    },
  ];
  return config;
}

export function applySafeTrainingFixes(
  trainingConfig: Record<string, unknown>,
  dataloaderConfig: Record<string, unknown>,
  fixes: TrainingFixSuggestion[]
): AppliedTrainingFixResult {
  let nextTrainingConfig = cloneRecord(trainingConfig);
  let nextDataloaderConfig = cloneRecord(dataloaderConfig);
  const labels: string[] = [];

  for (const fix of fixes) {
    if (fix.path.startsWith("training_config.")) {
      const path = fix.path.replace("training_config.", "").split(".");
      nextTrainingConfig =
        fix.value === null
          ? deleteAtPath(nextTrainingConfig, path)
          : updateAtPath(nextTrainingConfig, path, cloneRecord(fix.value));
      labels.push(fix.label);
      continue;
    }

    if (fix.path.startsWith("dataloader_config.")) {
      const path = fix.path.replace("dataloader_config.", "").split(".");
      nextDataloaderConfig =
        fix.value === null
          ? deleteAtPath(nextDataloaderConfig, path)
          : updateAtPath(nextDataloaderConfig, path, cloneRecord(fix.value));
      labels.push(fix.label);
    }
  }

  return {
    trainingConfig: nextTrainingConfig,
    dataloaderConfig: nextDataloaderConfig,
    labels,
  };
}
