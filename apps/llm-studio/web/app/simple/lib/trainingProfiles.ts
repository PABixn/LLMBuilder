import type {
  TrainingBatchLrRecommendation,
  TrainingBatchLrRecommendationOption,
  TrainingFixSuggestion,
} from "../../../lib/training/types";
import {
  SIMPLE_STARTER_DATASET_PATH,
} from "../constants";
import type {
  SimpleDatasetSource,
  SimpleLocalTrainFile,
  SimpleStreamingDatasetId,
  SimpleTrainingProfile,
} from "../types";
import { buildSimpleStreamingDatasetSpecs } from "./streamingDatasets";
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

const TOTAL_BATCH_TOKEN_LIMITS: Record<
  SimpleTrainingProfile,
  { floor: number; cap: number }
> = {
  quick: { floor: 2_048, cap: 8_192 },
  balanced: { floor: 8_192, cap: 32_768 },
  longer: { floor: 16_384, cap: 65_536 },
};

const LEARNING_RATE_LIMITS: Record<
  SimpleTrainingProfile,
  { fallback: number; floor: number; cap: number }
> = {
  quick: { fallback: 0.0005, floor: 0.0001, cap: 0.0008 },
  balanced: { fallback: 0.0003, floor: 0.00008, cap: 0.0006 },
  longer: { fallback: 0.0002, floor: 0.00005, cap: 0.0004 },
};

const STEP_LIMITS: Record<
  SimpleTrainingProfile,
  { fallback: number; floor: number; cap: number }
> = {
  quick: { fallback: 100, floor: 100, cap: 200 },
  balanced: { fallback: 500, floor: 500, cap: 20_000 },
  longer: { fallback: 1000, floor: 1000, cap: 6000 },
};

const LONGER_PROFILE_TOKEN_BUDGET_MULTIPLIER = 2;

function clampInteger(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Math.trunc(value)));
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function positiveIntegerOrFallback(value: number | null | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? Math.trunc(value)
    : fallback;
}

function totalBatchTokensForProfile(
  profile: SimpleTrainingProfile,
  currentValue: unknown,
  option: TrainingBatchLrRecommendationOption | null
): number {
  const limits = TOTAL_BATCH_TOKEN_LIMITS[profile];
  if (
    profile !== "quick" &&
    typeof option?.total_batch_size === "number" &&
    Number.isFinite(option.total_batch_size) &&
    option.total_batch_size > 0
  ) {
    return Math.trunc(option.total_batch_size);
  }
  const recommendedBatchTokens = positiveIntegerOrFallback(
    option?.total_batch_size,
    asNumber(currentValue, limits.floor)
  );
  return clampInteger(
    Math.max(recommendedBatchTokens, limits.floor),
    limits.floor,
    limits.cap
  );
}

function learningRateForProfile(
  profile: SimpleTrainingProfile,
  currentValue: unknown,
  option: TrainingBatchLrRecommendationOption | null
): number {
  const limits = LEARNING_RATE_LIMITS[profile];
  if (
    profile !== "quick" &&
    typeof option?.learning_rate === "number" &&
    Number.isFinite(option.learning_rate) &&
    option.learning_rate > 0
  ) {
    return option.learning_rate;
  }
  const recommendedLearningRate =
    typeof option?.learning_rate === "number" &&
    Number.isFinite(option.learning_rate) &&
    option.learning_rate > 0
      ? option.learning_rate
      : asNumber(currentValue, limits.fallback);
  return clampNumber(recommendedLearningRate, limits.floor, limits.cap);
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

function longerProfileStepCap(
  recommendation: TrainingBatchLrRecommendation | null | undefined
): number {
  const signals = recommendation?.signals;
  const datasetScale = signals?.dataset_scale ?? "";
  const tinyDataset =
    datasetScale === "tiny" ||
    datasetScale === "tiny_local" ||
    (typeof signals?.approx_local_tokens === "number" && signals.approx_local_tokens < 50_000);
  const smallDataset =
    datasetScale === "small_local" ||
    (typeof signals?.approx_local_tokens === "number" && signals.approx_local_tokens < 500_000);
  return tinyDataset ? 1500 : smallDataset ? 3000 : STEP_LIMITS.longer.cap;
}

function stepCountForProfile(
  profile: SimpleTrainingProfile,
  value: number | null | undefined,
  recommendation: TrainingBatchLrRecommendation | null | undefined
): number {
  const limits = STEP_LIMITS[profile];
  const requestedSteps = positiveIntegerOrFallback(value, limits.fallback);

  if (profile === "longer") {
    return clampInteger(
      Math.max(requestedSteps, limits.floor),
      limits.floor,
      longerProfileStepCap(recommendation)
    );
  }

  return clampInteger(
    Math.max(requestedSteps, limits.floor),
    limits.floor,
    limits.cap
  );
}

function recommendedStepsForProfile(
  profile: SimpleTrainingProfile,
  option: TrainingBatchLrRecommendationOption | null,
  recommendation: TrainingBatchLrRecommendation | null | undefined
): number {
  const recommendedMaxSteps = option?.recommended_max_steps;
  if (
    typeof recommendedMaxSteps === "number" &&
    Number.isFinite(recommendedMaxSteps) &&
    recommendedMaxSteps > 0
  ) {
    if (profile === "balanced") {
      return Math.trunc(recommendedMaxSteps);
    }
    if (profile === "longer") {
      return clampInteger(
        Math.ceil(recommendedMaxSteps * LONGER_PROFILE_TOKEN_BUDGET_MULTIPLIER),
        1,
        longerProfileStepCap(recommendation)
      );
    }
  }
  const requestedSteps =
    profile === "longer" && typeof recommendedMaxSteps === "number"
      ? recommendedMaxSteps * LONGER_PROFILE_TOKEN_BUDGET_MULTIPLIER
      : recommendedMaxSteps;
  return stepCountForProfile(profile, requestedSteps, recommendation);
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

function cadenceForFraction(maxSteps: number, fraction: number): number {
  return Math.max(1, Math.min(maxSteps, Math.round(maxSteps * fraction)));
}

function setCheckpointCadence(
  config: Record<string, unknown>,
  maxSteps: number
): Record<string, unknown> {
  return {
    ...config,
    save_every: cadenceForFraction(maxSteps, 0.1),
  };
}

export function applySimpleTrainingProfileGuardrails(
  config: Record<string, unknown>,
  profile: SimpleTrainingProfile,
  appliedRecommendation: TrainingBatchLrRecommendationOption | null = null
): Record<string, unknown> {
  let nextConfig = cloneRecord(config);
  const hasBackendRecommendation = profile !== "quick" && appliedRecommendation !== null;
  const maxSteps = hasBackendRecommendation
    ? positiveIntegerOrFallback(
        asNumber(nextConfig.max_steps, STEP_LIMITS[profile].fallback),
        STEP_LIMITS[profile].fallback
      )
    : stepCountForProfile(
        profile,
        asNumber(nextConfig.max_steps, STEP_LIMITS[profile].fallback),
        null
      );
  const learningRate = hasBackendRecommendation
    ? asNumber(asRecord(nextConfig.optimizer).lr, appliedRecommendation.learning_rate)
    : learningRateForProfile(
        profile,
        asRecord(nextConfig.optimizer).lr,
        null
      );

  nextConfig.max_steps = maxSteps;
  nextConfig.total_batch_size = hasBackendRecommendation
    ? positiveIntegerOrFallback(
        asNumber(nextConfig.total_batch_size, appliedRecommendation.total_batch_size),
        appliedRecommendation.total_batch_size
      )
    : totalBatchTokensForProfile(
        profile,
        nextConfig.total_batch_size,
        null
      );
  nextConfig = setOptimizerLearningRate(nextConfig, learningRate);
  nextConfig = refitScheduler(nextConfig, maxSteps, learningRate);

  if (profile === "quick") {
    nextConfig.sample_every = Math.max(10, Math.floor(maxSteps / 4));
    return setCheckpointCadence(nextConfig, maxSteps);
  }

  nextConfig = clampIntervalField(nextConfig, "sample_every", maxSteps);
  nextConfig = clampIntervalField(nextConfig, "save_every", maxSteps);
  return setCheckpointCadence(nextConfig, maxSteps);
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
    if (option.clear_manual_micro_batch) {
      delete config.micro_batch_size;
    }
  }
  const learningRate = learningRateForProfile(profile, asRecord(config.optimizer).lr, option);
  config = setOptimizerLearningRate(config, learningRate);
  config.total_batch_size = totalBatchTokensForProfile(
    profile,
    config.total_batch_size,
    option
  );

  config = refitScheduler(config, maxSteps, learningRate);

  if (profile === "quick") {
    config.sample_every = Math.max(10, Math.floor(maxSteps / 4));
    config = setCheckpointCadence(config, maxSteps);
  } else {
    config = clampIntervalField(config, "sample_every", maxSteps);
    config = clampIntervalField(config, "save_every", maxSteps);
    config = setCheckpointCadence(config, maxSteps);
  }

  const profileLabel =
    profile === "quick" ? "Quick check" : profile === "balanced" ? "Balanced" : "Longer run";
  const totalBatchTokens = asNumber(config.total_batch_size, TOTAL_BATCH_TOKEN_LIMITS[profile].floor);
  const note = option
    ? `${profileLabel}: ${maxSteps.toLocaleString()} steps at ${seqLen.toLocaleString()} tokens, ${totalBatchTokens.toLocaleString()} total batch tokens, with bounded backend learning-rate guidance.`
    : `${profileLabel}: ${maxSteps.toLocaleString()} steps at ${seqLen.toLocaleString()} tokens, ${totalBatchTokens.toLocaleString()} total batch tokens while waiting for backend guidance.`;

  return {
    config,
    note,
    appliedRecommendation: option,
  };
}

export function buildSimpleTrainingDataloaderConfig(
  template: Record<string, unknown>,
  datasetSource: SimpleDatasetSource,
  localTrainFiles: SimpleLocalTrainFile[],
  streamingPrimaryDatasetId: SimpleStreamingDatasetId,
  streamingAdditionalDatasetIds: SimpleStreamingDatasetId[]
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

  config.datasets = buildSimpleStreamingDatasetSpecs(
    streamingPrimaryDatasetId,
    streamingAdditionalDatasetIds,
    { includeStreamingFlag: true }
  );
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
