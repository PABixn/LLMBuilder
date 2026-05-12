import type {
  TrainingBatchLrRecommendation,
  TrainingBatchLrRecommendationFactor,
  TrainingBatchLrRecommendationOption,
} from "../../../lib/training/types";
import { formatDatasetScaleLabel } from "./display";
import { formatInteger } from "./metrics";
import { formatLearningRate } from "./run";

type AdvisorTooltipItem = {
  label: string;
  detail: string;
};

export type BatchLrAdvisorViewModel = {
  recommendationConfidenceLabel: string | null;
  recommendationConfidenceTone: string;
  highlightedRecommendationFactors: TrainingBatchLrRecommendationFactor[];
  recommendedRecommendationOption: TrainingBatchLrRecommendationOption | null;
  selectedBatchTooltipItems: AdvisorTooltipItem[];
  selectedBatchTooltipSummary: string;
  selectedLearningRateTooltipItems: AdvisorTooltipItem[];
  selectedLearningRateTooltipSummary: string;
  selectedPeakLearningRate: number | null;
  selectedRecommendationIsRecommended: boolean;
  selectedRecommendationOption: TrainingBatchLrRecommendationOption | null;
  selectedStepTooltipItems: AdvisorTooltipItem[];
  selectedStepTooltipSummary: string;
};

function formatRelativeDeltaPercent(value: number, baseline: number): string {
  if (!Number.isFinite(value) || !Number.isFinite(baseline) || baseline <= 0) {
    return "0%";
  }
  return `${Math.max(1, Math.round((Math.abs(value - baseline) / baseline) * 100))}%`;
}

function describeBatchProfileShift(
  selectedBatchSize: number,
  baselineBatchSize: number,
  baselineLabel: string,
  isRecommended: boolean
): string {
  if (isRecommended) {
    return "This is the default step size because it best balances stability, accumulation depth, and throughput for the current run.";
  }
  if (selectedBatchSize < baselineBatchSize) {
    return `This profile keeps the optimizer step ${formatRelativeDeltaPercent(
      selectedBatchSize,
      baselineBatchSize
    )} smaller than ${baselineLabel} so each update is less aggressive and easier to stabilize.`;
  }
  if (selectedBatchSize > baselineBatchSize) {
    return `This profile makes the optimizer step ${formatRelativeDeltaPercent(
      selectedBatchSize,
      baselineBatchSize
    )} larger than ${baselineLabel} to push more tokens through each optimizer update.`;
  }
  return `This profile keeps the same optimizer-step size as ${baselineLabel}; the main difference is how aggressively it uses that step.`;
}

function describeLearningRateProfileShift(
  selectedLearningRate: number,
  baselineLearningRate: number,
  baselineLabel: string,
  isRecommended: boolean,
  peakLearningRate: number | null
): string {
  if (isRecommended) {
    return "This is the default base LR because it is the best fit for the current model scale, data regime, and scheduler.";
  }
  if (selectedLearningRate < baselineLearningRate) {
    return `This profile lowers the base LR by ${formatRelativeDeltaPercent(
      selectedLearningRate,
      baselineLearningRate
    )} versus ${baselineLabel}${
      peakLearningRate ? `, keeping the effective peak near ${formatLearningRate(peakLearningRate)}` : ""
    }.`;
  }
  if (selectedLearningRate > baselineLearningRate) {
    return `This profile raises the base LR by ${formatRelativeDeltaPercent(
      selectedLearningRate,
      baselineLearningRate
    )} versus ${baselineLabel}${
      peakLearningRate ? `, letting the schedule top out near ${formatLearningRate(peakLearningRate)}` : ""
    }.`;
  }
  return `This profile lands on the same canonical LR as ${baselineLabel}, so the main change comes from batch layout rather than LR.`;
}

function describeStepProfileShift(
  selectedMaxSteps: number,
  baselineMaxSteps: number,
  baselineLabel: string,
  isRecommended: boolean
): string {
  if (isRecommended) {
    return "This max-step cap matches the advisor's recommended run-token budget for the selected optimizer-step size.";
  }
  if (selectedMaxSteps < baselineMaxSteps) {
    return `This profile shortens the run by ${formatRelativeDeltaPercent(
      selectedMaxSteps,
      baselineMaxSteps
    )} versus ${baselineLabel} because each optimizer step covers more tokens.`;
  }
  if (selectedMaxSteps > baselineMaxSteps) {
    return `This profile extends the run by ${formatRelativeDeltaPercent(
      selectedMaxSteps,
      baselineMaxSteps
    )} versus ${baselineLabel} so the smaller optimizer step still reaches a comparable token budget.`;
  }
  return `This profile keeps the same maximum training-step cap as ${baselineLabel}.`;
}

function describeBatchHardwareContext(
  deviceType: string,
  maxMemoryMicroBatchSize: number,
  microBatchSize: number,
  gradAccumSteps: number
): string {
  return `Preflight fits up to ${formatInteger(
    maxMemoryMicroBatchSize
  )} sequences per micro-step on ${deviceType}; this profile uses micro batch ${formatInteger(
    microBatchSize
  )} with ${formatInteger(gradAccumSteps)} accumulation step${gradAccumSteps === 1 ? "" : "s"}.`;
}

function describeBatchDatasetContext(datasetScale: string): string {
  if (datasetScale === "streaming") {
    return "Streaming-scale data lets the advisor lean more on hardware fit and model scale than on corpus-size caps.";
  }
  if (datasetScale === "mixed") {
    return "Mixed local and streaming data keeps batch sizing more conservative so the local portion is not washed out in each update.";
  }
  if (datasetScale === "tiny_local" || datasetScale === "small_local") {
    return "A small local corpus favors smaller optimizer steps so repeated passes over the same data do not get too sharp.";
  }
  return `${formatDatasetScaleLabel(
    datasetScale
  )} data still benefits from measured step sizing so each update covers a useful slice of the corpus without overreaching.`;
}

function describeLearningRateScheduleContext(
  peakFactor: number,
  peakLearningRate: number | null
): string {
  if (peakFactor > 1.01 && peakLearningRate) {
    return `The current scheduler still lifts this base LR during the run, topping out near ${formatLearningRate(
      peakLearningRate
    )}.`;
  }
  return "The current scheduler keeps the effective LR close to the base value, so the chosen base LR needs to be safe on its own.";
}

function describeLearningRateDatasetContext(datasetScale: string): string {
  if (datasetScale === "streaming") {
    return "With streaming-scale data, LR can stay anchored more to model scale and schedule shape than to corpus-size limits.";
  }
  if (datasetScale === "mixed") {
    return "A mixed data regime usually benefits from a moderate LR so the local data is not overfit while streaming data still moves quickly.";
  }
  if (datasetScale === "tiny_local" || datasetScale === "small_local") {
    return "Smaller local corpora generally favor a lower LR so repeated exposure to the same samples stays stable.";
  }
  return `${formatDatasetScaleLabel(
    datasetScale
  )} data supports a measured LR that still respects repeated passes over the local corpus.`;
}

function describeStepBudgetContext(
  estimatedTokensPerRecommendedRun: number,
  recommendedRunTokenBudget: number,
  parameterScaledRunTokenTarget: number
): string {
  if (recommendedRunTokenBudget < parameterScaledRunTokenTarget) {
    return `This profile lands near ${formatInteger(
      estimatedTokensPerRecommendedRun
    )} recommended run tokens. The unconstrained parameter-scaled anchor is ${formatInteger(
      parameterScaledRunTokenTarget
    )} tokens, but the active data regime reduces the practical recommendation to ${formatInteger(
      recommendedRunTokenBudget
    )}.`;
  }
  return `This profile lands near ${formatInteger(
    estimatedTokensPerRecommendedRun
  )} recommended run tokens, matching the parameter-scaled advisor budget of ${formatInteger(
    parameterScaledRunTokenTarget
  )} tokens.`;
}

function describeStepDatasetContext(
  datasetScale: string,
  estimatedLocalPassesAtRecommendedSteps: number | null
): string {
  if (estimatedLocalPassesAtRecommendedSteps !== null) {
    return `That works out to about ${estimatedLocalPassesAtRecommendedSteps.toFixed(
      1
    )} estimated passes over the current local corpus.`;
  }
  if (datasetScale === "mixed") {
    return "Mixed local and streaming data keeps the run cap more measured than an open-ended streaming-only launch.";
  }
  return "Streaming-scale data lets the advisor spend the run budget on model-scale optimization instead of strict corpus reuse limits.";
}

export function buildBatchLrAdvisorViewModel(
  recommendation: TrainingBatchLrRecommendation | null,
  selectedRecommendationOptionKey: string | null
): BatchLrAdvisorViewModel {
  const selectedRecommendationOption =
    recommendation?.options.find((option) => option.key === selectedRecommendationOptionKey) ??
    recommendation?.options.find((option) => option.key === recommendation.recommended_option_key) ??
    recommendation?.options[0] ??
    null;
  const recommendedRecommendationOption =
    recommendation?.options.find((option) => option.key === recommendation.recommended_option_key) ??
    recommendation?.options[0] ??
    null;
  const recommendationConfidenceTone =
    recommendation?.confidence === "high"
      ? "tone-good"
      : recommendation?.confidence === "low"
        ? "tone-warn"
        : "tone-neutral";
  const recommendationConfidenceLabel = recommendation?.confidence
    ? `${recommendation.confidence.charAt(0).toUpperCase()}${recommendation.confidence.slice(1)} confidence`
    : null;
  const selectedRecommendationIsRecommended =
    recommendation !== null &&
    selectedRecommendationOption !== null &&
    selectedRecommendationOption.key === recommendation.recommended_option_key;
  const selectedPeakLearningRate =
    recommendation && selectedRecommendationOption
      ? selectedRecommendationOption.learning_rate * recommendation.signals.schedule_peak_factor
      : null;
  const selectedBatchTooltipSummary =
    selectedRecommendationOption && recommendedRecommendationOption
      ? describeBatchProfileShift(
          selectedRecommendationOption.total_batch_size,
          recommendedRecommendationOption.total_batch_size,
          recommendedRecommendationOption.label,
          selectedRecommendationOption.key === recommendedRecommendationOption.key
        )
      : "";
  const selectedLearningRateTooltipSummary =
    selectedRecommendationOption && recommendedRecommendationOption
      ? describeLearningRateProfileShift(
          selectedRecommendationOption.learning_rate,
          recommendedRecommendationOption.learning_rate,
          recommendedRecommendationOption.label,
          selectedRecommendationOption.key === recommendedRecommendationOption.key,
          selectedPeakLearningRate
        )
      : "";
  const selectedStepTooltipSummary =
    selectedRecommendationOption && recommendedRecommendationOption
      ? describeStepProfileShift(
          selectedRecommendationOption.recommended_max_steps,
          recommendedRecommendationOption.recommended_max_steps,
          recommendedRecommendationOption.label,
          selectedRecommendationOption.key === recommendedRecommendationOption.key
        )
      : "";
  const selectedBatchTooltipItems =
    recommendation && selectedRecommendationOption
      ? [
          {
            label: "Hardware fit",
            detail: describeBatchHardwareContext(
              recommendation.signals.device_type,
              recommendation.signals.max_memory_micro_batch_size,
              selectedRecommendationOption.micro_batch_size,
              selectedRecommendationOption.grad_accum_steps
            ),
          },
          {
            label: "Data regime",
            detail: describeBatchDatasetContext(recommendation.signals.dataset_scale),
          },
        ]
      : [];
  const selectedLearningRateTooltipItems =
    recommendation && selectedRecommendationOption
      ? [
          {
            label: "Scheduler effect",
            detail: describeLearningRateScheduleContext(
              recommendation.signals.schedule_peak_factor,
              selectedPeakLearningRate
            ),
          },
          {
            label: "Data regime",
            detail: describeLearningRateDatasetContext(recommendation.signals.dataset_scale),
          },
        ]
      : [];
  const selectedStepTooltipItems =
    recommendation && selectedRecommendationOption
      ? [
          {
            label: "Run-token budget",
            detail: describeStepBudgetContext(
              selectedRecommendationOption.estimated_tokens_per_recommended_run,
              recommendation.signals.recommended_run_token_budget,
              recommendation.signals.parameter_scaled_run_token_target
            ),
          },
          {
            label: "Data regime",
            detail: describeStepDatasetContext(
              recommendation.signals.dataset_scale,
              selectedRecommendationOption.estimated_local_passes_at_recommended_steps
            ),
          },
        ]
      : [];
  const highlightedRecommendationFactors =
    recommendation?.factors.filter((factor) =>
      factor.code === "pretraining_anchor" ||
      factor.code === "dataset_scale" ||
      factor.code === "memory_fit" ||
      factor.code === "training_length"
    ) ?? [];

  return {
    recommendationConfidenceLabel,
    recommendationConfidenceTone,
    highlightedRecommendationFactors,
    recommendedRecommendationOption,
    selectedBatchTooltipItems,
    selectedBatchTooltipSummary,
    selectedLearningRateTooltipItems,
    selectedLearningRateTooltipSummary,
    selectedPeakLearningRate,
    selectedRecommendationIsRecommended,
    selectedRecommendationOption,
    selectedStepTooltipItems,
    selectedStepTooltipSummary,
  };
}
