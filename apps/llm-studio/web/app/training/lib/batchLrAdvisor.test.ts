import assert from "node:assert/strict";
import { test } from "node:test";
import type { TrainingBatchLrRecommendation } from "../../../lib/training/types";
import { buildBatchLrAdvisorViewModel } from "./batchLrAdvisor";

const recommendation: TrainingBatchLrRecommendation = {
  headline: "Use a balanced optimizer step",
  summary: "A balanced profile fits the current model and dataset.",
  confidence: "high",
  current_total_batch_size: 2048,
  current_learning_rate: 0.0003,
  current_micro_batch_size: null,
  current_grad_accum_steps: null,
  recommended_option_key: "balanced",
  options: [
    {
      key: "conservative",
      label: "Conservative",
      description: "Smaller updates",
      tone: "neutral",
      total_batch_size: 1024,
      micro_batch_size: 1,
      grad_accum_steps: 8,
      learning_rate: 0.0002,
      estimated_tokens_per_run: 102400,
      clear_manual_micro_batch: true,
    },
    {
      key: "balanced",
      label: "Balanced",
      description: "Recommended updates",
      tone: "recommended",
      total_batch_size: 2048,
      micro_batch_size: 2,
      grad_accum_steps: 8,
      learning_rate: 0.0003,
      estimated_tokens_per_run: 204800,
      clear_manual_micro_batch: false,
    },
  ],
  factors: [],
  signals: {
    device: "NVIDIA A100",
    device_type: "A100",
    total_parameters: 1000000000,
    parameter_memory_bytes_bf16: 2000000000,
    estimated_kv_cache_bytes_for_context_fp16: 256000000,
    block_count: 24,
    attention_component_count: 24,
    max_mlp_multiplier: 4,
    dataset_count: 1,
    local_dataset_count: 1,
    streaming_dataset_count: 0,
    local_file_count: 2,
    local_total_size_bytes: 1000000,
    dominant_dataset_weight: 1,
    dataset_scale: "small_local",
    schedule_peak_factor: 1.5,
    warmup_fraction: 0.1,
    max_memory_micro_batch_size: 3,
    recommended_batch_target: 2048,
  },
};

test("batch LR advisor falls back to the recommended option", () => {
  const viewModel = buildBatchLrAdvisorViewModel(recommendation, "missing");

  assert.equal(viewModel.selectedRecommendationOption?.key, "balanced");
  assert.equal(viewModel.selectedRecommendationIsRecommended, true);
  assert.equal(viewModel.recommendationConfidenceTone, "tone-good");
  assert.match(viewModel.selectedBatchTooltipSummary, /default step size/);
});

test("batch LR advisor explains alternate LR and batch profiles", () => {
  const viewModel = buildBatchLrAdvisorViewModel(recommendation, "conservative");

  assert.equal(viewModel.selectedRecommendationOption?.key, "conservative");
  assert.equal(viewModel.selectedRecommendationIsRecommended, false);
  assert.match(viewModel.selectedBatchTooltipSummary, /smaller than Balanced/);
  assert.match(viewModel.selectedLearningRateTooltipSummary, /lowers the base LR/);
  assert.equal(viewModel.selectedBatchTooltipItems.length, 2);
  assert.equal(viewModel.selectedLearningRateTooltipItems.length, 2);
});
