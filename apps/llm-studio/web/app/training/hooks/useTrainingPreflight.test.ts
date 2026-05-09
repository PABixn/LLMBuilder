import assert from "node:assert/strict";
import test from "node:test";

import type { TrainingBatchLrRecommendation } from "../../../lib/training/types";
import { selectRecommendationOptionKey } from "./useTrainingPreflight";

const recommendation: TrainingBatchLrRecommendation = {
  headline: "Use a balanced profile",
  summary: "Balanced batch and LR.",
  confidence: "high",
  current_total_batch_size: 32,
  current_learning_rate: 0.0003,
  current_micro_batch_size: null,
  current_grad_accum_steps: null,
  recommended_option_key: "balanced",
  options: [
    {
      key: "balanced",
      label: "Balanced",
      description: "Balanced profile.",
      tone: "recommended",
      total_batch_size: 32,
      micro_batch_size: 4,
      grad_accum_steps: 8,
      learning_rate: 0.0003,
      estimated_tokens_per_run: 1024,
      clear_manual_micro_batch: false,
    },
    {
      key: "conservative",
      label: "Conservative",
      description: "Smaller optimizer step.",
      tone: "neutral",
      total_batch_size: 16,
      micro_batch_size: 2,
      grad_accum_steps: 8,
      learning_rate: 0.00015,
      estimated_tokens_per_run: 512,
      clear_manual_micro_batch: true,
    },
  ],
  factors: [],
  signals: {
    device: "cpu",
    device_type: "cpu",
    total_parameters: 1000,
    parameter_memory_bytes_bf16: 2000,
    estimated_kv_cache_bytes_for_context_fp16: 3000,
    block_count: 2,
    attention_component_count: 2,
    max_mlp_multiplier: 4,
    dataset_count: 1,
    local_dataset_count: 0,
    streaming_dataset_count: 1,
    local_file_count: 0,
    local_total_size_bytes: null,
    dominant_dataset_weight: 1,
    dataset_scale: "small",
    schedule_peak_factor: 1,
    warmup_fraction: 0.1,
    max_memory_micro_batch_size: 4,
    recommended_batch_target: 32,
  },
};

test("preflight recommendation key keeps a still-valid manual selection", () => {
  assert.equal(selectRecommendationOptionKey("conservative", recommendation), "conservative");
});

test("preflight recommendation key falls back to the recommended option", () => {
  assert.equal(selectRecommendationOptionKey(null, recommendation), "balanced");
  assert.equal(selectRecommendationOptionKey("missing", recommendation), "balanced");
});

test("preflight recommendation key clears when no options are available", () => {
  assert.equal(selectRecommendationOptionKey("balanced", null), null);
  assert.equal(
    selectRecommendationOptionKey("balanced", {
      ...recommendation,
      options: [],
    }),
    null
  );
});
