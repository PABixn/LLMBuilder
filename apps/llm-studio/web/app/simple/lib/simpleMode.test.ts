import assert from "node:assert/strict";
import test from "node:test";

import type { ProjectDetail } from "../../../lib/api";
import type { TrainingJob as TokenizerJob } from "../../../lib/tokenizerLegacyApi";
import type {
  TrainingBatchLrRecommendation,
  TrainingCheckpointEntry,
  TrainingJob,
} from "../../../lib/training/types";
import {
  DEFAULT_SIMPLE_FLOW_STATE,
  SIMPLE_STREAMING_DATASET_FILTERS,
} from "../constants";
import { parseSimpleFlowState } from "../hooks/useSimpleFlowPersistence";
import { buildInferenceSettings } from "./inferencePresets";
import {
  SIMPLE_MODEL_PRESETS,
  assertPresetModelConfig,
  buildPresetModelConfig,
  getSimpleModelPreset,
  isSimpleModelPresetId,
  targetVocabForPresetDataset,
} from "./modelPresets";
import { deriveSimpleStepStatuses } from "./stepStatus";
import {
  buildSimpleTrainingDataloaderConfig,
  buildSimpleTrainingConfig,
  selectRecommendedTrainingOption,
} from "./trainingProfiles";
import {
  buildSimpleTokenizerDataloaderConfig,
  tokenizerBudgetForDataset,
} from "./tokenizerDefaults";
import {
  buildModelConfigWithSyncedVocab,
  modelNeedsTokenizerVocabSync,
  readTokenizerVocabSize,
} from "./vocabularySync";

function recommendation(): TrainingBatchLrRecommendation {
  return {
    headline: "Balanced",
    summary: "Use the balanced profile.",
    confidence: "high",
    current_total_batch_size: 16,
    current_learning_rate: 0.0003,
    current_micro_batch_size: null,
    current_grad_accum_steps: null,
    recommended_option_key: "balanced",
    options: [
      {
        key: "balanced",
        label: "Balanced",
        description: "Recommended",
        tone: "recommended",
        total_batch_size: 64,
        micro_batch_size: 4,
        grad_accum_steps: 16,
        learning_rate: 0.00025,
        estimated_tokens_per_run: 2048,
        recommended_max_steps: 120,
        estimated_tokens_per_recommended_run: 7680,
        estimated_local_passes_at_recommended_steps: null,
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
      local_dataset_count: 1,
      streaming_dataset_count: 0,
      local_file_count: 1,
      local_total_size_bytes: 1000,
      approx_local_tokens: 10_000,
      dominant_dataset_weight: 1,
      dataset_scale: "tiny",
      schedule_peak_factor: 1,
      warmup_fraction: 0.1,
      max_memory_micro_batch_size: 4,
      recommended_batch_target: 64,
      recommended_run_token_budget: 7680,
      parameter_scaled_run_token_target: 7680,
    },
  };
}

const trainingTemplate = {
  seq_len: 4096,
  max_steps: 500,
  total_batch_size: 16,
  micro_batch_size: 2,
  optimizer: {
    lr: 0.0003,
  },
  lr_scheduler: {
    type: "sequential",
    schedulers: [
      { type: "linear", steps: 50, start_factor: 0.1, end_factor: 1 },
      { type: "cosine_annealing", steps: 450, eta_min: 0.00001 },
    ],
  },
  sample_every: 200,
  save_every: 1000,
};

test("simple flow parser migrates malformed state to safe defaults", () => {
  assert.deepEqual(parseSimpleFlowState(null), DEFAULT_SIMPLE_FLOW_STATE);
  assert.equal(parseSimpleFlowState({ datasetSource: "bad" }).datasetSource, "starter");
  assert.equal(parseSimpleFlowState({ presetId: "bad" }).presetId, DEFAULT_SIMPLE_FLOW_STATE.presetId);
  assert.equal(parseSimpleFlowState({ trainingProfile: "longer" }).trainingProfile, "longer");
  assert.deepEqual(
    parseSimpleFlowState({
      localTrainFiles: [{ id: "a", fileName: "a.txt", filePath: "datasets/a.txt" }],
    }).localTrainFiles,
    [{ id: "a", fileName: "a.txt", filePath: "datasets/a.txt", sizeBytes: null, sizeChars: null }]
  );
});

test("model presets satisfy local structural constraints", () => {
  for (const preset of SIMPLE_MODEL_PRESETS) {
    const config = buildPresetModelConfig(preset.id);
    assert.doesNotThrow(() => assertPresetModelConfig(config));
  }
});

test("model presets carry coordinated simple-mode defaults", () => {
  const quickstart = getSimpleModelPreset(DEFAULT_SIMPLE_FLOW_STATE.presetId);
  assert.equal(isSimpleModelPresetId(quickstart.id), true);
  assert.equal(isSimpleModelPresetId("missing"), false);
  assert.equal(quickstart.defaultDatasetSource, "starter");
  assert.equal(quickstart.defaultTrainingProfile, "quick");
  assert.equal(quickstart.defaultExecutionKind, "local");
});

test("preset tokenizer targets stay small for starter data", () => {
  assert.equal(targetVocabForPresetDataset("nano-gpt-quick", "starter"), 1000);
  assert.equal(targetVocabForPresetDataset("gqa-balanced", "upload"), 16000);
  assert.equal(targetVocabForPresetDataset("gqa-balanced", "streaming"), 32000);
});

test("vocabulary sync reads tokenizer stats and updates model config", () => {
  const tokenizer = {
    stats: { vocab_size: 2048 },
    tokenizer_config: { vocab_size: 1000 },
  } as unknown as TokenizerJob;
  const project = {
    model_config: buildPresetModelConfig("nano-gpt-quick", { vocabSize: 1000 }),
  } as ProjectDetail;

  assert.equal(readTokenizerVocabSize(tokenizer), 2048);
  assert.equal(modelNeedsTokenizerVocabSync(project, tokenizer), true);
  assert.equal(buildModelConfigWithSyncedVocab(project.model_config, 2048).vocab_size, 2048);
});

test("training profiles apply backend recommendation and conservative caps", () => {
  const rec = recommendation();
  assert.equal(selectRecommendedTrainingOption(rec)?.key, "balanced");

  const quick = buildSimpleTrainingConfig(trainingTemplate, "quick", 1024, rec);
  assert.equal(quick.config.max_steps, 100);
  assert.equal(quick.config.seq_len, 256);
  assert.equal(quick.config.total_batch_size, 64);
  assert.equal((quick.config.optimizer as Record<string, unknown>).lr, 0.00025);
  assert.equal("micro_batch_size" in quick.config, false);
  assert.equal(quick.config.save_every, 100);

  const longer = buildSimpleTrainingConfig(trainingTemplate, "longer", 2048, rec);
  assert.equal(longer.config.max_steps, 240);
  assert.equal(longer.config.seq_len, 1024);
});

test("inference preset mapping hides numeric sampling controls behind stable choices", () => {
  assert.deepEqual(buildInferenceSettings("short", "precise"), {
    max_tokens: 48,
    temperature: 0.2,
    top_k: 20,
    repetition_penalty: 1.12,
  });
  assert.deepEqual(buildInferenceSettings("long", "creative"), {
    max_tokens: 160,
    temperature: 0.95,
    top_k: 80,
    repetition_penalty: 1.04,
  });
});

test("streaming data templates keep quality filters and tokenizer schema compatibility", () => {
  const tokenizerDataloader = buildSimpleTokenizerDataloaderConfig({
    datasetSource: "streaming",
    localTrainFiles: [],
    budgetLimit: tokenizerBudgetForDataset("streaming", 32000),
  });
  const tokenizerDataset = (tokenizerDataloader.datasets as Record<string, unknown>[])[0];
  assert.deepEqual(
    tokenizerDataset.filters,
    SIMPLE_STREAMING_DATASET_FILTERS.map((filter) => [...filter])
  );
  assert.equal("streaming" in tokenizerDataset, false);
  assert.equal((tokenizerDataloader.budget as Record<string, unknown>).limit, 16_000_000);

  const trainingDataloader = buildSimpleTrainingDataloaderConfig({}, "streaming", []);
  const trainingDataset = (trainingDataloader.datasets as Record<string, unknown>[])[0];
  assert.deepEqual(
    trainingDataset.filters,
    SIMPLE_STREAMING_DATASET_FILTERS.map((filter) => [...filter])
  );
  assert.equal(trainingDataset.streaming, true);
});

test("step readiness derives from artifacts instead of only persisted state", () => {
  const project = {
    valid: true,
    name: "model",
    artifact_file: "model.json",
  } as ProjectDetail;
  const tokenizer = {
    id: "tok",
    status: "completed",
    stats: { vocab_size: 1000 },
    tokenizer_config: { name: "tok", vocab_size: 1000 },
  } as unknown as TokenizerJob;
  const run = {
    id: "run",
    name: "run",
    status: "completed",
    checkpoint_count: 1,
  } as TrainingJob;
  const checkpoint = { step: 10 } as TrainingCheckpointEntry;
  const statuses = deriveSimpleStepStatuses({
    flow: {
      ...DEFAULT_SIMPLE_FLOW_STATE,
      projectId: "project",
      tokenizerJobId: "tok",
      trainingJobId: "run",
    },
    project,
    projectLoading: false,
    projectError: null,
    tokenizerJob: tokenizer,
    tokenizerError: null,
    datasetReady: true,
    datasetBlocker: null,
    tokenizerValidationError: null,
    trainingRun: run,
    trainingCheckpoints: [checkpoint],
    preflightValid: true,
    preflightError: null,
    trainingLaunching: false,
    inferenceGenerating: false,
    generationSucceeded: false,
    checkpointError: null,
  });

  assert.deepEqual(
    statuses.map((status) => status.state),
    ["completed", "completed", "completed", "ready"]
  );
});
