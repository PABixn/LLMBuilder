import type { SimpleFlowState } from "./types";

export const SIMPLE_FLOW_STORAGE_KEY = "llm-studio-simple-flow-v1";
export const SIMPLE_FLOW_VERSION = 1;
export const SIMPLE_POLL_INTERVAL_MS = 1200;
export const SIMPLE_RECENT_RUNS_POLL_INTERVAL_MS = 3500;
export const SIMPLE_STARTER_DATASET_PATH = "datasets/shake.txt";
export const SIMPLE_STARTER_DATASET_NAME = "Starter Shakespeare sample";
export const SIMPLE_DEFAULT_STREAMING_PRIMARY_DATASET_ID = "fineweb-edu";
export const SIMPLE_STREAMING_DATASET_FILTERS = [
  ["language_score", ">", 0.95],
  ["int_score", ">=", 4],
] as const;
export const SIMPLE_DEFAULT_PROMPT = "Once upon a time";
export const SIMPLE_LATEST_CHECKPOINT_VALUE = "latest";

export const DEFAULT_SIMPLE_FLOW_STATE: SimpleFlowState = {
  version: SIMPLE_FLOW_VERSION,
  presetId: "nano-gpt-quick",
  modelName: "Local quickstart model",
  targetVocabSize: 1000,
  targetContextLength: 512,
  projectId: null,
  tokenizerJobId: null,
  trainingJobId: null,
  datasetSource: "starter",
  localTrainFiles: [],
  streamingPrimaryDatasetId: SIMPLE_DEFAULT_STREAMING_PRIMARY_DATASET_ID,
  streamingAdditionalDatasetIds: [],
  trainingProfile: "quick",
  executionKind: "local",
  checkpointValue: SIMPLE_LATEST_CHECKPOINT_VALUE,
  lastCompletedStep: null,
};
