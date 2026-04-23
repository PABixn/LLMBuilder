import type { FilterOperator, WorkflowTarget } from "./types";

export const FILTER_OPERATORS: FilterOperator[] = ["==", "!=", ">", ">=", "<", "<=", "in", "not in"];
export const WEIGHT_SUM_EPSILON = 1e-9;
export const WEIGHT_SCALE = 1_000_000;

export const TRAINING_CONFIG_STORAGE_KEY = "llm-training-config-v1";
export const DATALOADER_CONFIG_STORAGE_KEY = "llm-training-dataloader-v1";
export const TRAINING_SELECTION_STORAGE_KEY = "llm-training-selection-v1";
export const ACTIVE_RUN_STORAGE_KEY = "llm-training-active-run-v1";
export const POLL_INTERVAL_MS = 1000;
export const RECENT_RUNS_POLL_INTERVAL_MS = 1000;

export const WORKFLOW_TARGET_HASH_MAP: Record<WorkflowTarget, string> = {
  model: "#settings-model",
  tokenizer: "#settings-tokenizer",
  training: "#settings-training",
  dataset: "#settings-dataset",
  preflight: "#settings-preflight",
};
