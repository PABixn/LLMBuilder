import type { FilterOperator, SettingsCategory } from "./types";

export const FILTER_OPERATORS: FilterOperator[] = [
  "==",
  "!=",
  ">",
  ">=",
  "<",
  "<=",
  "in",
  "not in",
];

export const LEGACY_TOKENIZER_THEME_STORAGE_KEY = "tokenizer-studio-theme";
export const TOKENIZER_FORM_STORAGE_KEY = "tokenizer-studio-tokenizer-form";
export const DATASET_FORM_STORAGE_KEY = "tokenizer-studio-dataset-form";
export const TRAINING_FORM_STORAGE_KEY = "tokenizer-studio-training-form";
export const ACTIVE_JOB_STORAGE_KEY = "tokenizer-studio-active-job-id";
export const PREVIEW_TEXT_STORAGE_KEY = "tokenizer-studio-preview-text";
export const HIDDEN_RECENT_JOB_IDS_STORAGE_KEY = "tokenizer-studio-hidden-recent-job-ids";

export const SETTINGS_CATEGORY_HASH_MAP: Record<SettingsCategory, string> = {
  tokenizer: "#settings-tokenizer",
  dataset: "#settings-dataset",
  training: "#settings-training",
};

export const WEIGHT_SUM_EPSILON = 1e-9;
export const WEIGHT_SCALE = 1_000_000;
export const MIN_STRICT_POSITIVE_WEIGHT = 1e-6;
export const LEGACY_TOKENIZER_API_SEGMENT = "apps/tokenizer-studio/api/";
export const CURRENT_TOKENIZER_API_SEGMENT = "apps/llm-studio/api/";
export const LEGACY_UPLOADS_SEGMENT = "/datasets/uploads/";
export const DEFAULT_SHAKE_DATASET_PATH = "datasets/shake.txt";
