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

export const TOKENIZER_FORM_STORAGE_KEY = "llm-studio-tokenizer-form";
export const DATASET_FORM_STORAGE_KEY = "llm-studio-tokenizer-dataset-form";
export const TRAINING_FORM_STORAGE_KEY = "llm-studio-tokenizer-training-form";
export const ACTIVE_JOB_STORAGE_KEY = "llm-studio-tokenizer-active-job-id";
export const PREVIEW_TEXT_STORAGE_KEY = "llm-studio-tokenizer-preview-text";
export const HIDDEN_RECENT_JOB_IDS_STORAGE_KEY = "llm-studio-tokenizer-hidden-recent-job-ids";

export const TOKENIZER_STORAGE_KEY_MIGRATIONS = [
  {
    currentKey: TOKENIZER_FORM_STORAGE_KEY,
    legacyKey: "tokenizer-studio-tokenizer-form",
  },
  {
    currentKey: DATASET_FORM_STORAGE_KEY,
    legacyKey: "tokenizer-studio-dataset-form",
  },
  {
    currentKey: TRAINING_FORM_STORAGE_KEY,
    legacyKey: "tokenizer-studio-training-form",
  },
  {
    currentKey: ACTIVE_JOB_STORAGE_KEY,
    legacyKey: "tokenizer-studio-active-job-id",
  },
  {
    currentKey: PREVIEW_TEXT_STORAGE_KEY,
    legacyKey: "tokenizer-studio-preview-text",
  },
  {
    currentKey: HIDDEN_RECENT_JOB_IDS_STORAGE_KEY,
    legacyKey: "tokenizer-studio-hidden-recent-job-ids",
  },
] as const;

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
