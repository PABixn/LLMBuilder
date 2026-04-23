import type { TokenizerPreviewToken } from "../../lib/tokenizerLegacyApi";

export type TokenizerType = "bpe" | "wordpiece" | "unigram";
export type PreTokenizerType = "byte_level" | "whitespace" | "metaspace";
export type DecoderType = "byte_level" | "wordpiece" | "metaspace";
export type BudgetUnit = "chars" | "bytes";
export type BudgetBehavior = "stop" | "truncate";
export type DatasetSourceMode = "local_file" | "streaming_hf";
export type FilterOperator = "==" | "!=" | ">" | ">=" | "<" | "<=" | "in" | "not in";
export type SettingsCategory = "tokenizer" | "dataset" | "training";

export interface StreamingFilterFormState {
  id: string;
  column: string;
  operator: FilterOperator;
  value: string;
}

export interface StreamingDatasetFormState {
  id: string;
  name: string;
  config: string;
  split: string;
  textColumns: string;
  weight: string;
  filters: StreamingFilterFormState[];
}

export interface LocalTrainFileFormState {
  id: string;
  fileName: string;
  filePath: string;
  sizeBytes: number | null;
  sizeChars: number | null;
}

export interface TokenizerFormState {
  name: string;
  tokenizerType: TokenizerType;
  vocabSize: string;
  minFrequency: string;
  specialTokens: string;
  byteFallback: boolean;
  unkToken: string;
  preTokenizer: PreTokenizerType;
  decoder: DecoderType;
}

export interface DatasetFormState {
  sourceMode: DatasetSourceMode;
  localTrainFiles: LocalTrainFileFormState[];
  hfToken: string;
  streamingDatasets: StreamingDatasetFormState[];
}

export interface TrainingFormState {
  budgetLimit: string;
  budgetUnit: BudgetUnit;
  budgetBehavior: BudgetBehavior;
  evaluationThresholds: string;
}

export interface BuildResult {
  value: Record<string, unknown> | null;
  error: string | null;
}

export interface PreviewSegment {
  kind: "plain" | "token";
  text: string;
  token?: TokenizerPreviewToken;
}

export type ToastLevel = "info" | "success" | "error";

export interface ToastState {
  id: string;
  level: ToastLevel;
  message: string;
  durationMs: number;
}

export type TauriInvokeFn = (
  command: string,
  args?: Record<string, unknown>
) => Promise<unknown>;

export type JobBadgeTone =
  | "pending"
  | "setup"
  | "training"
  | "saving"
  | "evaluating"
  | "running"
  | "completed"
  | "failed";
