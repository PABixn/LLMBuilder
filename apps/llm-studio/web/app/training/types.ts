export type AssetPickerKind = "project" | "tokenizer";
export type DatasetSourceMode = "local_file" | "streaming_hf";
export type FilterOperator = "==" | "!=" | ">" | ">=" | "<" | "<=" | "in" | "not in";
export type WorkflowTarget = "model" | "tokenizer" | "training" | "dataset" | "preflight";
export type MetricChartKey = "loss" | "lr" | "norm" | "tok_per_sec";
export type MetricValueNotation = "standard" | "exponential";
export type ConfigNumberMode = "integer" | "decimal" | "scientific";
export type ToastLevel = "info" | "success" | "error";

export interface ToastState {
  id: string;
  level: ToastLevel;
  title: string;
  body: string;
}

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

export interface TrainingStepProgressSnapshot {
  completedSteps: number;
  maxSteps: number;
  fraction: number;
  percentLabel: string;
  elapsedSeconds: number | null;
  etaSeconds: number | null;
}
