import {
  apiBaseUrl as runtimeApiBaseUrl,
  runtimeJsonRequest,
} from "./runtimeConfig";

export type JobStatus = "pending" | "running" | "completed" | "failed";
export type JobState =
  | "queued"
  | "initializing"
  | "preparing_dataset"
  | "training"
  | "saving_artifact"
  | "evaluating"
  | "running"
  | "completed"
  | "failed";
export type EvaluationSource = "training_dataset" | "legacy_file";

export interface TokenizerStats {
  num_records: number;
  num_chars: number;
  num_tokens: number;
  token_per_char: number;
  chars_per_token: number;
  avg_chars_per_record: number;
  avg_tokens_per_record: number;
  vocab_size: number;
  num_used_tokens: number;
  num_unused_tokens: number;
  rare_tokens: Record<string, number>;
  rare_token_fraction: Record<string, number>;
}

export interface TrainingJob {
  id: string;
  status: JobStatus;
  state: JobState;
  stage: string;
  progress: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  tokenizer_config: Record<string, unknown>;
  dataloader_config: Record<string, unknown>;
  evaluation_source: EvaluationSource;
  evaluation_thresholds: number[];
  evaluation_text_path: string;
  artifact_file: string | null;
  artifact_path: string | null;
  stats: TokenizerStats | null;
  error: string | null;
}

export interface ConfigTemplates {
  tokenizer_config_template: Record<string, unknown>;
  dataloader_config_template: Record<string, unknown>;
}

export interface TokenizerPreviewToken {
  index: number;
  id: number;
  token: string;
  start: number;
  end: number;
}

export interface TokenizerPreviewResult {
  job_id: string;
  text: string;
  text_length: number;
  num_tokens: number;
  tokens: TokenizerPreviewToken[];
}

export interface UploadedTrainFile {
  file_name: string;
  file_path: string;
  size_bytes: number;
  size_chars: number;
}

function resolveApiBaseUrl(): string {
  return normalizeApiBaseUrl(runtimeApiBaseUrl());
}

function normalizeApiBaseUrl(value: string): string {
  const trimmed = value === "/" ? "" : value.endsWith("/") ? value.slice(0, -1) : value;
  return trimmed.endsWith("/api/v1/tokenizer")
    ? trimmed
    : trimmed.endsWith("/api/v1")
      ? `${trimmed}/tokenizer`
      : `${trimmed}/api/v1/tokenizer`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  return runtimeJsonRequest<T>(`/tokenizer${path}`, init);
}

export function apiBaseUrl(): string {
  return resolveApiBaseUrl();
}

export async function fetchConfigTemplates(): Promise<ConfigTemplates> {
  return request<ConfigTemplates>("/config/templates");
}

export async function validateTokenizerConfig(
  config: Record<string, unknown>
): Promise<Record<string, unknown>> {
  const response = await request<{ normalized_config: Record<string, unknown> }>(
    "/validate/tokenizer",
    {
      method: "POST",
      body: JSON.stringify({ config }),
    }
  );
  return response.normalized_config;
}

export async function validateDataloaderConfig(
  config: Record<string, unknown>
): Promise<Record<string, unknown>> {
  const response = await request<{ normalized_config: Record<string, unknown> }>(
    "/validate/dataloader",
    {
      method: "POST",
      body: JSON.stringify({ config }),
    }
  );
  return response.normalized_config;
}

export async function uploadTrainFile(file: File): Promise<UploadedTrainFile> {
  const body = new FormData();
  body.append("file", file);
  return request<UploadedTrainFile>("/files/train", {
    method: "POST",
    body,
  });
}

export async function fetchLocalTrainFileStats(
  filePath: string
): Promise<UploadedTrainFile> {
  const params = new URLSearchParams({ file_path: filePath });
  return request<UploadedTrainFile>(`/files/stats?${params.toString()}`);
}

export async function createTrainingJob(payload: {
  tokenizer_config: Record<string, unknown>;
  dataloader_config: Record<string, unknown>;
  hf_token?: string | null;
  evaluation_thresholds: number[];
}): Promise<TrainingJob> {
  return request<TrainingJob>("/jobs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchTrainingJobs(): Promise<TrainingJob[]> {
  const response = await request<{ jobs: TrainingJob[] }>("/jobs");
  return response.jobs;
}

export async function fetchTrainingJob(jobId: string): Promise<TrainingJob> {
  return request<TrainingJob>(`/jobs/${jobId}`);
}

export async function deleteTrainingJob(jobId: string): Promise<void> {
  await request<void>(`/jobs/${jobId}`, {
    method: "DELETE",
  });
}

export async function previewJobTokenizer(
  jobId: string,
  payload: { text: string }
): Promise<TokenizerPreviewResult> {
  return request<TokenizerPreviewResult>(`/jobs/${jobId}/preview`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
