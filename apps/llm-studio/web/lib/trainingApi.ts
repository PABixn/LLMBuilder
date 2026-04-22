import { apiBaseUrl } from "./api";

export type TrainingJobStatus = "pending" | "running" | "completed" | "failed" | "cancelled";
export type TrainingJobState =
  | "queued"
  | "preflight"
  | "estimating_memory"
  | "initializing_model"
  | "building_dataloader"
  | "training"
  | "checkpointing"
  | "sampling"
  | "finalizing"
  | "completed"
  | "failed"
  | "cancelled";

export interface TrainingAssetRef {
  id: string;
  name: string;
  artifact_path?: string | null;
  artifact_file?: string | null;
  status?: string | null;
}

export interface TrainingIssue {
  code: string;
  message: string;
  path: string;
  severity: string;
}

export interface TrainingFixSuggestion {
  code: string;
  label: string;
  description: string;
  path: string;
  value?: unknown;
}

export interface TrainingCompatibilitySummary {
  model_context_length: number;
  model_vocab_size: number;
  tokenizer_vocab_size: number;
  seq_len: number;
  scheduler_total_steps: number;
  max_steps: number;
  missing_special_tokens: string[];
}

export interface DerivedRuntimeSummary {
  device: string;
  device_type: string;
  micro_batch_size: number;
  tokens_per_micro_step: number;
  tokens_per_world_step: number;
  grad_accum_steps: number;
  max_batch_size_from_total: number;
  max_batch_size_from_memory: number;
  max_allowed_batch_size: number;
  ddp: boolean;
  ddp_rank: number;
  ddp_world_size: number;
}

export interface TrainingBatchLrRecommendationOption {
  key: string;
  label: string;
  description: string;
  tone: "recommended" | "neutral";
  total_batch_size: number;
  micro_batch_size: number;
  grad_accum_steps: number;
  learning_rate: number;
  estimated_tokens_per_run: number;
  clear_manual_micro_batch: boolean;
}

export interface TrainingBatchLrRecommendationFactor {
  code: string;
  label: string;
  detail: string;
  tone: "good" | "neutral" | "warning";
}

export interface TrainingBatchLrRecommendationSignals {
  device: string;
  device_type: string;
  total_parameters: number;
  parameter_memory_bytes_bf16: number;
  estimated_kv_cache_bytes_for_context_fp16: number;
  block_count: number;
  attention_component_count: number;
  max_mlp_multiplier: number;
  dataset_count: number;
  local_dataset_count: number;
  streaming_dataset_count: number;
  local_file_count: number;
  local_total_size_bytes: number | null;
  dominant_dataset_weight: number;
  dataset_scale: string;
  schedule_peak_factor: number;
  warmup_fraction: number;
  max_memory_micro_batch_size: number;
  recommended_batch_target: number;
}

export interface TrainingBatchLrRecommendation {
  headline: string;
  summary: string;
  confidence: "high" | "medium" | "low";
  current_total_batch_size: number;
  current_learning_rate: number;
  current_micro_batch_size: number | null;
  current_grad_accum_steps: number | null;
  recommended_option_key: string;
  options: TrainingBatchLrRecommendationOption[];
  factors: TrainingBatchLrRecommendationFactor[];
  signals: TrainingBatchLrRecommendationSignals;
}

export interface TrainingPreflightResponse {
  valid: boolean;
  model_project: TrainingAssetRef;
  tokenizer: TrainingAssetRef;
  normalized_training_config: Record<string, unknown>;
  normalized_dataloader_config: Record<string, unknown>;
  warnings: TrainingIssue[];
  errors: TrainingIssue[];
  recommended_fixes: TrainingFixSuggestion[];
  compatibility: TrainingCompatibilitySummary | null;
  derived_runtime: DerivedRuntimeSummary | null;
  memory_estimate: Record<string, unknown> | null;
  batch_and_lr_recommendation: TrainingBatchLrRecommendation | null;
}

export interface TrainingMetricPoint {
  step: number;
  loss?: number | null;
  norm?: number | null;
  dt?: number | null;
  tok_per_sec?: number | null;
  lr?: number | null;
}

export interface TrainingSampleText {
  index: number;
  prompt?: string | null;
  text: string;
}

export interface TrainingSampleEntry {
  step: number;
  samples: TrainingSampleText[];
}

export interface TrainingCheckpointEntry {
  step: number;
  directory: string;
  created_at?: string | null;
  size_bytes: number;
  files: string[];
}

export interface TrainingLogsResponse {
  job_id: string;
  stdout_lines: string[];
  stderr_lines: string[];
}

export type TrainingDataPreview = Record<string, unknown>;

export interface TrainingJob {
  id: string;
  name: string;
  status: TrainingJobStatus;
  state: TrainingJobState;
  stage: string;
  progress: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  project_id: string;
  project_name: string;
  tokenizer_job_id: string;
  tokenizer_name: string;
  model_config: Record<string, unknown>;
  training_config: Record<string, unknown>;
  dataloader_config: Record<string, unknown>;
  resolved_runtime: Record<string, unknown> | null;
  memory_estimate: Record<string, unknown> | null;
  artifact_dir: string;
  artifact_bundle_file: string | null;
  stats_path: string;
  samples_path: string;
  stdout_path: string;
  stderr_path: string;
  last_step: number;
  max_steps: number;
  elapsed_seconds?: number | null;
  eta_seconds?: number | null;
  latest_loss: number | null;
  latest_grad_norm: number | null;
  latest_lr: number | null;
  latest_tokens_per_sec: number | null;
  checkpoint_count: number;
  sample_count: number;
  error: string | null;
  process_id: number | null;
  output_size_bytes: number;
}

export interface GenerateTrainingCompletionRequest {
  prompt: string;
  checkpoint_step?: number | null;
  max_tokens: number;
  temperature: number;
  top_k: number | null;
  seed: number;
  repetition_penalty: number;
}

export interface GenerateTrainingCompletionResponse {
  job_id: string;
  checkpoint_step: number;
  checkpoint_path: string;
  tokenizer_job_id: string;
  prompt: string;
  completion: string;
  text: string;
  prompt_token_count: number;
  generated_token_count: number;
  generated_token_ids: number[];
}

export type GenerateTrainingCompletionStreamEvent =
  | {
      type: "start";
      job_id: string;
      checkpoint_step: number;
      checkpoint_path: string;
      tokenizer_job_id: string;
      prompt: string;
      prompt_token_count: number;
    }
  | {
      type: "token";
      index: number;
      token_id: number;
      token_text: string;
    }
  | {
      type: "done";
      completion: string;
      text: string;
      generated_token_count: number;
      generated_token_ids: number[];
    }
  | {
      type: "error";
      detail: string;
    };

export interface TrainingConfigTemplates {
  training_config_template: Record<string, unknown>;
  dataloader_config_template: Record<string, unknown>;
}

export interface TrainingConfigSchemas {
  training_config_schema: Record<string, unknown>;
  dataloader_schema: Record<string, unknown>;
}

const API_BASE = resolveApiBaseUrl();
const RUNTIME_TOKEN =
  process.env.NEXT_PUBLIC_RUNTIME_TOKEN &&
  process.env.NEXT_PUBLIC_RUNTIME_TOKEN.trim() !== ""
    ? process.env.NEXT_PUBLIC_RUNTIME_TOKEN.trim()
    : null;

export class TrainingApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "TrainingApiError";
    this.status = status;
  }
}

function resolveApiBaseUrl(): string {
  const base = apiBaseUrl();
  const trimmed = base === "/" ? "" : base.endsWith("/") ? base.slice(0, -1) : base;
  return trimmed.endsWith("/api/v1/training")
    ? trimmed
    : trimmed.endsWith("/api/v1")
      ? `${trimmed}/training`
      : `${trimmed}/api/v1/training`;
}

function applyRuntimeHeaders(headers: Headers): void {
  if (RUNTIME_TOKEN && !headers.has("X-LLM-Studio-Token")) {
    headers.set("X-LLM-Studio-Token", RUNTIME_TOKEN);
  }
}

async function readErrorDetail(response: Response): Promise<string> {
  let detail = `Request failed (${response.status})`;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") {
      detail = body.detail;
    } else if (Array.isArray(body.detail)) {
      detail = body.detail
        .map((item) => {
          if (!item || typeof item !== "object") {
            return "Validation error";
          }
          const typed = item as { loc?: unknown; msg?: unknown };
          const location = Array.isArray(typed.loc)
            ? typed.loc.map(String).join(".")
            : "unknown";
          const message = typeof typed.msg === "string" ? typed.msg : "Validation error";
          return `${location}: ${message}`;
        })
        .join("; ");
    }
  } catch {
    // keep fallback detail
  }
  return detail;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers ?? {});
  const hasFormDataBody =
    typeof FormData !== "undefined" && init?.body instanceof FormData;
  if (init?.body && !hasFormDataBody && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  applyRuntimeHeaders(headers);

  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
    cache: "no-store",
  });

  if (!response.ok) {
    throw new TrainingApiError(await readErrorDetail(response), response.status);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export function trainingArtifactDownloadUrl(jobId: string): string {
  return `${API_BASE}/jobs/${jobId}/artifact`;
}

export async function fetchTrainingConfigTemplates(): Promise<TrainingConfigTemplates> {
  return request<TrainingConfigTemplates>("/config/templates");
}

export async function fetchTrainingConfigSchemas(): Promise<TrainingConfigSchemas> {
  return request<TrainingConfigSchemas>("/config/schemas");
}

export async function validateTrainingConfig(
  config: Record<string, unknown>,
  signal?: AbortSignal
): Promise<Record<string, unknown>> {
  const response = await request<{ normalized_config: Record<string, unknown> }>(
    "/validate/training-config",
    {
      method: "POST",
      body: JSON.stringify({ config }),
      signal,
    }
  );
  return response.normalized_config;
}

export async function validateTrainingDataloader(
  config: Record<string, unknown>,
  signal?: AbortSignal
): Promise<Record<string, unknown>> {
  const response = await request<{ normalized_config: Record<string, unknown> }>(
    "/validate/dataloader",
    {
      method: "POST",
      body: JSON.stringify({ config }),
      signal,
    }
  );
  return response.normalized_config;
}

export async function validateTrainingPreflight(
  payload: {
    project_id: string;
    tokenizer_job_id: string;
    training_config: Record<string, unknown>;
    dataloader_config: Record<string, unknown>;
  },
  signal?: AbortSignal
): Promise<TrainingPreflightResponse> {
  return request<TrainingPreflightResponse>("/validate/preflight", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  });
}

export async function createTrainingJob(
  payload: {
    project_id: string;
    tokenizer_job_id: string;
    training_config: Record<string, unknown>;
    dataloader_config: Record<string, unknown>;
    name?: string | null;
  }
): Promise<TrainingJob> {
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

export async function fetchTrainingMetrics(jobId: string, limit?: number): Promise<TrainingMetricPoint[]> {
  const params = new URLSearchParams();
  if (typeof limit === "number") {
    params.set("limit", String(limit));
  }
  const response = await request<{ job_id: string; metrics: TrainingMetricPoint[] }>(
    `/jobs/${jobId}/metrics${params.size ? `?${params.toString()}` : ""}`
  );
  return response.metrics;
}

export async function fetchTrainingSamples(jobId: string, limit = 50): Promise<TrainingSampleEntry[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  const response = await request<{ job_id: string; samples: TrainingSampleEntry[] }>(
    `/jobs/${jobId}/samples?${params.toString()}`
  );
  return response.samples;
}

export async function fetchTrainingLogs(jobId: string, lines?: number): Promise<TrainingLogsResponse> {
  const params = new URLSearchParams();
  if (typeof lines === "number") {
    params.set("lines", String(lines));
  }
  const query = params.toString();
  return request<TrainingLogsResponse>(`/jobs/${jobId}/logs${query ? `?${query}` : ""}`);
}

export async function fetchTrainingDataPreview(jobId: string): Promise<TrainingDataPreview> {
  return request<TrainingDataPreview>(`/jobs/${jobId}/data-preview`);
}

export async function fetchTrainingCheckpoints(jobId: string): Promise<TrainingCheckpointEntry[]> {
  const response = await request<{ job_id: string; checkpoints: TrainingCheckpointEntry[] }>(
    `/jobs/${jobId}/checkpoints`
  );
  return response.checkpoints;
}

export async function generateTrainingCompletion(
  jobId: string,
  payload: GenerateTrainingCompletionRequest
): Promise<GenerateTrainingCompletionResponse> {
  return request<GenerateTrainingCompletionResponse>(`/jobs/${jobId}/generate`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function streamTrainingCompletion(
  jobId: string,
  payload: GenerateTrainingCompletionRequest,
  onEvent: (event: GenerateTrainingCompletionStreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const headers = new Headers();
  headers.set("Content-Type", "application/json");
  applyRuntimeHeaders(headers);

  const response = await fetch(`${API_BASE}/jobs/${jobId}/generate/stream`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
    cache: "no-store",
    signal,
  });

  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }
  if (!response.body) {
    throw new Error("Inference stream did not include a response body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      const event = JSON.parse(trimmed) as GenerateTrainingCompletionStreamEvent;
      if (event.type === "error") {
        throw new Error(event.detail);
      }
      onEvent(event);
    }

    if (done) {
      break;
    }
  }

  const trailing = buffer.trim();
  if (trailing) {
    const event = JSON.parse(trailing) as GenerateTrainingCompletionStreamEvent;
    if (event.type === "error") {
      throw new Error(event.detail);
    }
    onEvent(event);
  }
}

export async function stopTrainingJob(jobId: string): Promise<TrainingJob> {
  return request<TrainingJob>(`/jobs/${jobId}/stop`, {
    method: "POST",
  });
}

export async function deleteTrainingJob(jobId: string): Promise<void> {
  await request<void>(`/jobs/${jobId}`, {
    method: "DELETE",
  });
}
