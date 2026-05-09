export type TrainingJobStatus = "pending" | "running" | "completed" | "failed" | "cancelled";
export type TrainingExecutorKind = "local" | "runpod_pod";
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
  executor_kind: TrainingExecutorKind;
  executor_status: string | null;
  runpod_pod_id: string | null;
  runpod_pod_name: string | null;
  runpod_network_volume_id: string | null;
  runpod_data_center_id: string | null;
  runpod_gpu_type_id: string | null;
  runpod_gpu_count: number;
  runpod_cloud_type: string | null;
  runpod_interruptible: boolean;
  runpod_cost_per_hr: number | null;
  runpod_public_ip: string | null;
  runpod_port_mappings: Record<string, unknown> | null;
  runpod_agent_base_url: string | null;
  runpod_last_heartbeat_at: string | null;
  runpod_last_sync_at: string | null;
  runpod_cleanup_policy: Record<string, unknown> | null;
  remote_workspace_path: string | null;
  remote_error: string | null;
}

export interface RunPodCleanupPolicy {
  pod: "delete_after_sync" | "stop_after_sync" | "keep";
  network_volume: "keep";
}

export interface TrainingExecutionTarget {
  kind: TrainingExecutorKind;
  api_key?: string | null;
  gpu_type_id?: string | null;
  gpu_count?: number | null;
  cloud_type?: "SECURE" | "COMMUNITY" | null;
  data_center_id?: string | null;
  interruptible?: boolean;
  network_volume_size_gb?: number | null;
  cleanup_policy?: RunPodCleanupPolicy;
}

export interface RunPodProviderDefaults {
  gpu_type_id: string;
  gpu_count: number;
  cloud_type: "SECURE" | "COMMUNITY";
  data_center_id: string | null;
  network_volume_size_gb: number;
  container_disk_gb: number;
  volume_mount_path: string;
  training_image: string;
  agent_port: number;
  agent_port_protocol: "tcp" | "http";
  cleanup_policy: RunPodCleanupPolicy;
}

export interface RunPodProviderStatus {
  configured: boolean;
  validated: boolean;
  source: "environment" | "memory" | "none";
  defaults: RunPodProviderDefaults;
}

export interface RunPodValidateKeyResponse {
  valid: boolean;
  message: string;
  account: Record<string, unknown> | null;
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
