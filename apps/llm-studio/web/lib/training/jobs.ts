import { request } from "./client";
import type {
  TrainingCheckpointEntry,
  TrainingConfigSchemas,
  TrainingConfigTemplates,
  TrainingDataPreview,
  TrainingExecutionTarget,
  TrainingJob,
  TrainingLogsResponse,
  TrainingMetricPoint,
  TrainingPreflightResponse,
  TrainingSampleEntry,
} from "./types";

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
    execution_target?: TrainingExecutionTarget;
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
