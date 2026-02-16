export type JobStatus = "pending" | "running" | "completed" | "failed";

export interface TokenizerStats {
  num_chars: number;
  num_tokens: number;
  token_per_char: number;
  vocab_size: number;
  num_used_tokens: number;
  num_unused_tokens: number;
  rare_tokens: Record<string, number>;
  rare_token_fraction: Record<string, number>;
}

export interface TrainingJob {
  id: string;
  status: JobStatus;
  stage: string;
  progress: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  tokenizer_config: Record<string, unknown>;
  dataloader_config: Record<string, unknown>;
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

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers ?? {});
  if (init?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
    cache: "no-store",
  });

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body?.detail)) {
        detail = body.detail
          .map((item: { loc?: unknown; msg?: unknown }) => {
            const location = Array.isArray(item?.loc)
              ? item.loc.join(".")
              : "unknown";
            const message =
              typeof item?.msg === "string" ? item.msg : "Validation error";
            return `${location}: ${message}`;
          })
          .join("; ");
      }
    } catch {
      // keep fallback detail
    }
    throw new Error(detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export function apiBaseUrl(): string {
  return API_BASE;
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

export async function createTrainingJob(payload: {
  tokenizer_config: Record<string, unknown>;
  dataloader_config: Record<string, unknown>;
  evaluation_thresholds: number[];
  evaluation_text_path?: string;
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

export function artifactDownloadUrl(jobId: string): string {
  return `${API_BASE}/jobs/${jobId}/artifact`;
}
