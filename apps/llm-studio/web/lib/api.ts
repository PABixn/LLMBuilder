import type { ModelConfig } from "./defaults";

export interface ValidationIssue {
  code: string;
  message: string;
  path: string;
}

export interface ModelValidationResponse {
  valid: boolean;
  normalized_config: ModelConfig;
  warnings: ValidationIssue[];
  errors: ValidationIssue[];
}

export interface ModelAnalysisSummary {
  total_parameters: number;
  trainable_parameters: number;
  parameter_memory_bytes_fp32: number;
  parameter_memory_bytes_bf16: number;
  estimated_kv_cache_bytes_per_token_fp16: number;
  estimated_kv_cache_bytes_for_context_fp16: number;
  block_count: number;
  component_count: number;
  attention_component_count: number;
  mlp_component_count: number;
  norm_component_count: number;
  activation_component_count: number;
  mlp_activation_step_count: number;
  min_head_dim: number | null;
  max_head_dim: number | null;
  instantiation_time_ms: number;
  module_counts: Record<string, number>;
}

export interface ModelAnalysisResponse {
  valid: boolean;
  normalized_config: ModelConfig;
  warnings: ValidationIssue[];
  errors: ValidationIssue[];
  instantiated: boolean;
  analysis: ModelAnalysisSummary | null;
  instantiation_error: string | null;
}

const API_BASE = resolveApiBaseUrl();
const RUNTIME_TOKEN =
  process.env.NEXT_PUBLIC_RUNTIME_TOKEN &&
  process.env.NEXT_PUBLIC_RUNTIME_TOKEN.trim() !== ""
    ? process.env.NEXT_PUBLIC_RUNTIME_TOKEN.trim()
    : null;

function resolveApiBaseUrl(): string {
  const explicit = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (explicit && explicit.trim() !== "") {
    return normalizeApiBaseUrl(explicit.trim());
  }

  if (process.env.NODE_ENV === "development") {
    return "http://127.0.0.1:8000/api/v1";
  }

  return "/api/v1";
}

function normalizeApiBaseUrl(value: string): string {
  if (value === "/") {
    return "";
  }
  return value.endsWith("/") ? value.slice(0, -1) : value;
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
          const message =
            typeof typed.msg === "string" ? typed.msg : "Validation error";
          return `${location}: ${message}`;
        })
        .join("; ");
    }
  } catch {
    // keep fallback detail
  }
  return detail;
}

function applyRuntimeHeaders(headers: Headers): void {
  if (RUNTIME_TOKEN && !headers.has("X-LLM-Studio-Token")) {
    headers.set("X-LLM-Studio-Token", RUNTIME_TOKEN);
  }
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
    throw new Error(await readErrorDetail(response));
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseValidationIssue(value: unknown): ValidationIssue | null {
  if (!isRecord(value)) {
    return null;
  }
  const code = typeof value.code === "string" ? value.code : "unknown_issue";
  const message =
    typeof value.message === "string" ? value.message : "Unknown validation issue";
  const path = typeof value.path === "string" ? value.path : "$";
  return { code, message, path };
}

function parseValidationIssues(value: unknown): ValidationIssue[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map(parseValidationIssue)
    .filter((item): item is ValidationIssue => item !== null);
}

function parseModelValidationResponse(raw: unknown): ModelValidationResponse {
  if (!isRecord(raw)) {
    return {
      valid: false,
      normalized_config: raw as ModelConfig,
      warnings: [],
      errors: [],
    };
  }

  return {
    valid: typeof raw.valid === "boolean" ? raw.valid : true,
    normalized_config: raw.normalized_config as ModelConfig,
    warnings: parseValidationIssues(raw.warnings),
    errors: parseValidationIssues(raw.errors),
  };
}

function isModelAnalysisSummary(value: unknown): value is ModelAnalysisSummary {
  return (
    isRecord(value) &&
    typeof value.total_parameters === "number" &&
    typeof value.trainable_parameters === "number"
  );
}

export function apiBaseUrl(): string {
  return API_BASE;
}

export async function validateModelConfig(
  config: ModelConfig,
  signal?: AbortSignal
): Promise<ModelValidationResponse> {
  const raw = await request<unknown>("/validate/model", {
    method: "POST",
    body: JSON.stringify({ config }),
    signal,
  });

  return parseModelValidationResponse(raw);
}

export async function analyzeModelConfig(
  config: ModelConfig,
  signal?: AbortSignal
): Promise<ModelAnalysisResponse> {
  const raw = await request<unknown>("/analyze/model", {
    method: "POST",
    body: JSON.stringify({ config }),
    signal,
  });

  if (!isRecord(raw)) {
    return {
      valid: false,
      normalized_config: config,
      warnings: [],
      errors: [],
      instantiated: false,
      analysis: null,
      instantiation_error: "Malformed backend analysis response.",
    };
  }

  const validation = parseModelValidationResponse(raw);
  const analysis = isModelAnalysisSummary(raw.analysis) ? raw.analysis : null;

  return {
    ...validation,
    instantiated: raw.instantiated === true,
    analysis,
    instantiation_error:
      typeof raw.instantiation_error === "string" ? raw.instantiation_error : null,
  };
}
