import { apiBaseUrl } from "../api";
import { TrainingApiError } from "./errors";

export const API_BASE = resolveApiBaseUrl();

const RUNTIME_TOKEN =
  process.env.NEXT_PUBLIC_RUNTIME_TOKEN &&
  process.env.NEXT_PUBLIC_RUNTIME_TOKEN.trim() !== ""
    ? process.env.NEXT_PUBLIC_RUNTIME_TOKEN.trim()
    : null;

function resolveApiBaseUrl(): string {
  const base = apiBaseUrl();
  const trimmed = base === "/" ? "" : base.endsWith("/") ? base.slice(0, -1) : base;
  return trimmed.endsWith("/api/v1/training")
    ? trimmed
    : trimmed.endsWith("/api/v1")
      ? `${trimmed}/training`
      : `${trimmed}/api/v1/training`;
}

export function applyRuntimeHeaders(headers: Headers): void {
  if (RUNTIME_TOKEN && !headers.has("X-LLM-Studio-Token")) {
    headers.set("X-LLM-Studio-Token", RUNTIME_TOKEN);
  }
}

export async function readErrorDetail(response: Response): Promise<string> {
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

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
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
