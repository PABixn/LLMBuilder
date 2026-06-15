import {
  bootstrapDesktopRuntime,
  isDesktopShell,
  requestDesktopRuntime,
  retryDesktopRuntime,
  type DesktopRuntimeBootstrap,
} from "./desktopBridge";

export type RuntimeEnvironment = "web" | "desktop";

export type RuntimeConfig = {
  environment: RuntimeEnvironment;
  apiBaseUrl: string;
  runtimeToken: string | null;
  capabilities: DesktopRuntimeBootstrap["capabilities"];
  versions: Record<string, string>;
};

export class RuntimeUnavailableError extends Error {
  constructor(message = "The LLM Studio runtime is unavailable.") {
    super(message);
    this.name = "RuntimeUnavailableError";
  }
}

export class RuntimeRequestAbortedError extends Error {
  constructor(message = "The LLM Studio request was cancelled.") {
    super(message);
    this.name = "RuntimeRequestAbortedError";
  }
}

export class RuntimeHttpError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "RuntimeHttpError";
    this.status = status;
  }
}

const EMPTY_CAPABILITIES: RuntimeConfig["capabilities"] = {
  native_save: false,
  open_logs: false,
  open_data: false,
  reveal_artifact: false,
  diagnostics_export: false,
};

let currentConfig = webRuntimeConfig();
let initialization: Promise<RuntimeConfig> | null = null;

export function getRuntimeConfig(): RuntimeConfig {
  return currentConfig;
}

export function apiBaseUrl(): string {
  return currentConfig.apiBaseUrl;
}

export async function initializeRuntimeConfig(options?: {
  retry?: boolean;
}): Promise<RuntimeConfig> {
  if (!isDesktopShell()) {
    currentConfig = webRuntimeConfig();
    return currentConfig;
  }
  if (initialization && !options?.retry) {
    return initialization;
  }

  initialization = (options?.retry ? retryDesktopRuntime() : bootstrapDesktopRuntime())
    .then((bootstrap) => {
      currentConfig = {
        environment: "desktop",
        apiBaseUrl: normalizeApiBaseUrl(bootstrap.api_base_url),
        runtimeToken: bootstrap.runtime_token,
        capabilities: bootstrap.capabilities,
        versions: bootstrap.versions,
      };
      return currentConfig;
    })
    .catch((error: unknown) => {
      initialization = null;
      currentConfig = desktopUnavailableConfig();
      throw new RuntimeUnavailableError(
        error instanceof Error && error.message.trim()
          ? error.message
          : "Desktop runtime bootstrap failed."
      );
    });

  return initialization;
}

export function runtimeHeaders(initial?: HeadersInit): Headers {
  const headers = new Headers(initial ?? {});
  if (currentConfig.runtimeToken && !headers.has("X-LLM-Studio-Token")) {
    headers.set("X-LLM-Studio-Token", currentConfig.runtimeToken);
  }
  return headers;
}

export function runtimeApiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${currentConfig.apiBaseUrl}${normalizedPath}`;
}

export async function runtimeFetch(path: string, init?: RequestInit): Promise<Response> {
  if (currentConfig.environment === "desktop" && !currentConfig.runtimeToken) {
    throw new RuntimeUnavailableError("Desktop runtime token is unavailable.");
  }
  if (currentConfig.environment === "desktop") {
    return desktopRuntimeFetch(path, init);
  }
  return fetch(runtimeApiUrl(path), {
    ...init,
    headers: runtimeHeaders(init?.headers),
    cache: "no-store",
  });
}

async function desktopRuntimeFetch(path: string, init?: RequestInit): Promise<Response> {
  if (init?.signal?.aborted) {
    throw abortError();
  }
  const headers = new Headers(init?.headers ?? {});
  headers.delete("X-LLM-Studio-Token");
  const body = await serializeDesktopRequestBody(init?.body, headers);
  const response = await requestDesktopRuntime({
    method: init?.method?.toUpperCase() || "GET",
    path: runtimeApiUrl(path).slice(currentConfig.apiBaseUrl.length),
    headers: Object.fromEntries(headers.entries()),
    body: Array.from(body),
  });
  if (init?.signal?.aborted) {
    throw abortError();
  }
  const responseBody =
    response.status === 204 || response.status === 205 || response.status === 304
      ? null
      : new Uint8Array(response.body);
  return new Response(responseBody, {
    status: response.status,
    headers: response.headers,
  });
}

async function serializeDesktopRequestBody(
  body: BodyInit | null | undefined,
  headers: Headers
): Promise<Uint8Array> {
  if (body == null) {
    return new Uint8Array();
  }
  const materialized = new Response(body);
  const contentType = materialized.headers.get("Content-Type");
  if (contentType && !headers.has("Content-Type")) {
    headers.set("Content-Type", contentType);
  }
  return new Uint8Array(await materialized.arrayBuffer());
}

function abortError(): Error {
  const error = new Error("The LLM Studio request was cancelled.");
  error.name = "AbortError";
  return error;
}

export async function runtimeRequest(path: string, init?: RequestInit): Promise<Response> {
  try {
    return await runtimeFetch(path, init);
  } catch (error: unknown) {
    if (error instanceof RuntimeUnavailableError) {
      throw error;
    }
    if (isAbortError(error)) {
      throw new RuntimeRequestAbortedError();
    }
    throw new RuntimeUnavailableError(
      currentConfig.environment === "desktop"
        ? "The local LLM Studio backend is unavailable."
        : "The LLM Studio API is unavailable."
    );
  }
}

export async function readRuntimeErrorDetail(response: Response): Promise<string> {
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
    // Keep the status-only fallback for non-JSON or malformed error responses.
  }
  return detail;
}

export async function runtimeJsonRequest<T>(
  path: string,
  init?: RequestInit,
  errorFactory: (message: string, status: number) => Error = (message, status) =>
    new RuntimeHttpError(message, status)
): Promise<T> {
  const headers = new Headers(init?.headers ?? {});
  const hasFormDataBody =
    typeof FormData !== "undefined" && init?.body instanceof FormData;
  if (init?.body && !hasFormDataBody && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await runtimeRequest(path, {
    ...init,
    headers,
    cache: "no-store",
  });

  if (!response.ok) {
    throw errorFactory(await readRuntimeErrorDetail(response), response.status);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  try {
    return (await response.json()) as T;
  } catch {
    throw errorFactory(
      `The LLM Studio runtime returned invalid JSON (${response.status}).`,
      response.status
    );
  }
}

export function __setRuntimeConfigForTests(config: RuntimeConfig): void {
  currentConfig = config;
  initialization = null;
}

export function __resetRuntimeConfigForTests(): void {
  currentConfig = webRuntimeConfig();
  initialization = null;
}

function webRuntimeConfig(): RuntimeConfig {
  return {
    environment: "web",
    apiBaseUrl: resolveWebApiBaseUrl(),
    runtimeToken:
      process.env.NEXT_PUBLIC_RUNTIME_TOKEN?.trim() || null,
    capabilities: EMPTY_CAPABILITIES,
    versions: {},
  };
}

function desktopUnavailableConfig(): RuntimeConfig {
  return {
    environment: "desktop",
    apiBaseUrl: "",
    runtimeToken: null,
    capabilities: EMPTY_CAPABILITIES,
    versions: {},
  };
}

function resolveWebApiBaseUrl(): string {
  const explicit = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (explicit) {
    return normalizeApiBaseUrl(explicit);
  }
  return process.env.NODE_ENV === "development"
    ? "http://127.0.0.1:8000/api/v1"
    : "/api/v1";
}

function normalizeApiBaseUrl(value: string): string {
  if (value === "/") {
    return "";
  }
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

function isAbortError(error: unknown): boolean {
  return (
    error instanceof Error &&
    error.name === "AbortError"
  );
}
