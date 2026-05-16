import { request } from "./client";
import type {
  RunPodProviderCatalog,
  RunPodProviderDefaults,
  RunPodProviderStatus,
  RunPodValidateKeyResponse,
} from "./types";

export async function fetchRunPodStatus(): Promise<RunPodProviderStatus> {
  return request<RunPodProviderStatus>("/providers/runpod/status");
}

export async function fetchRunPodDefaults(): Promise<RunPodProviderDefaults> {
  return request<RunPodProviderDefaults>("/providers/runpod/defaults");
}

export async function fetchRunPodCatalog(): Promise<RunPodProviderCatalog> {
  return request<RunPodProviderCatalog>("/providers/runpod/catalog");
}

export async function validateRunPodKey(apiKey: string): Promise<RunPodValidateKeyResponse> {
  return request<RunPodValidateKeyResponse>("/providers/runpod/validate-key", {
    method: "POST",
    body: JSON.stringify({ api_key: apiKey }),
  });
}
