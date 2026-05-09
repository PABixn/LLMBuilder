import { API_BASE, applyRuntimeHeaders, readErrorDetail, request } from "./client";
import type {
  GenerateTrainingCompletionRequest,
  GenerateTrainingCompletionResponse,
  GenerateTrainingCompletionStreamEvent,
} from "./types";

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
