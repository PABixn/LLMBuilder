import { TrainingApiError } from "./errors";
import {
  apiBaseUrl,
  readRuntimeErrorDetail,
  runtimeJsonRequest,
} from "../runtimeConfig";

export function trainingApiBaseUrl(): string {
  const base = apiBaseUrl();
  const trimmed = base === "/" ? "" : base.endsWith("/") ? base.slice(0, -1) : base;
  return trimmed.endsWith("/api/v1/training")
    ? trimmed
    : trimmed.endsWith("/api/v1")
      ? `${trimmed}/training`
      : `${trimmed}/api/v1/training`;
}

export const readErrorDetail = readRuntimeErrorDetail;

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  return runtimeJsonRequest<T>(
    `/training${path}`,
    init,
    (message, status) => new TrainingApiError(message, status)
  );
}
