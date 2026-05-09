import { API_BASE } from "./client";

export function trainingArtifactDownloadUrl(jobId: string): string {
  return `${API_BASE}/jobs/${jobId}/artifact`;
}
