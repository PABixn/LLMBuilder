import type { TrainingJob } from "../../../lib/tokenizerLegacyApi";
import type { JobBadgeTone } from "../types";

export function formatDate(value: string | null): string {
  if (!value) {
    return "-";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

export function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function describeJobState(state: TrainingJob["state"]): string {
  switch (state) {
    case "queued":
      return "Queued";
    case "initializing":
      return "Initializing";
    case "preparing_dataset":
      return "Preparing data";
    case "training":
      return "Training";
    case "saving_artifact":
      return "Saving artifact";
    case "evaluating":
      return "Evaluating";
    case "running":
      return "Running";
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    default:
      return "Running";
  }
}

export function jobBadgeTone(state: TrainingJob["state"]): JobBadgeTone {
  switch (state) {
    case "queued":
      return "pending";
    case "initializing":
    case "preparing_dataset":
      return "setup";
    case "training":
      return "training";
    case "saving_artifact":
      return "saving";
    case "evaluating":
      return "evaluating";
    case "running":
      return "running";
    case "completed":
      return "completed";
    case "failed":
      return "failed";
    default:
      return "running";
  }
}

export function evaluationSourceLabel(source: TrainingJob["evaluation_source"]): string {
  return source === "training_dataset"
    ? "Training dataset (same config)"
    : "Legacy external file";
}
