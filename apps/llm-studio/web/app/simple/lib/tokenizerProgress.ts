import type { TrainingJob as TokenizerJob } from "../../../lib/tokenizerLegacyApi";

type TokenizerProgressPillState = "running" | "completed" | "failed";

export interface SimpleTokenizerProgressState {
  pillLabel: string;
  pillState: TokenizerProgressPillState;
  headline: string;
  detail: string;
  progress: number;
  progressLabel: string;
  statusLabel: string;
  recordsLabel: string;
  tokensLabel: string;
  vocabLabel: string;
}

interface BuildTokenizerProgressOptions {
  job: TokenizerJob | null;
  validating: boolean;
  starting: boolean;
}

function clampProgress(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function formatState(value: string | null | undefined): string {
  if (!value) {
    return "Starting";
  }
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatNumber(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString()
    : "Pending";
}

function jobProgress(job: TokenizerJob): number {
  if (job.status === "completed") {
    return 1;
  }
  if (job.status === "failed") {
    return clampProgress(job.progress);
  }
  const minimumVisibleProgress = job.status === "pending" ? 0.08 : 0.12;
  return Math.max(minimumVisibleProgress, clampProgress(job.progress));
}

function progressLabel(progress: number): string {
  return `${Math.round(clampProgress(progress) * 100)}%`;
}

export function buildSimpleTokenizerProgressState({
  job,
  starting,
  validating,
}: BuildTokenizerProgressOptions): SimpleTokenizerProgressState | null {
  if (!job) {
    if (validating) {
      return {
        pillLabel: "Checking",
        pillState: "running",
        headline: "Checking tokenizer inputs",
        detail: "Validating tokenizer and dataset settings before the job starts.",
        progress: 0.04,
        progressLabel: "4%",
        statusLabel: "Checking",
        recordsLabel: "Pending",
        tokensLabel: "Pending",
        vocabLabel: "Pending",
      };
    }
    if (starting) {
      return {
        pillLabel: "Starting",
        pillState: "running",
        headline: "Starting tokenizer job",
        detail: "Submitting the tokenizer job and waiting for the first backend update.",
        progress: 0.08,
        progressLabel: "8%",
        statusLabel: "Starting",
        recordsLabel: "Pending",
        tokensLabel: "Pending",
        vocabLabel: "Pending",
      };
    }
    return null;
  }

  const progress = jobProgress(job);
  const stats = job.stats;
  const base = {
    progress,
    progressLabel: progressLabel(progress),
    statusLabel: formatState(job.status),
    recordsLabel: formatNumber(stats?.num_records),
    tokensLabel: formatNumber(stats?.num_tokens),
    vocabLabel: formatNumber(stats?.vocab_size),
  };

  if (job.status === "completed") {
    return {
      ...base,
      pillLabel: "Tokenizer ready",
      pillState: "completed",
      headline: "Tokenizer artifact ready",
      detail: "The trained tokenizer is ready and the model vocabulary can stay synced.",
    };
  }

  if (job.status === "failed") {
    return {
      ...base,
      pillLabel: "Failed",
      pillState: "failed",
      headline: "Tokenizer job failed",
      detail: job.error || "The backend stopped the tokenizer job before producing an artifact.",
    };
  }

  return {
    ...base,
    pillLabel: job.status === "pending" ? "Queued" : "Training",
    pillState: "running",
    headline: job.status === "pending" ? "Tokenizer job queued" : "Tokenizer training in progress",
    detail:
      job.status === "pending"
        ? "The backend accepted the job and will start processing shortly."
        : "Progress updates automatically while the backend trains and saves the tokenizer.",
  };
}
