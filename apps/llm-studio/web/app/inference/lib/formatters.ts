import { formatBytes } from "../../../lib/workspaceAssets";
import type { TrainingCheckpointEntry, TrainingJob } from "../../../lib/training/types";

export function formatInteger(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "0";
  }
  return Math.round(value).toLocaleString();
}

export function formatDate(value: string | null): string {
  if (!value) {
    return "not finished";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function completedArtifactName(job: TrainingJob): string {
  const name = job.name?.trim();
  if (name) {
    return name;
  }
  return job.artifact_bundle_file ?? `Training run ${job.id.slice(0, 8)}`;
}

export function formatJobMeta(job: TrainingJob): string {
  const pieces = [
    `step ${formatInteger(job.last_step)}`,
    `${formatInteger(job.checkpoint_count)} checkpoints`,
    `finished ${formatDate(job.finished_at)}`,
  ];
  return pieces.join(" | ");
}

export function matchesJobQuery(job: TrainingJob, normalizedQuery: string): boolean {
  if (normalizedQuery === "") {
    return true;
  }

  return [
    job.id,
    job.name,
    job.project_name,
    job.tokenizer_name,
    job.artifact_bundle_file,
    job.artifact_dir,
    job.stage,
  ]
    .filter((value): value is string => typeof value === "string")
    .some((value) => value.toLowerCase().includes(normalizedQuery));
}

export function checkpointOptionValue(checkpoint: TrainingCheckpointEntry): string {
  return String(checkpoint.step);
}

export function formatCheckpointName(checkpoint: TrainingCheckpointEntry): string {
  return `Step ${formatInteger(checkpoint.step)}`;
}

export function formatCheckpointMeta(checkpoint: TrainingCheckpointEntry): string {
  const pieces = [
    checkpoint.created_at ? `saved ${formatDate(checkpoint.created_at)}` : "saved time unavailable",
    formatBytes(checkpoint.size_bytes),
    `${formatInteger(checkpoint.files.length)} files`,
  ];
  return pieces.join(" | ");
}

export function matchesCheckpointQuery(
  checkpoint: TrainingCheckpointEntry,
  normalizedQuery: string
): boolean {
  if (normalizedQuery === "") {
    return true;
  }

  return [
    String(checkpoint.step),
    checkpoint.directory,
    checkpoint.created_at ?? "",
    String(checkpoint.size_bytes),
    ...checkpoint.files,
  ].some((value) => value.toLowerCase().includes(normalizedQuery));
}
