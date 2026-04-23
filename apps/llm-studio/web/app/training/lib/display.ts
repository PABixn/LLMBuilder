import type {
  TrainingIssue,
  TrainingJob,
  TrainingJobStatus,
} from "../../../lib/trainingApi";
import type { ProjectDetail } from "../../../lib/api";
import type { TrainingJob as TokenizerTrainingJob } from "../../../lib/tokenizerLegacyApi";
import { formatInteger } from "./metrics";
import type { TrainingStepProgressSnapshot } from "../types";

const TERMINAL_TRAINING_STATUSES = new Set<TrainingJobStatus>([
  "completed",
  "failed",
  "cancelled",
]);

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asRecord(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

function asRecordArray(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is Record<string, unknown> => isRecord(item));
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function asNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function humanizeConfigToken(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\bbpe\b/i, "BPE")
    .replace(/\bwordpiece\b/i, "WordPiece")
    .replace(/\bunigram\b/i, "Unigram");
}

function firstAttentionConfig(modelConfig: Record<string, unknown>): Record<string, unknown> {
  const blocks = Array.isArray(modelConfig.blocks) ? modelConfig.blocks : [];
  for (const block of blocks) {
    const components = asRecordArray(asRecord(block).components);
    for (const component of components) {
      const attention = asRecord(component.attention);
      if (Object.keys(attention).length > 0) {
        return attention;
      }
    }
  }
  return {};
}

export function formatModelConfigMeta(value: unknown): string {
  const modelConfig = asRecord(value);
  const blocks = Array.isArray(modelConfig.blocks) ? modelConfig.blocks : [];
  const attention = firstAttentionConfig(modelConfig);
  const parts: string[] = [];
  if (blocks.length > 0) {
    parts.push(`${formatInteger(blocks.length)} layers`);
  }
  const headCount = asNumber(attention.n_head, 0);
  if (headCount > 0) {
    parts.push(`${formatInteger(headCount)} attention heads`);
  }
  const embeddingSize = asNumber(modelConfig.n_embd, 0);
  if (embeddingSize > 0) {
    parts.push(`${formatInteger(embeddingSize)} embedding width`);
  }
  const contextLength = asNumber(modelConfig.context_length, 0);
  if (contextLength > 0) {
    parts.push(`${formatInteger(contextLength)} context length`);
  }
  const vocabSize = asNumber(modelConfig.vocab_size, 0);
  if (vocabSize > 0) {
    parts.push(`${formatInteger(vocabSize)} vocabulary size`);
  }
  return parts.length > 0 ? parts.join(" • ") : "Model dimensions unavailable";
}

export function formatTokenizerMeta(job: TokenizerTrainingJob): string {
  const config = job.tokenizer_config;
  const stats = job.stats;
  const parts: string[] = [];
  const vocabSize = stats?.vocab_size ?? asNumber(config.vocab_size, 0);
  if (vocabSize > 0) {
    parts.push(`${formatInteger(vocabSize)} vocabulary size`);
  }
  const tokenizerType = asString(config.tokenizer_type);
  const preTokenizer = asString(config.pre_tokenizer);
  if (tokenizerType || preTokenizer) {
    parts.push([tokenizerType, preTokenizer].filter(Boolean).map(humanizeConfigToken).join(" / "));
  }
  if (typeof stats?.chars_per_token === "number" && Number.isFinite(stats.chars_per_token)) {
    parts.push(`${stats.chars_per_token.toFixed(2)} characters per token`);
  }
  const specialTokenCount = Array.isArray(config.special_tokens) ? config.special_tokens.length : 0;
  if (specialTokenCount > 0) {
    parts.push(`${formatInteger(specialTokenCount)} special tokens`);
  }
  return parts.length > 0 ? parts.join(" • ") : "Tokenizer details unavailable";
}

export function formatDuration(seconds: number | null | undefined): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) {
    return "n/a";
  }
  const whole = Math.floor(seconds);
  const hrs = Math.floor(whole / 3600);
  const mins = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;
  if (hrs > 0) {
    return `${hrs}h ${mins}m`;
  }
  if (mins > 0) {
    return `${mins}m ${secs}s`;
  }
  return `${secs}s`;
}

function normalizeRuntimeSeconds(value: number | null | undefined): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return null;
  }
  return value;
}

function formatProgressPercent(fraction: number): string {
  const percentage = clamp(fraction * 100, 0, 100);
  if (percentage <= 0 || percentage >= 100) {
    return `${Math.round(percentage)}%`;
  }
  if (percentage < 10) {
    return `${percentage.toFixed(1)}%`;
  }
  return `${Math.round(percentage)}%`;
}

export function deriveTrainingStepProgress(run: TrainingJob | null): TrainingStepProgressSnapshot {
  const maxSteps = Math.max(0, run?.max_steps ?? 0);
  const completedSteps =
    maxSteps > 0
      ? clamp(Math.max(0, run?.last_step ?? 0), 0, maxSteps)
      : Math.max(0, run?.last_step ?? 0);
  const fraction =
    maxSteps > 0 ? clamp(completedSteps / maxSteps, 0, 1) : run?.status === "completed" ? 1 : 0;
  const elapsedSeconds = normalizeRuntimeSeconds(run?.elapsed_seconds);
  const remainingSteps = maxSteps > 0 ? Math.max(maxSteps - completedSteps, 0) : 0;
  const etaFromElapsed =
    elapsedSeconds !== null && completedSteps > 0 && remainingSteps > 0
      ? (elapsedSeconds / completedSteps) * remainingSteps
      : remainingSteps === 0 && maxSteps > 0
        ? 0
        : null;

  return {
    completedSteps,
    maxSteps,
    fraction,
    percentLabel: formatProgressPercent(fraction),
    elapsedSeconds,
    etaSeconds: normalizeRuntimeSeconds(run?.eta_seconds) ?? etaFromElapsed,
  };
}

export function formatTrainingEta(
  snapshot: TrainingStepProgressSnapshot,
  status: TrainingJobStatus | null
): string {
  if (status === "completed") {
    return "0s";
  }
  if (status === "failed" || status === "cancelled" || snapshot.maxSteps <= 0) {
    return "n/a";
  }
  if (snapshot.completedSteps <= 0) {
    return "Waiting for first logged step";
  }
  if (snapshot.etaSeconds === null) {
    return "Calculating...";
  }
  return formatDuration(snapshot.etaSeconds);
}

export function formatTrainingElapsed(
  snapshot: TrainingStepProgressSnapshot,
  status: TrainingJobStatus | null
): string {
  if (snapshot.elapsedSeconds !== null) {
    return formatDuration(snapshot.elapsedSeconds);
  }
  if (status === "running" || status === "pending") {
    return "Waiting for first logged step";
  }
  return "n/a";
}

export function recommendationFactorToneClass(tone: "good" | "neutral" | "warning"): string {
  if (tone === "good") {
    return "tone-good";
  }
  if (tone === "warning") {
    return "tone-warning";
  }
  return "tone-neutral";
}

export function formatDatasetScaleLabel(value: string): string {
  return value.replaceAll("_", " ");
}

export function numbersRoughlyEqual(left: number, right: number, tolerance = 0.000001): boolean {
  return Math.abs(left - right) <= tolerance;
}

export function statusTone(status: string): string {
  if (status === "completed") {
    return "tone-good";
  }
  if (status === "failed" || status === "cancelled") {
    return "tone-error";
  }
  if (status === "running") {
    return "tone-neutral";
  }
  return "tone-warn";
}

export function issueTone(issue: TrainingIssue): "error" | "warning" {
  return issue.severity === "warning" ? "warning" : "error";
}

const ISSUE_LOCATION_LABELS: Record<string, string> = {
  "$.model_config.vocab_size": "Model vocabulary size",
  "$.training_config": "Training settings",
  "$.training_config.lr_scheduler": "Learning rate schedule",
  "$.training_config.max_steps": "Max training steps",
  "$.training_config.micro_batch_size": "Micro batch size",
  "$.training_config.optimizer.lr": "Learning rate",
  "$.training_config.save_every": "Checkpoint cadence",
  "$.training_config.sample_every": "Sample cadence",
  "$.training_config.seq_len": "Sequence length",
  "$.training_config.total_batch_size": "Total batch size (tokens)",
  "$.dataloader_config": "Dataset settings",
};

export function formatIssueLocation(path: string): string {
  const direct = ISSUE_LOCATION_LABELS[path];
  if (direct) {
    return direct;
  }

  const datasetMatch = path.match(/^\$\.dataloader_config\.datasets\[(\d+)]\.data_files$/);
  if (datasetMatch) {
    return `Dataset ${Number(datasetMatch[1]) + 1} data files`;
  }

  if (path.startsWith("$.training_config.")) {
    return humanizeConfigPath(path.replace("$.training_config.", ""));
  }
  if (path.startsWith("$.dataloader_config.")) {
    return humanizeConfigPath(path.replace("$.dataloader_config.", ""));
  }
  if (path.startsWith("$.model_config.")) {
    return humanizeConfigPath(path.replace("$.model_config.", ""));
  }
  return path;
}

export function humanizeConfigPath(path: string): string {
  return path
    .replace(/\[(\d+)]/g, " $1")
    .split(".")
    .filter(Boolean)
    .map((part) => humanizeConfigToken(part))
    .join(" ");
}

export function prettyJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function defaultRunName(
  project: ProjectDetail | null,
  tokenizer: TokenizerTrainingJob | null
): string {
  const projectName = project?.name ?? "model";
  const tokenizerName = asString(tokenizer?.tokenizer_config?.name, "tokenizer");
  return `${projectName} x ${tokenizerName}`;
}

export function canStopTrainingRun(job: TrainingJob | null | undefined): job is TrainingJob {
  return job != null && (job.status === "pending" || job.status === "running");
}

export function shouldPollTrainingRun(job: TrainingJob): boolean {
  return !TERMINAL_TRAINING_STATUSES.has(job.status);
}
