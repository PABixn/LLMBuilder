import type { TrainingJob, TrainingMetricPoint } from "../../../lib/training/types";
import { deriveTrainingStepProgress } from "../../training/lib/display";
import { formatLearningRate } from "../../training/lib/run";

interface SimpleRunMonitorProps {
  run: TrainingJob | null;
  metrics: TrainingMetricPoint[];
  checkpointCount: number;
  sampleCount: number;
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

function formatStateLabel(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function SimpleRunMonitor({
  run,
  metrics,
  checkpointCount,
  sampleCount,
}: SimpleRunMonitorProps) {
  const latestMetric = metrics[metrics.length - 1] ?? null;
  if (!run) {
    return null;
  }
  const progress = deriveTrainingStepProgress(run);

  return (
    <div className="simpleRunMonitor">
      <div
        className="simpleProgressBar"
        aria-label={`Training progress ${progress.percentLabel}`}
      >
        <span
          style={{ width: `${Math.max(0, Math.min(100, progress.fraction * 100))}%` }}
        />
      </div>
      <div className="simpleSummaryGrid">
        <span>
          <strong>{formatStateLabel(run.status)}</strong>
          <small>Status</small>
        </span>
        <span>
          <strong>
            {progress.completedSteps.toLocaleString()} / {progress.maxSteps.toLocaleString()}
          </strong>
          <small>Steps</small>
        </span>
        <span>
          <strong>{formatNumber(run.latest_loss ?? latestMetric?.loss, 4)}</strong>
          <small>Loss</small>
        </span>
        <span>
          <strong>{formatLearningRate(run.latest_lr ?? latestMetric?.lr)}</strong>
          <small>LR</small>
        </span>
        <span>
          <strong>{formatNumber(run.latest_tokens_per_sec ?? latestMetric?.tok_per_sec, 1)}</strong>
          <small>Tokens/sec</small>
        </span>
        <span>
          <strong>{checkpointCount.toLocaleString()}</strong>
          <small>Checkpoints</small>
        </span>
        <span>
          <strong>{sampleCount.toLocaleString()}</strong>
          <small>Samples</small>
        </span>
      </div>
      {run.error ? <div className="inlineNotice tone-error">{run.error}</div> : null}
    </div>
  );
}
