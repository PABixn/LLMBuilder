import type { TrainingJob, TrainingMetricPoint } from "../../../lib/training/types";

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

export function SimpleRunMonitor({
  run,
  metrics,
  checkpointCount,
  sampleCount,
}: SimpleRunMonitorProps) {
  const latestMetric = metrics[metrics.length - 1] ?? null;
  if (!run) {
    return <p className="simpleMuted">No training run started.</p>;
  }

  return (
    <div className="simpleRunMonitor">
      <div className="simpleProgressBar" aria-label={`Training progress ${Math.round(run.progress * 100)} percent`}>
        <span style={{ width: `${Math.max(0, Math.min(100, run.progress * 100))}%` }} />
      </div>
      <div className="simpleSummaryGrid">
        <span>
          <strong>{run.stage || run.state}</strong>
          <small>Stage</small>
        </span>
        <span>
          <strong>
            {run.last_step.toLocaleString()} / {run.max_steps.toLocaleString()}
          </strong>
          <small>Steps</small>
        </span>
        <span>
          <strong>{formatNumber(run.latest_loss ?? latestMetric?.loss, 4)}</strong>
          <small>Loss</small>
        </span>
        <span>
          <strong>{formatNumber(run.latest_lr ?? latestMetric?.lr, 6)}</strong>
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
