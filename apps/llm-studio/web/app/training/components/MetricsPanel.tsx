import type {
  TrainingJob,
  TrainingMetricPoint,
} from "../../../lib/training/types";
import { formatInteger, formatMetric } from "../lib/metrics";
import { formatLearningRate } from "../lib/run";
import { MetricChart } from "./MetricChart";

interface MetricsPanelProps {
  activeRun: TrainingJob;
  metrics: TrainingMetricPoint[];
}

export function MetricsPanel({ activeRun, metrics }: MetricsPanelProps) {
  return (
    <div className="trainingChartGrid">
      <MetricChart
        title="Loss"
        metricKey="loss"
        metrics={metrics}
        latestValue={formatMetric(activeRun.latest_loss, 4)}
        stroke="var(--brand)"
        digits={4}
      />
      <MetricChart
        title="Learning Rate"
        metricKey="lr"
        metrics={metrics}
        latestValue={formatLearningRate(activeRun.latest_lr)}
        stroke="var(--ok)"
        digits={3}
      />
      <MetricChart
        title="Gradient Norm"
        metricKey="norm"
        metrics={metrics}
        latestValue={formatMetric(activeRun.latest_grad_norm, 3)}
        stroke="var(--warn)"
        digits={3}
      />
      <MetricChart
        title="Throughput"
        metricKey="tok_per_sec"
        metrics={metrics}
        latestValue={formatInteger(activeRun.latest_tokens_per_sec)}
        stroke="var(--danger)"
        digits={1}
      />
    </div>
  );
}
