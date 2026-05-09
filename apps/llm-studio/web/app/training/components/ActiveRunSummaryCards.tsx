import {
  FiActivity,
  FiArchive,
  FiCheckCircle,
  FiRefreshCw,
} from "react-icons/fi";

import type { TrainingJob } from "../../../lib/training/types";
import { formatBytes } from "../../../lib/workspaceAssets";
import { formatInteger, formatMetric } from "../lib/metrics";
import { formatLearningRate } from "../lib/run";
import type { TrainingStepProgressSnapshot } from "../types";

interface ActiveRunSummaryCardsProps {
  activeRun: TrainingJob;
  progress: TrainingStepProgressSnapshot;
}

export function ActiveRunSummaryCards({
  activeRun,
  progress,
}: ActiveRunSummaryCardsProps) {
  return (
    <div className="statusGrid">
      <div className="statusCard">
        <div className="statusCardIcon"><FiActivity /></div>
        <div>
          <div className="statusCardTitle">Training step</div>
          <div className="statusCardValue">
            {formatInteger(progress.completedSteps)} / {formatInteger(progress.maxSteps)}
          </div>
          <div className="statusCardDetail">Training progress: {progress.percentLabel} of steps</div>
        </div>
      </div>
      <div className="statusCard">
        <div className="statusCardIcon"><FiCheckCircle /></div>
        <div>
          <div className="statusCardTitle">Loss</div>
          <div className="statusCardValue">{formatMetric(activeRun.latest_loss, 4)}</div>
          <div className="statusCardDetail">Gradient norm: {formatMetric(activeRun.latest_grad_norm, 3)}</div>
        </div>
      </div>
      <div className="statusCard">
        <div className="statusCardIcon"><FiRefreshCw /></div>
        <div>
          <div className="statusCardTitle">Learning Rate</div>
          <div className="statusCardValue">{formatLearningRate(activeRun.latest_lr)}</div>
          <div className="statusCardDetail">Tokens per second: {formatInteger(activeRun.latest_tokens_per_sec)}</div>
        </div>
      </div>
      <div className="statusCard">
        <div className="statusCardIcon"><FiArchive /></div>
        <div>
          <div className="statusCardTitle">Saved artifacts</div>
          <div className="statusCardValue">
            {formatInteger(activeRun.checkpoint_count)} checkpoints
          </div>
          <div className="statusCardDetail">
            {formatInteger(activeRun.sample_count)} sample groups • {formatBytes(activeRun.output_size_bytes)}
          </div>
        </div>
      </div>
    </div>
  );
}
