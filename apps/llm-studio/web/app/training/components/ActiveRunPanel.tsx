import {
  useMemo,
} from "react";
import {
  FiX,
  FiXCircle,
} from "react-icons/fi";

import type {
  TrainingCheckpointEntry,
  TrainingJob,
  TrainingMetricPoint,
  TrainingSampleEntry,
} from "../../../lib/training/types";
import { formatDate } from "../../../lib/workspaceAssets";
import {
  canStopTrainingRun,
  deriveTrainingStepProgress,
  formatTrainingElapsed,
  formatTrainingEta,
  statusTone,
} from "../lib/display";
import {
  formatStatusLabel,
} from "../lib/run";
import { ActiveRunSummaryCards } from "./ActiveRunSummaryCards";
import { CheckpointsPanel } from "./CheckpointsPanel";
import { LogsPanel } from "./LogsPanel";
import { MetricsPanel } from "./MetricsPanel";
import { RunDetailsPanel } from "./RunDetailsPanel";
import { RunPodLifecyclePanel } from "./RunPodLifecyclePanel";
import { SamplesPanel } from "./SamplesPanel";

interface ActiveRunPanelProps {
  activeRun: TrainingJob | null;
  checkpoints: TrainingCheckpointEntry[];
  logs: {
    stderr: string[];
    stdout: string[];
  };
  metrics: TrainingMetricPoint[];
  onClose: () => void;
  onStopRun: (jobId: string) => void;
  pollIntervalSeconds: number;
  samples: TrainingSampleEntry[];
  stoppingRunId: string | null;
}

export function ActiveRunPanel({
  activeRun,
  checkpoints,
  logs,
  metrics,
  onClose,
  onStopRun,
  pollIntervalSeconds,
  samples,
  stoppingRunId,
}: ActiveRunPanelProps) {
  const activeRunStepProgress = useMemo(() => deriveTrainingStepProgress(activeRun), [activeRun]);
  const activeRunCanBeStopped = canStopTrainingRun(activeRun);
  const stoppingActiveRun = activeRunCanBeStopped && stoppingRunId === activeRun?.id;

  return (
    <section className="panelCard trainingActiveRunPanel">
      <div className="panelHead">
        <div>
          <h2>Active run</h2>
          <p className="panelCopy">
            Updates every {pollIntervalSeconds} seconds.
          </p>
        </div>
        <div className="trainingActiveRunHeaderActions">
          {activeRun ? (
            <span className={`pillBadge ${statusTone(activeRun.status)}`}>{formatStatusLabel(activeRun.status)}</span>
          ) : null}
          {activeRunCanBeStopped && activeRun ? (
            <button
              type="button"
              className="buttonDanger buttonSmall"
              onClick={() => onStopRun(activeRun.id)}
              disabled={stoppingActiveRun}
            >
              <FiXCircle aria-hidden="true" />
              {stoppingActiveRun ? "Stopping..." : "Stop run"}
            </button>
          ) : null}
          <button
            type="button"
            className="trainingActiveRunCloseButton"
            onClick={onClose}
            aria-label="Close active run"
            title="Close active run"
          >
            <FiX aria-hidden="true" />
          </button>
        </div>
      </div>

      {activeRun ? (
        <>
          <div className="trainingProgress">
            <div className="trainingSectionHeader">
              <h3>{activeRun.name}</h3>
            </div>
            <div className="trainingProgressBar">
              <span style={{ width: `${activeRunStepProgress.fraction * 100}%` }} />
            </div>
            <div className="trainingInlineMeta">
              <span>Run ID: {activeRun.id.slice(0, 8)}</span>
              <span>Runs on: {activeRun.executor_kind === "runpod_pod" ? "RunPod" : "Local machine"}</span>
              <span>Created {formatDate(activeRun.created_at)}</span>
              <span>Started {activeRun.started_at ? formatDate(activeRun.started_at) : "Waiting"}</span>
              <span>Elapsed: {formatTrainingElapsed(activeRunStepProgress, activeRun.status)}</span>
              <span>ETA: {formatTrainingEta(activeRunStepProgress, activeRun.status)}</span>
            </div>
          </div>

          <RunPodLifecyclePanel activeRun={activeRun} />
          <ActiveRunSummaryCards activeRun={activeRun} progress={activeRunStepProgress} />
          <MetricsPanel activeRun={activeRun} metrics={metrics} />
          <SamplesPanel samples={samples} />
          <CheckpointsPanel checkpoints={checkpoints} />
          <LogsPanel logs={logs} />
          <RunDetailsPanel activeRun={activeRun} />
        </>
      ) : (
        <div className="trainingEmpty">No active run selected.</div>
      )}
    </section>
  );
}
