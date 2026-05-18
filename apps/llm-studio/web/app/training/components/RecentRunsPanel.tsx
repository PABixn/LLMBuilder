import {
  FiTrash2,
  FiXCircle,
} from "react-icons/fi";

import type { TrainingJob } from "../../../lib/training/types";
import { formatDate } from "../../../lib/workspaceAssets";
import {
  canStopTrainingRun,
  statusTone,
} from "../lib/display";
import {
  formatStatusLabel,
} from "../lib/run";
import { HelpTooltip, InfoTooltip } from "../../shared/components/HelpTooltip";

interface RecentRunsPanelProps {
  activeRunId: string | null;
  onDeleteRun: (jobId: string) => void;
  onRefresh: () => void;
  onSelectRun: (jobId: string) => void;
  onStopRun: (jobId: string) => void;
  recentRuns: TrainingJob[];
  stoppingRunId: string | null;
}

export function RecentRunsPanel({
  activeRunId,
  onDeleteRun,
  onRefresh,
  onSelectRun,
  onStopRun,
  recentRuns,
  stoppingRunId,
}: RecentRunsPanelProps) {
  return (
    <section className="panelCard trainingRecentRunsPanel">
      <div className="panelHead">
        <div>
          <h2>
            Recent runs
            <InfoTooltip label="Recent training runs explanation" align="left" width="wide">
              <p>
                Recent runs are fetched from the backend. Selecting a run opens its metrics,
                logs, checkpoints, samples, and RunPod lifecycle if available.
              </p>
            </InfoTooltip>
          </h2>
          <p className="panelCopy trainingRecentPanelCopy">
            Select a run to view its progress.
          </p>
        </div>
        <HelpTooltip label="Refresh recent runs" content="Reloads the recent training run list from the backend.">
          <button type="button" className="buttonGhost buttonSmall" onClick={onRefresh}>
            Refresh
          </button>
        </HelpTooltip>
      </div>
      <div className="trainingRecentList">
        {recentRuns.length ? (
          recentRuns.map((job) => (
            <div key={job.id} className={`trainingRecentCard ${activeRunId === job.id ? "is-active" : ""}`}>
              <button
                type="button"
                className="trainingRecentSelect"
                onClick={() => onSelectRun(job.id)}
              >
                <div>
                  <strong className="trainingRecentTitle">{job.name}</strong>
                  <p>{job.project_name} / {job.tokenizer_name}</p>
                </div>
                <div className="trainingRecentRowMeta">
                  <span className={`pillBadge ${statusTone(job.status)}`}>{formatStatusLabel(job.status)}</span>
                  <p>{formatDate(job.created_at)}</p>
                </div>
              </button>
              <div className="trainingRecentIconActions">
                {canStopTrainingRun(job) ? (
                  <HelpTooltip label={`Stop ${job.name}`} content="Requests the backend to stop this running or pending training job.">
                    <button
                      type="button"
                      className="trainingRecentIconButton trainingRecentIconButton-danger"
                      onClick={() => onStopRun(job.id)}
                      disabled={stoppingRunId === job.id}
                      aria-label={`Stop ${job.name}`}
                    >
                      <FiXCircle aria-hidden="true" />
                    </button>
                  </HelpTooltip>
                ) : null}
                <HelpTooltip label={`Delete ${job.name}`} content={job.status === "running" || job.status === "pending" ? "Running and pending jobs must be stopped or finish before they can be deleted." : "Deletes this training run record and its associated workspace entry when confirmed."}>
                  <button
                    type="button"
                    className="trainingRecentIconButton trainingRecentIconButton-danger"
                    onClick={() => onDeleteRun(job.id)}
                    disabled={job.status === "running" || job.status === "pending"}
                    aria-label={`Delete ${job.name}`}
                  >
                    <FiTrash2 aria-hidden="true" />
                  </button>
                </HelpTooltip>
              </div>
            </div>
          ))
        ) : (
          <div className="trainingEmpty">
            No training runs yet.
          </div>
        )}
      </div>
    </section>
  );
}
