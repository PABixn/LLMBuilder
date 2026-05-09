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
          <h2>Recent Runs</h2>
          <p className="panelCopy trainingRecentPanelCopy">
            Recent jobs stay navigable after refresh so you can jump between current and past runs quickly.
          </p>
        </div>
        <button type="button" className="buttonGhost buttonSmall" onClick={onRefresh}>
          Refresh
        </button>
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
                  <button
                    type="button"
                    className="trainingRecentIconButton trainingRecentIconButton-danger"
                    onClick={() => onStopRun(job.id)}
                    disabled={stoppingRunId === job.id}
                    aria-label={`Stop ${job.name}`}
                    title={stoppingRunId === job.id ? "Stopping run" : "Stop run"}
                  >
                    <FiXCircle aria-hidden="true" />
                  </button>
                ) : null}
                <button
                  type="button"
                  className="trainingRecentIconButton trainingRecentIconButton-danger"
                  onClick={() => onDeleteRun(job.id)}
                  disabled={job.status === "running" || job.status === "pending"}
                  aria-label={`Delete ${job.name}`}
                  title="Delete run"
                >
                  <FiTrash2 aria-hidden="true" />
                </button>
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
