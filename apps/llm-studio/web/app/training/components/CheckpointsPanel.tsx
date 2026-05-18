import type { TrainingCheckpointEntry } from "../../../lib/training/types";
import {
  formatBytes,
  formatDate,
} from "../../../lib/workspaceAssets";

interface CheckpointsPanelProps {
  checkpoints: TrainingCheckpointEntry[];
}

export function CheckpointsPanel({ checkpoints }: CheckpointsPanelProps) {
  return (
    <details className="sectionDisclosure">
      <summary className="sectionDisclosureSummary">Checkpoints</summary>
      <div className="trainingCheckpointList">
        {checkpoints.length ? (
          checkpoints.map((checkpoint) => (
            <div key={checkpoint.directory} className="trainingCheckpointCard">
              <div className="trainingCheckpointTitle">Step {checkpoint.step}</div>
              <div className="trainingCheckpointMeta">
                {checkpoint.created_at ? formatDate(checkpoint.created_at) : "Time unavailable"} • {formatBytes(checkpoint.size_bytes)}
              </div>
              <div className="trainingCheckpointMeta">{checkpoint.files.join(", ")}</div>
            </div>
          ))
        ) : (
          <div className="trainingEmpty">No checkpoints yet.</div>
        )}
      </div>
    </details>
  );
}
