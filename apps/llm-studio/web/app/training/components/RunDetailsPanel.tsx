import type {
  TrainingDataPreview,
  TrainingJob,
} from "../../../lib/training/types";
import { prettyJson } from "../lib/display";

interface RunDetailsPanelProps {
  activeRun: TrainingJob;
  dataPreview: TrainingDataPreview | null;
}

export function RunDetailsPanel({ activeRun, dataPreview }: RunDetailsPanelProps) {
  return (
    <>
      <details className="sectionDisclosure">
        <summary className="sectionDisclosureSummary">Training Data Preview</summary>
        {dataPreview ? (
          <div className="trainingJsonGrid">
            <pre className="trainingCodeBlock">{prettyJson(dataPreview)}</pre>
          </div>
        ) : (
          <div className="trainingEmpty">
            The trainer has not published a data preview for this run yet.
          </div>
        )}
      </details>

      <details className="sectionDisclosure">
        <summary className="sectionDisclosureSummary">Resolved runtime and configurations</summary>
        <div className="trainingJsonGrid">
          <pre className="trainingCodeBlock">{prettyJson(activeRun.resolved_runtime)}</pre>
          <pre className="trainingCodeBlock">{prettyJson(activeRun.memory_estimate)}</pre>
        </div>
      </details>
    </>
  );
}
