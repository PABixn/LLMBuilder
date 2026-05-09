import type { TrainingJob } from "../../../lib/training/types";
import { prettyJson } from "../lib/display";

interface RunDetailsPanelProps {
  activeRun: TrainingJob;
}

export function RunDetailsPanel({ activeRun }: RunDetailsPanelProps) {
  return (
    <details className="sectionDisclosure">
      <summary className="sectionDisclosureSummary">Resolved runtime and configurations</summary>
      <div className="trainingJsonGrid">
        <pre className="trainingCodeBlock">{prettyJson(activeRun.resolved_runtime)}</pre>
        <pre className="trainingCodeBlock">{prettyJson(activeRun.memory_estimate)}</pre>
      </div>
    </details>
  );
}
