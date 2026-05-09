import { prettyJson } from "../lib/display";

interface GeneratedConfigPanelProps {
  dataloaderConfig: Record<string, unknown>;
  trainingConfig: Record<string, unknown>;
}

export function GeneratedConfigPanel({
  dataloaderConfig,
  trainingConfig,
}: GeneratedConfigPanelProps) {
  return (
    <details className="settingsPanel">
      <summary>Generated configuration JSON</summary>
      <div className="settingsGrid">
        <div className="trainingJsonGrid">
          <pre className="trainingCodeBlock">{prettyJson(trainingConfig)}</pre>
          <pre className="trainingCodeBlock">{prettyJson(dataloaderConfig)}</pre>
        </div>
      </div>
    </details>
  );
}
