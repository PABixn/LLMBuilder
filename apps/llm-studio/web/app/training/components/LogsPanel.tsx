interface LogsPanelProps {
  logs: {
    stderr: string[];
    stdout: string[];
  };
}

export function LogsPanel({ logs }: LogsPanelProps) {
  return (
    <details className="sectionDisclosure">
      <summary className="sectionDisclosureSummary">Logs</summary>
      <div className="trainingDualLog">
        <div className="trainingLogBox">{logs.stdout.join("\n") || "No output yet."}</div>
        <div className="trainingLogBox">{logs.stderr.join("\n") || "No errors yet."}</div>
      </div>
    </details>
  );
}
