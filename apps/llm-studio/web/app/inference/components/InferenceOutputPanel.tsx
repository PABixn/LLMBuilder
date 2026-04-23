import type { GenerateTrainingCompletionResponse } from "../../../lib/trainingApi";
import { formatInteger } from "../lib/formatters";

type InferenceOutputPanelProps = {
  result: GenerateTrainingCompletionResponse | null;
};

export function InferenceOutputPanel({ result }: InferenceOutputPanelProps) {
  return (
    <section className="panelCard inferenceOutputPanel">
      <div className="panelHead">
        <div>
          <h2>Continuation</h2>
          <p className="panelCopy">Generated text appears as a direct continuation of the prefix.</p>
        </div>
        {result ? (
          <span className="trainingAssetMeta">
            Step {formatInteger(result.checkpoint_step)} |{" "}
            {formatInteger(result.generated_token_count)} tokens
          </span>
        ) : null}
      </div>
      {result ? (
        <div className="inferenceOutput">
          <span className="inferencePromptText">{result.prompt}</span>
          <span className="inferenceCompletionText">{result.completion}</span>
        </div>
      ) : (
        <div className="trainingEmpty">Run a completion to see model output.</div>
      )}
    </section>
  );
}
