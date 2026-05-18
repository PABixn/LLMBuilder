import type { GenerateTrainingCompletionResponse } from "../../../lib/training/types";
import { formatInteger } from "../lib/formatters";
import { InfoTooltip } from "../../shared/components/HelpTooltip";

type InferenceOutputPanelProps = {
  result: GenerateTrainingCompletionResponse | null;
};

export function InferenceOutputPanel({ result }: InferenceOutputPanelProps) {
  return (
    <section className="panelCard inferenceOutputPanel">
      <div className="panelHead">
        <div>
          <h2>
            Continuation
            <InfoTooltip label="Inference output explanation" align="left" width="wide">
              <p>
                Output is shown as the original prompt followed by the generated continuation.
                The step label shows which checkpoint produced the text.
              </p>
            </InfoTooltip>
          </h2>
          <p className="panelCopy">The model output appears here.</p>
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
        <div className="trainingEmpty">Generate text to see output.</div>
      )}
    </section>
  );
}
