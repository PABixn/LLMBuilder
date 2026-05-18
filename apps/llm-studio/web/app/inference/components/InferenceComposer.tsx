import { FiPlay } from "react-icons/fi";

import { ConfigNumberInput } from "../../shared/components/ConfigNumberInput";
import { FieldLabelText, HelpTooltip, InfoTooltip } from "../../shared/components/HelpTooltip";

type InferenceComposerProps = {
  prompt: string;
  maxTokens: number;
  temperature: number;
  topK: number;
  seed: number;
  repetitionPenalty: number;
  generating: boolean;
  canGenerate: boolean;
  onPromptChange: (value: string) => void;
  onMaxTokensChange: (value: number) => void;
  onTemperatureChange: (value: number) => void;
  onTopKChange: (value: number) => void;
  onSeedChange: (value: number) => void;
  onRepetitionPenaltyChange: (value: number) => void;
  onGenerate: () => void;
};

export function InferenceComposer({
  prompt,
  maxTokens,
  temperature,
  topK,
  seed,
  repetitionPenalty,
  generating,
  canGenerate,
  onPromptChange,
  onMaxTokensChange,
  onTemperatureChange,
  onTopKChange,
  onSeedChange,
  onRepetitionPenaltyChange,
  onGenerate,
}: InferenceComposerProps) {
  return (
    <form
      className="panelCard heroCard inferenceComposer"
      onSubmit={(event) => {
        event.preventDefault();
        onGenerate();
      }}
    >
      <div className="panelHead">
        <div>
          <h2>
            Prompt
            <InfoTooltip label="Inference prompt explanation" align="left" width="wide">
              <strong>Prompt</strong>
              <p>
                The selected trained model continues this text using the checkpoint and
                generation settings below. It is completion-style inference, not a chat template.
              </p>
            </InfoTooltip>
          </h2>
          <p className="panelCopy">
            Enter text for the model to continue.
          </p>
        </div>
        <HelpTooltip label="Generate explanation" align="right" content="Sends the prompt and generation settings to the backend for the selected training artifact. Requires a model and non-empty prompt.">
          <button type="submit" className="buttonPrimary" disabled={!canGenerate}>
            <FiPlay /> {generating ? "Generating..." : "Generate"}
          </button>
        </HelpTooltip>
      </div>

      <label className="fieldLabel fullWidthField">
        <FieldLabelText tooltipLabel="Prompt text explanation" tooltip="The exact text the model receives as context. The output begins after this text and is limited by max tokens.">
          Prompt text
        </FieldLabelText>
        <textarea
          value={prompt}
          onChange={(event) => onPromptChange(event.target.value)}
          placeholder="Start a sentence..."
        />
      </label>

      <div className="fieldGrid compact">
        <label className="fieldLabel">
          <FieldLabelText tooltipLabel="Max tokens explanation" tooltip="Maximum number of new tokens to generate. It does not count the prompt tokens already supplied.">
            Max tokens
          </FieldLabelText>
          <ConfigNumberInput min={1} max={1024} value={maxTokens} onCommit={onMaxTokensChange} />
        </label>
        <label className="fieldLabel">
          <FieldLabelText tooltipLabel="Temperature explanation" tooltip="Controls randomness in token selection. 0 is deterministic; higher values produce more varied but less predictable text.">
            Temperature
          </FieldLabelText>
          <ConfigNumberInput
            mode="decimal"
            min={0}
            max={5}
            step={0.1}
            value={temperature}
            onCommit={onTemperatureChange}
          />
        </label>
        <label className="fieldLabel">
          <FieldLabelText tooltipLabel="Top K choices explanation" tooltip="Restricts each next-token choice to the K most likely tokens. Lower values are safer and repetitive; higher values allow broader vocabulary.">
            Top K choices
          </FieldLabelText>
          <ConfigNumberInput min={1} max={50000} value={topK} onCommit={onTopKChange} />
        </label>
        <label className="fieldLabel">
          <FieldLabelText tooltipLabel="Seed explanation" tooltip="Random seed for reproducible sampling when settings and checkpoint are unchanged. Change it to get another variation.">
            Seed
          </FieldLabelText>
          <ConfigNumberInput min={0} value={seed} onCommit={onSeedChange} />
        </label>
        <label className="fieldLabel">
          <FieldLabelText tooltipLabel="Repetition penalty explanation" tooltip="Penalizes tokens that already appeared. Values above 1 reduce loops; too high can make output unnatural.">
            Repetition penalty
          </FieldLabelText>
          <ConfigNumberInput
            mode="decimal"
            min={0.1}
            max={5}
            step={0.1}
            value={repetitionPenalty}
            onCommit={onRepetitionPenaltyChange}
          />
        </label>
      </div>
    </form>
  );
}
