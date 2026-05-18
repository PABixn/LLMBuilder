import { FiPlay } from "react-icons/fi";

import { ConfigNumberInput } from "../../shared/components/ConfigNumberInput";

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
          <h2>Prompt</h2>
          <p className="panelCopy">
            Enter text for the model to continue.
          </p>
        </div>
        <button type="submit" className="buttonPrimary" disabled={!canGenerate}>
          <FiPlay /> {generating ? "Generating..." : "Generate"}
        </button>
      </div>

      <label className="fieldLabel fullWidthField">
        <span>Prompt text</span>
        <textarea
          value={prompt}
          onChange={(event) => onPromptChange(event.target.value)}
          placeholder="Start a sentence..."
        />
      </label>

      <div className="fieldGrid compact">
        <label className="fieldLabel">
          <span>Max tokens</span>
          <ConfigNumberInput min={1} max={1024} value={maxTokens} onCommit={onMaxTokensChange} />
        </label>
        <label className="fieldLabel">
          <span>Temperature</span>
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
          <span>Top K choices</span>
          <ConfigNumberInput min={1} max={50000} value={topK} onCommit={onTopKChange} />
        </label>
        <label className="fieldLabel">
          <span>Seed</span>
          <ConfigNumberInput min={0} value={seed} onCommit={onSeedChange} />
        </label>
        <label className="fieldLabel">
          <span>Repetition penalty</span>
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
