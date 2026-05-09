import {
  FiPlus,
  FiRefreshCw,
  FiTrash2,
} from "react-icons/fi";

import { ConfigNumberInput } from "../../shared/components/ConfigNumberInput";
import {
  asNumber,
  asString,
} from "../lib/object";

interface SamplingPromptsPanelProps {
  handleAddPrompt: () => void;
  handlePromptChange: (index: number, field: string, value: unknown) => void;
  handleRemovePrompt: (index: number) => void;
  handleResetPrompts: () => void;
  isResettingPrompts: boolean;
  promptEntries: Record<string, unknown>[];
}

export function SamplingPromptsPanel({
  handleAddPrompt,
  handlePromptChange,
  handleRemovePrompt,
  handleResetPrompts,
  isResettingPrompts,
  promptEntries,
}: SamplingPromptsPanelProps) {
  return (
    <details className="settingsPanel" open>
      <summary>Sampling prompts</summary>
      <div className="settingsGrid">
        <div className="settingsGroup">
          <div className="trainingSettingsPanelHead">
            <div className="settingsGroupHeader">
              <h3>Prompt presets</h3>
              <p className="settingsGroupHint">
                Short prefixes for checking raw pretraining continuations during the run.
              </p>
            </div>
            <div className="trainingPromptToolbar">
              <span className="pillBadge tone-neutral">
                {promptEntries.length} preset{promptEntries.length === 1 ? "" : "s"}
              </span>
              <button
                type="button"
                className="buttonGhost buttonSmall"
                onClick={handleResetPrompts}
                disabled={isResettingPrompts}
              >
                <FiRefreshCw /> {isResettingPrompts ? "Resetting..." : "Reset prompts"}
              </button>
              <button
                type="button"
                className="buttonGhost buttonSmall"
                onClick={handleAddPrompt}
              >
                <FiPlus /> Add prompt
              </button>
            </div>
          </div>
          <p className="trainingPromptHintLine">
            Use autocomplete-style starts, not chat instructions or evaluation tasks.
          </p>
          <div className="trainingPromptGrid">
            {promptEntries.map((prompt, index) => (
              <article key={`prompt-${index}`} className="trainingPromptCard">
                <div className="trainingPromptCardHead">
                  <div className="trainingPromptTitleGroup">
                    <div className="trainingPromptTitle">Prompt {index + 1}</div>
                    <p className="trainingPromptMeta">
                      {Math.max(0, asString(prompt.prompt).trim().length)} characters
                    </p>
                  </div>
                  <button
                    type="button"
                    className="textButton trainingPromptRemoveButton"
                    onClick={() => handleRemovePrompt(index)}
                    aria-label={`Remove prompt ${index + 1}`}
                    title={`Remove prompt ${index + 1}`}
                  >
                    <FiTrash2 aria-hidden="true" />
                  </button>
                </div>

                <label className="fieldLabel trainingPromptEditor">
                  <span>Prompt text</span>
                  <textarea
                    rows={4}
                    value={asString(prompt.prompt)}
                    onChange={(event) =>
                      handlePromptChange(index, "prompt", event.target.value)
                    }
                    placeholder="Hello"
                  />
                </label>

                <div className="trainingPromptFields">
                  <label className="fieldLabel">
                    <span>Max tokens</span>
                    <ConfigNumberInput
                      value={asNumber(prompt.max_tokens, 64)}
                      onCommit={(value) =>
                        handlePromptChange(
                          index,
                          "max_tokens",
                          value
                        )
                      }
                    />
                  </label>
                  <label className="fieldLabel">
                    <span>Temperature</span>
                    <ConfigNumberInput
                      mode="decimal"
                      step="0.05"
                      value={asNumber(prompt.temperature, 0.7)}
                      onCommit={(value) =>
                        handlePromptChange(
                          index,
                          "temperature",
                          value
                        )
                      }
                    />
                  </label>
                  <label className="fieldLabel">
                    <span>Top-k</span>
                    <ConfigNumberInput
                      value={asNumber(prompt.top_k, 40)}
                      onCommit={(value) => handlePromptChange(index, "top_k", value)}
                    />
                  </label>
                </div>
              </article>
            ))}
          </div>
        </div>
      </div>
    </details>
  );
}
