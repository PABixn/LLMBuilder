import {
  FiPlus,
  FiRefreshCw,
  FiTrash2,
} from "react-icons/fi";

import { ConfigNumberInput } from "../../shared/components/ConfigNumberInput";
import { FieldLabelText, HelpTooltip, InfoTooltip } from "../../shared/components/HelpTooltip";
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
      <summary>
        <span>Sampling prompts</span>
        <InfoTooltip label="Sampling prompts explanation" align="right" width="wide">
          <strong>Sampling prompts</strong>
          <p>
            During training, the app periodically asks the current checkpoint to continue
            these text starts. They are for quick qualitative checks, not chat instructions.
          </p>
        </InfoTooltip>
      </summary>
      <div className="settingsGrid">
        <div className="settingsGroup">
          <div className="trainingSettingsPanelHead">
            <div className="settingsGroupHeader">
              <h3>Prompt presets</h3>
              <p className="settingsGroupHint">
                Prompts used for sample text during training.
              </p>
            </div>
            <div className="trainingPromptToolbar">
              <span className="pillBadge tone-neutral">
                {promptEntries.length} preset{promptEntries.length === 1 ? "" : "s"}
              </span>
              <HelpTooltip label="Reset sampling prompts" content="Restores the default sampling prompt presets and replaces the current list.">
                <button
                  type="button"
                  className="buttonGhost buttonSmall"
                  onClick={handleResetPrompts}
                  disabled={isResettingPrompts}
                >
                  <FiRefreshCw /> {isResettingPrompts ? "Resetting..." : "Reset prompts"}
                </button>
              </HelpTooltip>
              <HelpTooltip label="Add sampling prompt" content="Adds another prompt preset. It will be included when sample generation runs during training.">
                <button
                  type="button"
                  className="buttonGhost buttonSmall"
                  onClick={handleAddPrompt}
                >
                  <FiPlus /> Add prompt
                </button>
              </HelpTooltip>
            </div>
          </div>
          <p className="trainingPromptHintLine">
            Use short text starts, not chat instructions.
          </p>
          <div className="trainingPromptGrid">
            {promptEntries.map((prompt, index) => (
              <article key={`prompt-${index}`} className="trainingPromptCard">
                <div className="trainingPromptCardHead">
                  <div className="trainingPromptTitleGroup">
                    <div className="trainingPromptTitle">Prompt {index + 1}</div>
                    <p className="trainingPromptMeta">
                      {Math.max(0, asString(prompt.prompt).trim().length)} chars
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
                  <FieldLabelText tooltipLabel="Prompt text explanation" tooltip="The text prefix the model will continue during training samples. Short starts make progress easier to compare across checkpoints.">
                    Prompt text
                  </FieldLabelText>
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
                    <FieldLabelText tooltipLabel="Prompt max tokens explanation" tooltip="Maximum number of new tokens generated for this sample. Larger values are slower and make logs longer.">
                      Max tokens
                    </FieldLabelText>
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
                    <FieldLabelText tooltipLabel="Sampling temperature explanation" tooltip="Controls randomness during sample generation. Lower values are steadier; higher values are more varied. This affects samples only, not training.">
                      Temperature
                    </FieldLabelText>
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
                    <FieldLabelText tooltipLabel="Top-k explanation" tooltip="Limits sampling to the K most likely next tokens. Smaller K is more constrained; larger K allows more variety. This affects samples only.">
                      Top-k
                    </FieldLabelText>
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
