import type { Dispatch, SetStateAction } from "react";

import type { StudioDocument, StudioDocumentNumericField } from "../../types";
import { integerInputValue, parseIntegerInput } from "../../utils/format";
import { FieldLabelText, HelpTooltip, InfoTooltip } from "../../../shared/components/HelpTooltip";

type BaseModelPanelProps = {
  documentState: StudioDocument;
  updateBaseField: (key: StudioDocumentNumericField, value: number) => void;
  setDocumentState: Dispatch<SetStateAction<StudioDocument>>;
  projectName: string;
  setProjectName: Dispatch<SetStateAction<string>>;
  currentProjectId: string | null;
  isProjectLoading: boolean;
  isProjectSaving: boolean;
  createNewProject: () => Promise<void>;
};

export function BaseModelPanel({
  documentState,
  updateBaseField,
  setDocumentState,
  projectName,
  setProjectName,
  currentProjectId,
  isProjectLoading,
  isProjectSaving,
  createNewProject,
}: BaseModelPanelProps) {
  const normalizedProjectName = projectName.trim();
  const actionButtonLabel = isProjectSaving ? "Saving..." : "New config";
  const actionButtonDisabled = isProjectLoading || isProjectSaving;
  const projectStatusCopy = isProjectLoading
    ? "Loading config..."
    : isProjectSaving
      ? currentProjectId
        ? "Saving changes..."
        : "Creating config..."
      : currentProjectId
        ? `Changes save automatically to ${currentProjectId.slice(0, 8)}.`
        : normalizedProjectName === ""
          ? "Enter a name to save this config."
          : "This config will save automatically.";

  return (
    <section id="base-model" className="panelCard">
      <div className="panelHead">
        <div>
          <p className="panelEyebrow">Base model</p>
          <h2>
            Core config
            <InfoTooltip label="Core model config explanation" align="left" width="wide">
              <strong>Core config</strong>
              <p>
                These dimensions define the model&apos;s shape. Training and tokenizer
                compatibility depend on vocabulary size and context length.
              </p>
            </InfoTooltip>
          </h2>
          <p className="panelCopy">
            Set the main model dimensions.
          </p>
        </div>
      </div>
      <div className="fieldGrid">
        <div className="fieldInlineRow coreConfigProjectRow">
          <label className="fieldLabel inlineField" htmlFor="model_project_name">
            <FieldLabelText tooltipLabel="Config name explanation" tooltip="Name used to auto-save this model config in the workspace. Existing configs continue saving to their current project ID.">
              Config name
            </FieldLabelText>
            <input
              id="model_project_name"
              type="text"
              value={projectName}
              onChange={(event) => setProjectName(event.target.value)}
              placeholder="GPT-2 baseline"
              maxLength={200}
              autoComplete="off"
            />
          </label>
          <HelpTooltip label="New config explanation" content="Creates a separate saved model config using the current form values and name. It does not reset fields by itself.">
            <button
              type="button"
              className="buttonGhost"
              onClick={() => {
                void createNewProject();
              }}
              disabled={actionButtonDisabled}
            >
              {actionButtonLabel}
            </button>
          </HelpTooltip>
        </div>
        <p className="fieldNote fullWidthField coreConfigProjectNote" aria-live="polite">
          {projectStatusCopy}
        </p>
        <label className="fieldLabel" htmlFor="context_length">
          <FieldLabelText tooltipLabel="Context length explanation" tooltip="Maximum tokens the model can attend to at once. Training sequence length must be less than or equal to this value.">
            Context length
          </FieldLabelText>
          <input
            id="context_length"
            type="number"
            min={1}
            step={1}
            value={integerInputValue(documentState.context_length)}
            onChange={(event) =>
              updateBaseField(
                "context_length",
                parseIntegerInput(event.target.value, documentState.context_length)
              )
            }
          />
        </label>
        <label className="fieldLabel" htmlFor="vocab_size">
          <FieldLabelText tooltipLabel="Vocab size explanation" tooltip="Number of token IDs supported by the model embedding table. It must match the tokenizer vocabulary size before training.">
            Vocab size
          </FieldLabelText>
          <input
            id="vocab_size"
            type="number"
            min={1}
            step={1}
            value={integerInputValue(documentState.vocab_size)}
            onChange={(event) =>
              updateBaseField(
                "vocab_size",
                parseIntegerInput(event.target.value, documentState.vocab_size)
              )
            }
          />
        </label>
        <label className="fieldLabel" htmlFor="n_embd">
          <FieldLabelText tooltipLabel="Embedding size explanation" tooltip="Width of token embeddings and hidden states. Larger values increase model capacity and memory use.">
            Embedding size
          </FieldLabelText>
          <input
            id="n_embd"
            type="number"
            min={1}
            step={1}
            value={integerInputValue(documentState.n_embd)}
            onChange={(event) =>
              updateBaseField("n_embd", parseIntegerInput(event.target.value, documentState.n_embd))
            }
          />
        </label>
        <label className="toggleField tall" htmlFor="weight_tying">
          <input
            id="weight_tying"
            type="checkbox"
            checked={documentState.weight_tying}
            onChange={(event) =>
              setDocumentState((current) => ({ ...current, weight_tying: event.target.checked }))
            }
          />
          <span className="fieldLabelText">
            <span>Weight tying</span>
            <InfoTooltip label="Weight tying explanation" width="wide">
              <p>
                Shares weights between token embeddings and output projection. This reduces
                parameter count and is common for GPT-style language models.
              </p>
            </InfoTooltip>
          </span>
        </label>
      </div>
    </section>
  );
}
