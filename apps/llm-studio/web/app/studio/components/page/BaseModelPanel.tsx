import type { Dispatch, SetStateAction } from "react";

import type { StudioDocument, StudioDocumentNumericField } from "../../types";
import { integerInputValue, parseIntegerInput } from "../../utils/format";

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
  const actionButtonLabel = isProjectSaving ? "Working..." : "New config";
  const actionButtonDisabled = isProjectLoading || isProjectSaving;
  const projectStatusCopy = isProjectLoading
    ? "Loading saved model config..."
    : isProjectSaving
      ? currentProjectId
        ? "Auto-saving changes..."
        : "Creating saved model config..."
      : currentProjectId
        ? `Changes auto-save to saved config ${currentProjectId.slice(0, 8)}. Click New config to create another one.`
        : normalizedProjectName === ""
          ? "Enter a config name to create a saved model config. Nothing is saved to the workspace while the name is empty."
          : "A new saved model config will be created automatically.";

  return (
    <section id="base-model" className="panelCard">
      <div className="panelHead">
        <div>
          <p className="panelEyebrow">Base Model</p>
          <h2>Core config</h2>
          <p className="panelCopy">
            Shared dimensions used by the builder and validation checks.
          </p>
        </div>
      </div>
      <div className="fieldGrid">
        <div className="fieldInlineRow coreConfigProjectRow">
          <label className="fieldLabel inlineField" htmlFor="model_project_name">
            <span>Config Name</span>
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
        </div>
        <p className="fieldNote fullWidthField coreConfigProjectNote" aria-live="polite">
          {projectStatusCopy}
        </p>
        <label className="fieldLabel" htmlFor="context_length">
          <span>Context Length</span>
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
          <span>Vocab Size</span>
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
          <span>Embedding Dimension</span>
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
          <span>Weight Tying</span>
        </label>
      </div>
    </section>
  );
}
