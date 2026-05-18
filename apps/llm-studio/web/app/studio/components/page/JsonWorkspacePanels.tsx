import type { Dispatch, RefObject, SetStateAction } from "react";
import { FiCopy, FiDownload, FiRefreshCw, FiUpload } from "react-icons/fi";

import type { ModelConfig } from "../../../../lib/defaults";
import { FieldLabelText, HelpTooltip, InfoTooltip } from "../../../shared/components/HelpTooltip";

type JsonWorkspacePanelsProps = {
  fileInputRef: RefObject<HTMLInputElement | null>;
  previewJson: string;
  copyJson: () => Promise<void>;
  exportJson: () => void;
  importDraft: string;
  setImportDraft: Dispatch<SetStateAction<string>>;
  applyImportText: (text: string) => void;
  modelConfig: ModelConfig;
};

export function JsonWorkspacePanels({
  fileInputRef,
  previewJson,
  copyJson,
  exportJson,
  importDraft,
  setImportDraft,
  applyImportText,
  modelConfig,
}: JsonWorkspacePanelsProps) {
  return (
    <div id="json-preview" className="twoColLayout previewLayout">
      <section className="panelCard previewPanel">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">JSON</p>
            <h2>
              JSON preview
              <InfoTooltip label="JSON preview explanation" align="left" width="wide">
                <p>
                  This is the exact model configuration generated from the visual builder.
                  Training uses the same fields after validation.
                </p>
              </InfoTooltip>
            </h2>
            <p className="panelCopy">
              JSON for the current config.
            </p>
          </div>
          <div className="actionCluster">
            <HelpTooltip label="Copy JSON" content="Copies the current model JSON to the clipboard.">
              <button
                type="button"
                className="buttonGhost iconOnly"
                onClick={copyJson}
                aria-label="Copy JSON"
              >
                <FiCopy />
              </button>
            </HelpTooltip>
            <HelpTooltip label="Export JSON" content="Downloads the current model JSON as a file.">
              <button
                type="button"
                className="buttonGhost iconOnly"
                onClick={exportJson}
                aria-label="Export JSON"
              >
                <FiDownload />
              </button>
            </HelpTooltip>
          </div>
        </div>
        <pre className="jsonPreview">
          <code>{previewJson}</code>
        </pre>
      </section>

      <section className="panelCard importPanel">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">Import</p>
            <h2>
              Import JSON
              <InfoTooltip label="Import JSON explanation" align="left" width="wide">
                <p>
                  Paste or load a model config to replace the builder state after validation.
                  Use this for configs produced outside the visual editor.
                </p>
              </InfoTooltip>
            </h2>
            <p className="panelCopy">
              Paste or load model JSON.
            </p>
          </div>
        </div>

        <div className="actionRowWrap">
          <HelpTooltip label="Choose JSON file" content="Opens a file picker for a local model JSON file and loads it into the import draft.">
            <button
              type="button"
              className="buttonGhost iconOnly"
              onClick={() => fileInputRef.current?.click()}
              aria-label="Choose JSON file"
            >
              <FiUpload />
            </button>
          </HelpTooltip>
          <HelpTooltip label="Apply import" content="Parses the import draft and applies it to the builder if it is a valid model config.">
            <button
              type="button"
              className="buttonGhost iconOnly"
              onClick={() => applyImportText(importDraft)}
              aria-label="Apply import"
            >
              <FiRefreshCw />
            </button>
          </HelpTooltip>
          <HelpTooltip label="Use current config" content="Copies the current builder config into the import editor so you can modify and reapply it.">
            <button
              type="button"
              className="buttonGhost iconOnly"
              onClick={() => setImportDraft(JSON.stringify(modelConfig, null, 2))}
              aria-label="Use current config"
            >
              <FiCopy />
            </button>
          </HelpTooltip>
        </div>

        <label className="fieldLabel" htmlFor="import-draft">
          <FieldLabelText tooltipLabel="Import JSON editor explanation" tooltip="Editable JSON draft. Applying it replaces the builder state only if parsing and model validation succeed.">
            Import JSON
          </FieldLabelText>
          <textarea
            id="import-draft"
            value={importDraft}
            onChange={(event) => setImportDraft(event.target.value)}
            placeholder="Paste model JSON..."
            rows={16}
          />
        </label>
      </section>
    </div>
  );
}
