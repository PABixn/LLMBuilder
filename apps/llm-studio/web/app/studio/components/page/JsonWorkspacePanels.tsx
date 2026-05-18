import type { Dispatch, RefObject, SetStateAction } from "react";
import { FiCopy, FiDownload, FiRefreshCw, FiUpload } from "react-icons/fi";

import type { ModelConfig } from "../../../../lib/defaults";

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
            <h2>JSON preview</h2>
            <p className="panelCopy">
              JSON for the current config.
            </p>
          </div>
          <div className="actionCluster">
            <button
              type="button"
              className="buttonGhost iconOnly"
              onClick={copyJson}
              aria-label="Copy JSON"
              title="Copy JSON"
            >
              <FiCopy />
            </button>
            <button
              type="button"
              className="buttonGhost iconOnly"
              onClick={exportJson}
              aria-label="Export JSON"
              title="Export JSON"
            >
              <FiDownload />
            </button>
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
            <h2>Import JSON</h2>
            <p className="panelCopy">
              Paste or load model JSON.
            </p>
          </div>
        </div>

        <div className="actionRowWrap">
          <button
            type="button"
            className="buttonGhost iconOnly"
            onClick={() => fileInputRef.current?.click()}
            aria-label="Choose JSON file"
            title="Choose JSON file"
          >
            <FiUpload />
          </button>
          <button
            type="button"
            className="buttonGhost iconOnly"
            onClick={() => applyImportText(importDraft)}
            aria-label="Apply import"
            title="Apply import"
          >
            <FiRefreshCw />
          </button>
          <button
            type="button"
            className="buttonGhost iconOnly"
            onClick={() => setImportDraft(JSON.stringify(modelConfig, null, 2))}
            aria-label="Use current config"
            title="Use current config"
          >
            <FiCopy />
          </button>
        </div>

        <label className="fieldLabel" htmlFor="import-draft">
          <span>Import JSON</span>
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
