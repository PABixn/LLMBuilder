import type { Dispatch, RefObject, SetStateAction } from "react";
import { FiCopy, FiDownload, FiRefreshCw, FiUpload } from "react-icons/fi";

import type { ModelConfig } from "../../../../lib/defaults";

import type { BuilderMetrics } from "../../types";

type JsonWorkspacePanelsProps = {
  fileInputRef: RefObject<HTMLInputElement | null>;
  previewJson: string;
  copyJson: () => Promise<void>;
  exportJson: () => void;
  importDraft: string;
  setImportDraft: Dispatch<SetStateAction<string>>;
  applyImportText: (text: string) => void;
  modelConfig: ModelConfig;
  metrics: BuilderMetrics;
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
  metrics,
}: JsonWorkspacePanelsProps) {
  const localActivationTotal = metrics.activationCount + metrics.mlpActivationStepCount;

  return (
    <div id="json-preview" className="twoColLayout previewLayout">
      <section className="panelCard previewPanel">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">JSON Preview</p>
            <h2>JSON preview</h2>
            <p className="panelCopy">
              Live JSON generated from the current visual config.
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
            <p className="panelEyebrow">Import / Workflow</p>
            <h2>JSON Import</h2>
            <p className="panelCopy">
              Paste or load a `/model` JSON document to rebuild the visual editor.
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
            aria-label="Apply Import Text"
            title="Apply Import Text"
          >
            <FiRefreshCw />
          </button>
          <button
            type="button"
            className="buttonGhost iconOnly"
            onClick={() => setImportDraft(JSON.stringify(modelConfig, null, 2))}
            aria-label="Load Current Config into Import Editor"
            title="Load Current Config into Import Editor"
          >
            <FiCopy />
          </button>
        </div>

        <label className="fieldLabel" htmlFor="import-draft">
          <span>JSON Import</span>
          <textarea
            id="import-draft"
            value={importDraft}
            onChange={(event) => setImportDraft(event.target.value)}
            placeholder="Paste /model JSON here..."
            rows={16}
          />
        </label>

        <div className="workflowList">
          <div className="workflowItem">
            <div className="workflowTitle">Counts</div>
            <div className="workflowStats">
              <span>{metrics.attentionCount} attention</span>
              <span>{metrics.mlpCount} mlp</span>
              <span>{metrics.normCount} norm</span>
              <span>{localActivationTotal} activations</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
