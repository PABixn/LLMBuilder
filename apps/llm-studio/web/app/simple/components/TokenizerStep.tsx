import { useRef, type ChangeEvent } from "react";
import { FiCheckCircle, FiExternalLink, FiPlay, FiTrash2, FiUpload } from "react-icons/fi";

import {
  SIMPLE_STARTER_DATASET_NAME,
  SIMPLE_STREAMING_DATASET_NAME,
} from "../constants";
import type { SimpleDatasetSource, SimpleModeController } from "../types";

interface TokenizerStepProps {
  controller: SimpleModeController;
}

const DATASET_OPTIONS: Array<{ id: SimpleDatasetSource; label: string; description: string }> = [
  {
    id: "starter",
    label: "Use starter dataset",
    description: SIMPLE_STARTER_DATASET_NAME,
  },
  {
    id: "upload",
    label: "Upload text files",
    description: "Use your own local data",
  },
  {
    id: "streaming",
    label: "Use streaming template",
    description: SIMPLE_STREAMING_DATASET_NAME,
  },
];

export function TokenizerStep({ controller }: TokenizerStepProps) {
  const { flow, tokenizerStep, updateFlow } = controller;
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFilesSelected = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.currentTarget.files ?? []);
    event.currentTarget.value = "";
    void tokenizerStep.uploadFiles(files);
  };

  return (
    <div className="simpleStepGrid">
      <div className="simplePanel">
        <div className="simplePanelHeader">
          <div>
            <p className="simpleEyebrow">Data</p>
            <h3>Choose text data</h3>
          </div>
          <a className="buttonGhost buttonSmall" href="/tokenizer">
            <FiExternalLink aria-hidden="true" /> Expert
          </a>
        </div>

        <div className="simpleSegmented">
          {DATASET_OPTIONS.map((option) => (
            <button
              key={option.id}
              type="button"
              className={flow.datasetSource === option.id ? "is-selected" : ""}
              onClick={() =>
                updateFlow((current) => ({
                  ...current,
                  datasetSource: option.id,
                }))
              }
            >
              <strong>{option.label}</strong>
              <span>{option.description}</span>
            </button>
          ))}
        </div>

        {flow.datasetSource === "upload" ? (
          <div className="simpleUploadBox">
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt,text/plain"
              multiple
              className="simpleHiddenInput"
              onChange={handleFilesSelected}
            />
            <button
              type="button"
              className="buttonGhost"
              disabled={tokenizerStep.uploading}
              onClick={() => fileInputRef.current?.click()}
            >
              <FiUpload aria-hidden="true" />
              {tokenizerStep.uploading ? "Uploading" : "Add text files"}
            </button>
            {flow.localTrainFiles.length > 0 ? (
              <div className="simpleFileList">
                {flow.localTrainFiles.map((file) => (
                  <div key={file.id} className="simpleFileRow">
                    <span>
                      <strong>{file.fileName}</strong>
                      <small>{file.filePath}</small>
                    </span>
                    <button
                      type="button"
                      className="buttonGhost iconOnly"
                      aria-label={`Remove ${file.fileName}`}
                      onClick={() => tokenizerStep.removeLocalFile(file.id)}
                    >
                      <FiTrash2 aria-hidden="true" />
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="simpleMuted">No uploaded files yet.</p>
            )}
          </div>
        ) : null}
      </div>

      <div className="simplePanel">
        <div className="simplePanelHeader">
          <div>
            <p className="simpleEyebrow">Tokenizer</p>
            <h3>Train tokenizer</h3>
          </div>
          {tokenizerStep.tokenizerJob?.status === "completed" ? (
            <span className="simpleStatusPill is-completed">Complete</span>
          ) : null}
        </div>

        <div className="simpleSummaryGrid">
          <span>
            <strong>{flow.targetVocabSize.toLocaleString()}</strong>
            <small>Vocabulary target</small>
          </span>
          <span>
            <strong>BPE byte-level</strong>
            <small>Hidden default</small>
          </span>
          <span>
            <strong>Auto matched</strong>
            <small>Model vocab sync</small>
          </span>
        </div>

        {tokenizerStep.validationMessage ? (
          <div className="inlineNotice tone-success">{tokenizerStep.validationMessage}</div>
        ) : null}
        {tokenizerStep.validationError ? (
          <div className="inlineNotice tone-error">{tokenizerStep.validationError}</div>
        ) : null}
        {tokenizerStep.tokenizerError ? (
          <div className="inlineNotice tone-error">{tokenizerStep.tokenizerError}</div>
        ) : null}

        {tokenizerStep.previewResult ? (
          <div className="simpleOutputBox">
            <strong>Preview</strong>
            <span>
              {tokenizerStep.previewResult.num_tokens.toLocaleString()} tokens from the default prompt.
            </span>
          </div>
        ) : null}

        <div className="simpleActionRow">
          <button
            type="button"
            className="buttonGhost"
            disabled={tokenizerStep.validating || !flow.projectId}
            onClick={() => void tokenizerStep.validateTokenizer()}
          >
            <FiCheckCircle aria-hidden="true" />
            {tokenizerStep.validating ? "Validating" : "Validate tokenizer"}
          </button>
          <button
            type="button"
            className="buttonPrimary"
            disabled={tokenizerStep.training || !flow.projectId || !tokenizerStep.datasetReady}
            onClick={() => void tokenizerStep.startTokenizerTraining()}
          >
            <FiPlay aria-hidden="true" />
            {tokenizerStep.training ? "Starting" : "Train tokenizer"}
          </button>
        </div>

        <details className="simpleDetails">
          <summary>Advanced details</summary>
          <pre>{JSON.stringify({ tokenizer: tokenizerStep.tokenizerConfig, dataloader: tokenizerStep.dataloaderConfig }, null, 2)}</pre>
        </details>
      </div>
    </div>
  );
}
