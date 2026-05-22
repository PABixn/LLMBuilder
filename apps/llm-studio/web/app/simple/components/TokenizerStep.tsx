import { useRef, useState, type ChangeEvent } from "react";
import { FiExternalLink, FiPlay, FiTrash2, FiUpload } from "react-icons/fi";

import type { TokenizerPreviewResult } from "../../../lib/tokenizerLegacyApi";
import { SIMPLE_STARTER_DATASET_NAME } from "../constants";
import {
  isSimpleStreamingDatasetId,
  normalizeSimpleStreamingSelection,
  SIMPLE_STREAMING_DATASETS,
} from "../lib/streamingDatasets";
import { buildSimpleTokenizerProgressState } from "../lib/tokenizerProgress";
import {
  displayTokenLabel,
  makePreviewSegments,
  tokenHue,
} from "../../tokenizer/lib/preview";
import type {
  SimpleDatasetSource,
  SimpleModeController,
  SimpleStreamingDatasetId,
} from "../types";

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
    description: "Choose one source or a light mix",
  },
];

function SimpleTokenizerPreview({
  onPreviewTextChange,
  previewError,
  previewing,
  previewText,
  result,
}: {
  onPreviewTextChange: (value: string) => void;
  previewError: string | null;
  previewing: boolean;
  previewText: string;
  result: TokenizerPreviewResult | null;
}) {
  const [expanded, setExpanded] = useState(true);
  const previewSegments = result ? makePreviewSegments(result.text, result.tokens) : [];
  const tokenCountLabel = result
    ? `${result.num_tokens.toLocaleString()} token${result.num_tokens === 1 ? "" : "s"}`
    : "Preview";

  return (
    <details
      className="simpleTokenizerPreview"
      open={expanded}
      onToggle={(event) => setExpanded(event.currentTarget.open)}
    >
      <summary>
        <span>
          <strong>Token preview</strong>
          <small>
            {previewing ? "Tokenizing..." : result ? tokenCountLabel : "Ready to tokenize"}
          </small>
        </span>
        <span className="simplePreviewBadge">{tokenCountLabel}</span>
      </summary>

      <div className="simpleTokenizerPreviewBody">
        <label className="fieldLabel simplePreviewField">
          <span>Text to tokenize</span>
          <textarea
            value={previewText}
            onChange={(event) => onPreviewTextChange(event.currentTarget.value)}
            placeholder="Type text to tokenize..."
            rows={4}
          />
        </label>

        {previewError ? <p className="inlineNotice tone-error">{previewError}</p> : null}

        {result && result.text.length > 0 ? (
          <>
            <div className="simpleTokenizedText" aria-live="polite">
              {previewSegments.map((segment, segmentIndex) => {
                if (segment.kind === "plain" || !segment.token) {
                  return (
                    <span key={`plain-${segmentIndex}`} className="simplePlainSegment">
                      {segment.text}
                    </span>
                  );
                }

                const token = segment.token;
                const hue = tokenHue(token.index);
                return (
                  <span
                    key={`token-${token.index}-${token.start}-${token.end}`}
                    className="simpleTokenMark"
                    title={`Token ID: ${token.id}`}
                    style={{
                      backgroundColor: `hsla(${hue}, 95%, 55%, 0.2)`,
                      borderColor: `hsla(${hue}, 70%, 55%, 0.7)`,
                    }}
                  >
                    {segment.text}
                  </span>
                );
              })}
            </div>

            <div className="simpleTokenChipList" aria-label="Tokenizer output tokens">
              {result.tokens.map((token) => {
                const hue = tokenHue(token.index);
                return (
                  <span
                    key={`chip-${token.index}-${token.id}`}
                    className="simpleTokenChip"
                    title={`Token ID: ${token.id}`}
                    style={{
                      backgroundColor: `hsla(${hue}, 95%, 55%, 0.17)`,
                      borderColor: `hsla(${hue}, 70%, 55%, 0.62)`,
                    }}
                  >
                    <code>{displayTokenLabel(token.token)}</code>
                  </span>
                );
              })}
            </div>
          </>
        ) : previewText.trim() === "" ? (
          <p className="simpleMuted">Enter text to preview tokens.</p>
        ) : previewing ? (
          <p className="simpleMuted">Tokenizing...</p>
        ) : null}
      </div>
    </details>
  );
}

export function TokenizerStep({ controller }: TokenizerStepProps) {
  const { flow, tokenizerStep, updateFlow } = controller;
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const tokenizerProgress = buildSimpleTokenizerProgressState({
    job: tokenizerStep.tokenizerJob,
    starting: tokenizerStep.training,
    validating: tokenizerStep.validating,
  });
  const tokenizerRunning =
    tokenizerStep.tokenizerJob?.status === "pending" ||
    tokenizerStep.tokenizerJob?.status === "running";
  const tokenizerPreviewReady =
    tokenizerStep.tokenizerJob?.status === "completed" &&
    Boolean(tokenizerStep.tokenizerJob.artifact_file);
  const trainButtonLabel = tokenizerStep.validating
    ? "Checking"
    : tokenizerStep.training
      ? "Starting"
      : tokenizerRunning
        ? "Training"
        : "Train tokenizer";

  const handleFilesSelected = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.currentTarget.files ?? []);
    event.currentTarget.value = "";
    void tokenizerStep.uploadFiles(files);
  };

  const selectStreamingPrimaryDataset = (datasetId: SimpleStreamingDatasetId) => {
    const selection = normalizeSimpleStreamingSelection(
      datasetId,
      flow.streamingAdditionalDatasetIds
    );
    updateFlow((current) => ({
      ...current,
      datasetSource: "streaming",
      streamingPrimaryDatasetId: selection.primaryId,
      streamingAdditionalDatasetIds: selection.additionalIds,
    }));
  };

  const toggleStreamingAdditionalDataset = (
    datasetId: SimpleStreamingDatasetId,
    checked: boolean
  ) => {
    updateFlow((current) => {
      const additions = checked
        ? [...current.streamingAdditionalDatasetIds, datasetId]
        : current.streamingAdditionalDatasetIds.filter((additionalId) => additionalId !== datasetId);
      const selection = normalizeSimpleStreamingSelection(
        current.streamingPrimaryDatasetId,
        additions
      );
      return {
        ...current,
        datasetSource: "streaming",
        streamingPrimaryDatasetId: selection.primaryId,
        streamingAdditionalDatasetIds: selection.additionalIds,
      };
    });
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

        {flow.datasetSource === "streaming" ? (
          <div className="simpleStreamingMixer">
            <label className="fieldLabel">
              <span>Main streaming dataset</span>
              <select
                value={flow.streamingPrimaryDatasetId}
                onChange={(event) => {
                  const datasetId = event.currentTarget.value;
                  if (isSimpleStreamingDatasetId(datasetId)) {
                    selectStreamingPrimaryDataset(datasetId);
                  }
                }}
              >
                {SIMPLE_STREAMING_DATASETS.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.label}
                  </option>
                ))}
              </select>
            </label>

            <div className="simpleCheckboxGrid" aria-label="Additional streaming datasets">
              {SIMPLE_STREAMING_DATASETS.filter(
                (dataset) => dataset.id !== flow.streamingPrimaryDatasetId
              ).map((dataset) => (
                <label key={dataset.id} className="simpleChoiceCheck">
                  <input
                    type="checkbox"
                    checked={flow.streamingAdditionalDatasetIds.includes(dataset.id)}
                    onChange={(event) =>
                      toggleStreamingAdditionalDataset(dataset.id, event.currentTarget.checked)
                    }
                  />
                  <span>
                    <strong>{dataset.label}</strong>
                    <small>{dataset.description}</small>
                  </span>
                </label>
              ))}
            </div>
          </div>
        ) : null}

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
          {tokenizerProgress ? (
            <span className={`simpleStatusPill is-${tokenizerProgress.pillState}`}>
              {tokenizerProgress.pillLabel}
            </span>
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

        {tokenizerProgress ? (
          <div className="simpleTokenizerMonitor" aria-live="polite">
            <div className="simpleProgressHeader">
              <span>
                <strong>{tokenizerProgress.headline}</strong>
                <small>{tokenizerProgress.detail}</small>
              </span>
              <strong className="simpleProgressValue">{tokenizerProgress.progressLabel}</strong>
            </div>
            <div
              className="simpleProgressBar"
              aria-label={`Tokenizer progress ${tokenizerProgress.progressLabel}`}
            >
              <span style={{ width: tokenizerProgress.progressLabel }} />
            </div>
            <div className="simpleSummaryGrid simpleSummaryGridFour">
              <span>
                <strong>{tokenizerProgress.statusLabel}</strong>
                <small>Status</small>
              </span>
              <span>
                <strong>{tokenizerProgress.recordsLabel}</strong>
                <small>Records</small>
              </span>
              <span>
                <strong>{tokenizerProgress.tokensLabel}</strong>
                <small>Tokens</small>
              </span>
              <span>
                <strong>{tokenizerProgress.vocabLabel}</strong>
                <small>Vocab</small>
              </span>
            </div>
          </div>
        ) : null}

        {tokenizerStep.validationMessage ? (
          <div className="inlineNotice tone-success">{tokenizerStep.validationMessage}</div>
        ) : null}
        {tokenizerStep.validationError ? (
          <div className="inlineNotice tone-error">{tokenizerStep.validationError}</div>
        ) : null}
        {tokenizerStep.tokenizerError ? (
          <div className="inlineNotice tone-error">{tokenizerStep.tokenizerError}</div>
        ) : null}

        {tokenizerPreviewReady ? (
          <SimpleTokenizerPreview
            onPreviewTextChange={tokenizerStep.setPreviewText}
            previewError={tokenizerStep.previewError}
            previewing={tokenizerStep.previewing}
            previewText={tokenizerStep.previewText}
            result={tokenizerStep.previewResult}
          />
        ) : null}

        <div className="simpleActionRow">
          <button
            type="button"
            className="buttonPrimary"
            disabled={
              tokenizerStep.training ||
              tokenizerStep.validating ||
              tokenizerRunning ||
              !flow.projectId ||
              !tokenizerStep.datasetReady
            }
            onClick={() => void tokenizerStep.startTokenizerTraining()}
          >
            <FiPlay aria-hidden="true" />
            {trainButtonLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
