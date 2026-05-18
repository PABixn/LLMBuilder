import Link from "next/link";
import { FiActivity, FiRefreshCw, FiSearch } from "react-icons/fi";

import type { TrainingCheckpointEntry, TrainingJob } from "../../../lib/training/types";
import {
  completedArtifactName,
  formatCheckpointMeta,
  formatCheckpointName,
  formatJobMeta,
} from "../lib/formatters";

type TrainingArtifactPanelProps = {
  selectedJob: TrainingJob | null;
  loading: boolean;
  pickerOpen: boolean;
  checkpointPickerOpen: boolean;
  checkpointValue: string;
  latestCheckpointValue: string;
  selectedCheckpoint: TrainingCheckpointEntry | null;
  checkpointsLoading: boolean;
  checkpointError: string | null;
  onRefreshJobs: () => void;
  onOpenModelPicker: () => void;
  onOpenCheckpointPicker: () => void;
};

export function TrainingArtifactPanel({
  selectedJob,
  loading,
  pickerOpen,
  checkpointPickerOpen,
  checkpointValue,
  latestCheckpointValue,
  selectedCheckpoint,
  checkpointsLoading,
  checkpointError,
  onRefreshJobs,
  onOpenModelPicker,
  onOpenCheckpointPicker,
}: TrainingArtifactPanelProps) {
  return (
    <div className="panelCard heroCard inferenceArtifactPanel">
      <div className="panelHead">
        <div>
          <h2>Model</h2>
          <p className="panelCopy">
            Choose a trained model and checkpoint.
          </p>
        </div>
        <div className="actionCluster">
          <button type="button" className="buttonGhost" onClick={onRefreshJobs} disabled={loading}>
            <FiRefreshCw /> Refresh
          </button>
          <Link className="buttonGhost" href="/training">
            <FiActivity /> Training
          </Link>
        </div>
      </div>

      <div className="inferenceAssetStack">
        {selectedJob ? (
          <div className="trainingAssetCard inferenceModelCard">
            <span className="trainingAssetLabel">Model</span>
            <span className="trainingAssetName">{completedArtifactName(selectedJob)}</span>
            <span className="trainingAssetMeta">{formatJobMeta(selectedJob)}</span>
            <span className="trainingAssetMeta">Tokenizer: {selectedJob.tokenizer_name}</span>
            {selectedJob.artifact_bundle_file ? (
              <span className="trainingAssetMeta">{selectedJob.artifact_bundle_file}</span>
            ) : null}
            <div className="trainingAssetActions">
              <button
                type="button"
                className="buttonGhost buttonSmall"
                aria-haspopup="dialog"
                aria-expanded={pickerOpen}
                onClick={onOpenModelPicker}
              >
                <FiSearch /> Change model
              </button>
            </div>
          </div>
        ) : (
          <div className="trainingAssetCard inferenceModelCard">
            <span className="trainingAssetLabel">Model</span>
            <span className="trainingAssetName">No model selected</span>
            <span className="trainingAssetMeta">
              Train a model first, then select it here.
            </span>
            <div className="trainingAssetActions">
              <button
                type="button"
                className="buttonGhost buttonSmall"
                aria-haspopup="dialog"
                aria-expanded={pickerOpen}
                disabled={loading}
                onClick={onOpenModelPicker}
              >
                <FiSearch /> Choose model
              </button>
            </div>
          </div>
        )}

        <div className="trainingAssetCard inferenceCheckpointCard">
          <span className="trainingAssetLabel">Checkpoint</span>
          <span className="trainingAssetName">
            {checkpointValue === latestCheckpointValue
              ? "Latest checkpoint"
              : selectedCheckpoint
                ? formatCheckpointName(selectedCheckpoint)
                : "No checkpoint selected"}
          </span>
          <span className="trainingAssetMeta">
            {checkpointsLoading
              ? "Loading checkpoints..."
              : checkpointError
                ? checkpointError
                : selectedCheckpoint
                  ? formatCheckpointMeta(selectedCheckpoint)
                  : selectedJob
                    ? "This run has no checkpoints."
                    : "Choose a model first."}
          </span>
          {checkpointValue === latestCheckpointValue && selectedCheckpoint ? (
            <span className="trainingAssetMeta">
              Uses {formatCheckpointName(selectedCheckpoint)} now.
            </span>
          ) : null}
          {selectedCheckpoint?.files.length ? (
            <span className="trainingAssetMeta">{selectedCheckpoint.files.join(", ")}</span>
          ) : null}
          <div className="trainingAssetActions">
            <button
              type="button"
              className="buttonGhost buttonSmall"
              aria-haspopup="dialog"
              aria-expanded={checkpointPickerOpen}
              disabled={!selectedJob || checkpointsLoading}
              onClick={onOpenCheckpointPicker}
            >
              <FiSearch /> Choose checkpoint
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
