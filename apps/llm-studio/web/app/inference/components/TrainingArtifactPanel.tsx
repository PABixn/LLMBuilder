import Link from "next/link";
import { FiActivity, FiRefreshCw, FiSearch } from "react-icons/fi";

import type { TrainingCheckpointEntry, TrainingJob } from "../../../lib/training/types";
import {
  completedArtifactName,
  formatCheckpointMeta,
  formatCheckpointName,
  formatJobMeta,
} from "../lib/formatters";
import { HelpTooltip, InfoTooltip } from "../../shared/components/HelpTooltip";

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
          <h2>
            Model
            <InfoTooltip label="Inference model panel explanation" align="left" width="wide">
              <p>
                Inference uses a completed training run and one checkpoint. The model config,
                tokenizer, and checkpoint weights are loaded together by the backend.
              </p>
            </InfoTooltip>
          </h2>
          <p className="panelCopy">
            Choose a trained model and checkpoint.
          </p>
        </div>
        <div className="actionCluster">
          <HelpTooltip label="Refresh trained models" content="Reloads the completed training runs and checkpoint metadata from the backend.">
            <button type="button" className="buttonGhost" onClick={onRefreshJobs} disabled={loading}>
              <FiRefreshCw /> Refresh
            </button>
          </HelpTooltip>
          <Link className="buttonGhost" href="/training">
            <FiActivity /> Training
          </Link>
        </div>
      </div>

      <div className="inferenceAssetStack">
        {selectedJob ? (
          <div className="trainingAssetCard inferenceModelCard">
            <span className="trainingAssetLabel">
              Model
              <InfoTooltip label="Selected inference model explanation" align="left">
                <p>Completed training run whose artifact bundle will be used for generation.</p>
              </InfoTooltip>
            </span>
            <span className="trainingAssetName">{completedArtifactName(selectedJob)}</span>
            <span className="trainingAssetMeta">{formatJobMeta(selectedJob)}</span>
            <span className="trainingAssetMeta">Tokenizer: {selectedJob.tokenizer_name}</span>
            {selectedJob.artifact_bundle_file ? (
              <span className="trainingAssetMeta">{selectedJob.artifact_bundle_file}</span>
            ) : null}
            <div className="trainingAssetActions">
              <HelpTooltip label="Change inference model" content="Opens completed training runs that can be loaded for generation.">
                <button
                  type="button"
                  className="buttonGhost buttonSmall"
                  aria-haspopup="dialog"
                  aria-expanded={pickerOpen}
                  onClick={onOpenModelPicker}
                >
                  <FiSearch /> Change model
                </button>
              </HelpTooltip>
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
              <HelpTooltip label="Choose inference model" content="Select a completed training run. Inference is unavailable until at least one completed run with artifacts exists.">
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
              </HelpTooltip>
            </div>
          </div>
        )}

        <div className="trainingAssetCard inferenceCheckpointCard">
          <span className="trainingAssetLabel">
            Checkpoint
            <InfoTooltip label="Checkpoint explanation" align="left" width="wide">
              <p>
                A saved set of model weights from a training step. Latest checkpoint follows
                the newest saved step, while a specific checkpoint stays pinned.
              </p>
            </InfoTooltip>
          </span>
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
            <HelpTooltip label="Choose checkpoint" content="Pick the exact saved step to use for generation, or use Latest so generation follows the newest checkpoint available.">
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
            </HelpTooltip>
          </div>
        </div>
      </div>
    </div>
  );
}
