import Link from "next/link";
import { FiActivity, FiLayers, FiRefreshCw, FiSearch, FiXCircle } from "react-icons/fi";

import type { TrainingCheckpointEntry, TrainingJob } from "../../../lib/training/types";
import {
  checkpointOptionValue,
  formatCheckpointMeta,
  formatCheckpointName,
} from "../lib/formatters";

type InferenceCheckpointPickerDialogProps = {
  open: boolean;
  selectedJob: TrainingJob | null;
  checkpointsLoading: boolean;
  checkpointError: string | null;
  checkpoints: TrainingCheckpointEntry[];
  latestCheckpoint: TrainingCheckpointEntry | null;
  visibleCheckpointOptions: TrainingCheckpointEntry[];
  checkpointValue: string;
  latestCheckpointValue: string;
  showLatestCheckpointOption: boolean;
  query: string;
  searchPlaceholder: string;
  onClose: () => void;
  onQueryChange: (value: string) => void;
  onRefresh: () => void;
  onClearSearch: () => void;
  onSelectCheckpoint: (value: string) => void;
};

export function InferenceCheckpointPickerDialog({
  open,
  selectedJob,
  checkpointsLoading,
  checkpointError,
  checkpoints,
  latestCheckpoint,
  visibleCheckpointOptions,
  checkpointValue,
  latestCheckpointValue,
  showLatestCheckpointOption,
  query,
  searchPlaceholder,
  onClose,
  onQueryChange,
  onRefresh,
  onClearSearch,
  onSelectCheckpoint,
}: InferenceCheckpointPickerDialogProps) {
  if (!open) {
    return null;
  }

  return (
    <div className="trainingAssetPickerOverlay" onClick={onClose} role="presentation">
      <section
        id="inference-checkpoint-picker"
        className="panelCard trainingAssetPickerDialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="inference-checkpoint-picker-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="trainingAssetPickerHeader">
          <div>
            <h2 id="inference-checkpoint-picker-title">Choose checkpoint</h2>
            <p className="panelCopy">
              Use the latest checkpoint or choose a saved step.
            </p>
          </div>
          <button
            type="button"
            className="buttonGhost iconOnly"
            onClick={onClose}
            aria-label="Close checkpoint picker"
          >
            <FiXCircle />
          </button>
        </div>

        <div className="trainingAssetPickerControls">
          <label className="trainingAssetPickerSearch">
            <FiSearch />
            <input value={query} onChange={(event) => onQueryChange(event.target.value)} placeholder={searchPlaceholder} />
          </label>
          <button
            type="button"
            className="buttonGhost"
            disabled={!selectedJob || checkpointsLoading}
            onClick={onRefresh}
          >
            <FiRefreshCw /> Refresh
          </button>
        </div>

        <div className="trainingAssetPickerResults">
          {checkpointsLoading ? <div className="trainingEmpty">Loading checkpoints...</div> : null}

          {!checkpointsLoading && checkpointError ? (
            <div className="inlineNotice tone-info">{checkpointError}</div>
          ) : null}

          {!checkpointsLoading && !checkpointError && checkpoints.length === 0 ? (
            <div className="trainingAssetPickerEmpty">
              <h3>No checkpoints found</h3>
              <p className="panelCopy">
                Choose a run that saved checkpoints.
              </p>
              <Link className="buttonGhost" href="/training">
                <FiActivity /> Open training
              </Link>
            </div>
          ) : null}

          {!checkpointsLoading && !checkpointError && checkpoints.length > 0 ? (
            <>
              {showLatestCheckpointOption ? (
                <button
                  type="button"
                  className={`trainingAssetPickerOption ${
                    checkpointValue === latestCheckpointValue ? "is-selected" : ""
                  }`}
                  onClick={() => onSelectCheckpoint(latestCheckpointValue)}
                >
                  <div className="trainingAssetPickerOptionHead">
                    <div>
                      <div className="trainingAssetName">Latest checkpoint</div>
                    </div>
                    <span
                      className={`pillBadge ${
                        checkpointValue === latestCheckpointValue ? "tone-good" : "tone-neutral"
                      }`}
                    >
                      {checkpointValue === latestCheckpointValue ? "Selected" : "Use latest"}
                    </span>
                  </div>
                  <div className="trainingAssetPickerOptionMeta">
                    Uses the newest saved step when you generate.
                  </div>
                  {latestCheckpoint ? (
                    <div className="trainingAssetPickerOptionMeta">
                      Latest now: {formatCheckpointName(latestCheckpoint)} |{" "}
                      {formatCheckpointMeta(latestCheckpoint)}
                    </div>
                  ) : null}
                </button>
              ) : null}

              {!showLatestCheckpointOption && visibleCheckpointOptions.length === 0 ? (
                <div className="trainingAssetPickerEmpty">
                  <h3>No matching checkpoints</h3>
                  <p className="panelCopy">
                    Clear search to see all checkpoints.
                  </p>
                  <button type="button" className="buttonGhost" onClick={onClearSearch}>
                    <FiLayers /> Clear search
                  </button>
                </div>
              ) : null}

              {visibleCheckpointOptions.map((checkpoint) => {
                const value = checkpointOptionValue(checkpoint);
                return (
                  <button
                    key={checkpoint.directory}
                    type="button"
                    className={`trainingAssetPickerOption ${
                      checkpointValue === value ? "is-selected" : ""
                    }`}
                    onClick={() => onSelectCheckpoint(value)}
                  >
                    <div className="trainingAssetPickerOptionHead">
                      <div>
                        <div className="trainingAssetName">{formatCheckpointName(checkpoint)}</div>
                      </div>
                      <span
                        className={`pillBadge ${
                          checkpointValue === value ? "tone-good" : "tone-neutral"
                        }`}
                      >
                        {checkpointValue === value ? "Selected" : "Use checkpoint"}
                      </span>
                    </div>
                    <div className="trainingAssetPickerOptionMeta">
                      {formatCheckpointMeta(checkpoint)}
                    </div>
                    <div className="trainingAssetPickerOptionMeta">
                      {checkpoint.files.join(", ") || checkpoint.directory}
                    </div>
                  </button>
                );
              })}
            </>
          ) : null}
        </div>
      </section>
    </div>
  );
}
