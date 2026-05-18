import Link from "next/link";
import { FiActivity, FiLayers, FiRefreshCw, FiSearch, FiXCircle } from "react-icons/fi";

import type { TrainingJob } from "../../../lib/training/types";
import { completedArtifactName, formatJobMeta } from "../lib/formatters";
import { HelpTooltip, InfoTooltip } from "../../shared/components/HelpTooltip";

type InferenceModelPickerDialogProps = {
  open: boolean;
  loading: boolean;
  error: string | null;
  selectedJobId: string;
  completedJobs: TrainingJob[];
  visibleJobs: TrainingJob[];
  query: string;
  searchPlaceholder: string;
  onClose: () => void;
  onQueryChange: (value: string) => void;
  onRefresh: () => void;
  onClearSearch: () => void;
  onSelectJob: (jobId: string) => void;
};

export function InferenceModelPickerDialog({
  open,
  loading,
  error,
  selectedJobId,
  completedJobs,
  visibleJobs,
  query,
  searchPlaceholder,
  onClose,
  onQueryChange,
  onRefresh,
  onClearSearch,
  onSelectJob,
}: InferenceModelPickerDialogProps) {
  if (!open) {
    return null;
  }

  return (
    <div className="trainingAssetPickerOverlay" onClick={onClose} role="presentation">
      <section
        id="inference-model-picker"
        className="panelCard trainingAssetPickerDialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="inference-model-picker-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="trainingAssetPickerHeader">
          <div>
            <h2 id="inference-model-picker-title">
              Choose model
              <InfoTooltip label="Inference model picker explanation" align="left" width="wide">
                <p>
                  Only completed training runs that can be loaded for generation appear here.
                  Selecting one also refreshes checkpoint options for that run.
                </p>
              </InfoTooltip>
            </h2>
            <p className="panelCopy">
              Only completed runs with checkpoints are shown.
            </p>
          </div>
          <button
            type="button"
            className="buttonGhost iconOnly"
            onClick={onClose}
            aria-label="Close model picker"
          >
            <FiXCircle />
          </button>
        </div>

        <div className="trainingAssetPickerControls">
          <label className="trainingAssetPickerSearch">
            <FiSearch />
            <input value={query} onChange={(event) => onQueryChange(event.target.value)} placeholder={searchPlaceholder} />
          </label>
          <HelpTooltip label="Refresh inference models" content="Reloads completed training runs from the backend.">
            <button type="button" className="buttonGhost" onClick={onRefresh} disabled={loading}>
              <FiRefreshCw /> Refresh
            </button>
          </HelpTooltip>
        </div>

        <div className="trainingAssetPickerResults">
          {loading ? <div className="trainingEmpty">Loading models...</div> : null}

          {!loading && error ? <div className="inlineNotice tone-info">{error}</div> : null}

          {!loading && !error && completedJobs.length === 0 ? (
            <div className="trainingAssetPickerEmpty">
              <h3>No models found</h3>
              <p className="panelCopy">
                Finish a training run with a checkpoint first.
              </p>
              <HelpTooltip label="Open training" content="Go to the training page to create a completed run with checkpoints.">
                <Link className="buttonGhost" href="/training">
                  <FiActivity /> Open training
                </Link>
              </HelpTooltip>
            </div>
          ) : null}

          {!loading && !error && completedJobs.length > 0 && visibleJobs.length === 0 ? (
            <div className="trainingAssetPickerEmpty">
              <h3>No matching models</h3>
              <p className="panelCopy">
                Clear search or refresh the list.
              </p>
              <HelpTooltip label="Clear model search" content="Clears the model picker search text so all available completed runs are visible again.">
                <button type="button" className="buttonGhost" onClick={onClearSearch}>
                  <FiLayers /> Clear search
                </button>
              </HelpTooltip>
            </div>
          ) : null}

          {!loading && !error
            ? visibleJobs.map((job) => (
                <button
                  key={job.id}
                  type="button"
                  className={`trainingAssetPickerOption ${
                    selectedJobId === job.id ? "is-selected" : ""
                  }`}
                  onClick={() => onSelectJob(job.id)}
                >
                  <div className="trainingAssetPickerOptionHead">
                    <div>
                      <div className="trainingAssetName">{completedArtifactName(job)}</div>
                    </div>
                    <span
                      className={`pillBadge ${
                        selectedJobId === job.id ? "tone-good" : "tone-neutral"
                      }`}
                    >
                      {selectedJobId === job.id ? "Selected" : "Use model"}
                    </span>
                  </div>
                  <div className="trainingAssetPickerOptionMeta">{formatJobMeta(job)}</div>
                  <div className="trainingAssetPickerOptionMeta">Tokenizer: {job.tokenizer_name}</div>
                  <div className="trainingAssetPickerOptionMeta">
                    {job.artifact_bundle_file ?? job.artifact_dir}
                  </div>
                </button>
              ))
            : null}
        </div>
      </section>
    </div>
  );
}
