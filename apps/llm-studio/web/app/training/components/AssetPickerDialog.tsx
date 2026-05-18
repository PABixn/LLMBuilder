import Link from "next/link";
import {
  FiCpu,
  FiLayers,
  FiRefreshCw,
  FiSearch,
  FiXCircle,
} from "react-icons/fi";

import type { ProjectSummary } from "../../../lib/api";
import type { TrainingJob as TokenizerTrainingJob } from "../../../lib/tokenizerLegacyApi";
import {
  formatBytes,
  formatDate,
} from "../../../lib/workspaceAssets";
import { formatInteger } from "../lib/metrics";
import {
  asString,
} from "../lib/object";
import type { AssetPickerKind } from "../types";
import { HelpTooltip, InfoTooltip } from "../../shared/components/HelpTooltip";

interface AssetPickerDialogProps {
  onClose: () => void;
  onOpenPicker: (kind: AssetPickerKind) => void;
  onProjectSelected: (projectId: string) => void;
  onQueryChange: (value: string) => void;
  onTokenizerSelected: (jobId: string) => void;
  pickerError: string | null;
  pickerKind: AssetPickerKind | null;
  pickerLoading: boolean;
  pickerQuery: string;
  selectedProjectId: string | null;
  selectedTokenizerJobId: string | null;
  visiblePickerProjects: ProjectSummary[];
  visiblePickerTokenizerJobs: TokenizerTrainingJob[];
}

export function AssetPickerDialog({
  onClose,
  onOpenPicker,
  onProjectSelected,
  onQueryChange,
  onTokenizerSelected,
  pickerError,
  pickerKind,
  pickerLoading,
  pickerQuery,
  selectedProjectId,
  selectedTokenizerJobId,
  visiblePickerProjects,
  visiblePickerTokenizerJobs,
}: AssetPickerDialogProps) {
  if (!pickerKind) {
    return null;
  }

  return (
    <div
      className="trainingAssetPickerOverlay"
      onClick={onClose}
      role="presentation"
    >
      <section
        id="training-asset-picker"
        className="panelCard trainingAssetPickerDialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="training-asset-picker-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="trainingAssetPickerHeader">
          <div>
            <h2 id="training-asset-picker-title">
              {pickerKind === "project" ? "Choose model" : "Choose tokenizer"}
              <InfoTooltip label="Training asset picker explanation" align="left" width="wide">
                <p>
                  This picker only changes the asset selected for the training page. It does
                  not edit the source model config or tokenizer job.
                </p>
              </InfoTooltip>
            </h2>
            <p className="panelCopy">
              {pickerKind === "project"
                ? "Select a saved model config."
                : "Only completed tokenizers are shown."}
            </p>
          </div>
          <button
            type="button"
            className="buttonGhost iconOnly"
            onClick={onClose}
            aria-label="Close asset picker"
          >
            <FiXCircle />
          </button>
        </div>

        <div className="trainingAssetPickerControls">
          <label className="trainingAssetPickerSearch">
            <FiSearch />
            <input
              value={pickerQuery}
              onChange={(event) => onQueryChange(event.target.value)}
              placeholder={
                pickerKind === "project"
                  ? "Search models"
                  : "Search tokenizers"
              }
            />
          </label>
          <HelpTooltip label="Refresh picker results" content="Reloads the workspace inventory for this picker while keeping the current search text.">
            <button
              type="button"
              className="buttonGhost"
              onClick={() => {
                onOpenPicker(pickerKind);
              }}
            >
              <FiRefreshCw /> Refresh
            </button>
          </HelpTooltip>
        </div>

        <div className="trainingAssetPickerResults">
          {pickerLoading ? <div className="trainingEmpty">Loading workspace...</div> : null}

          {!pickerLoading && pickerError ? (
            <div className="inlineNotice tone-info">{pickerError}</div>
          ) : null}

          {!pickerLoading && !pickerError && pickerKind === "project" && visiblePickerProjects.length === 0 ? (
            <div className="trainingAssetPickerEmpty">
              <h3>No models found</h3>
              <p className="panelCopy">
                Save a model config first.
              </p>
              <HelpTooltip label="Open workspace from picker" content="Go to the workspace to create or manage saved model configs.">
                <Link className="buttonGhost" href="/">
                  <FiLayers /> Open workspace
                </Link>
              </HelpTooltip>
            </div>
          ) : null}

          {!pickerLoading && !pickerError && pickerKind === "tokenizer" && visiblePickerTokenizerJobs.length === 0 ? (
            <div className="trainingAssetPickerEmpty">
              <h3>No tokenizers found</h3>
              <p className="panelCopy">
                Finish tokenizer training first.
              </p>
              <HelpTooltip label="Open tokenizer from picker" content="Go to tokenizer training to create a completed tokenizer artifact.">
                <Link className="buttonGhost" href="/tokenizer">
                  <FiCpu /> Open tokenizer
                </Link>
              </HelpTooltip>
            </div>
          ) : null}

          {!pickerLoading && !pickerError && pickerKind === "project"
            ? visiblePickerProjects.map((project) => (
                <button
                  key={project.id}
                  type="button"
                  className={`trainingAssetPickerOption ${
                    selectedProjectId === project.id ? "is-selected" : ""
                  }`}
                  onClick={() => {
                    onProjectSelected(project.id);
                    onClose();
                  }}
                >
                  <div className="trainingAssetPickerOptionHead">
                    <div>
                      <div className="trainingAssetName">
                        {project.name ?? project.id}
                      </div>
                    </div>
                    <span
                      className={`pillBadge ${
                        selectedProjectId === project.id ? "tone-good" : "tone-neutral"
                      }`}
                    >
                      {selectedProjectId === project.id ? "Selected" : "Use model"}
                    </span>
                  </div>
                  <div className="trainingAssetPickerOptionMeta">
                    {formatDate(project.created_at)} • {formatBytes(project.size_bytes)}
                  </div>
                  <div className="trainingAssetPickerOptionMeta">{project.artifact_file}</div>
                </button>
              ))
            : null}

          {!pickerLoading && !pickerError && pickerKind === "tokenizer"
            ? visiblePickerTokenizerJobs.map((job) => (
                <button
                  key={job.id}
                  type="button"
                  className={`trainingAssetPickerOption ${
                    selectedTokenizerJobId === job.id ? "is-selected" : ""
                  }`}
                  onClick={() => {
                    onTokenizerSelected(job.id);
                    onClose();
                  }}
                >
                  <div className="trainingAssetPickerOptionHead">
                    <div>
                      <div className="trainingAssetName">
                        {asString(job.tokenizer_config.name, job.id)}
                      </div>
                    </div>
                    <span
                      className={`pillBadge ${
                        selectedTokenizerJobId === job.id ? "tone-good" : "tone-neutral"
                      }`}
                    >
                      {selectedTokenizerJobId === job.id ? "Selected" : "Use tokenizer"}
                    </span>
                  </div>
                  <div className="trainingAssetPickerOptionMeta">
                    {formatDate(job.created_at)}
                    {job.stats?.vocab_size ? ` • vocab ${formatInteger(job.stats.vocab_size)}` : ""}
                  </div>
                  <div className="trainingAssetPickerOptionMeta">
                    {job.artifact_file ?? job.artifact_path ?? "Tokenizer path unavailable"}
                  </div>
                </button>
              ))
            : null}
        </div>
      </section>
    </div>
  );
}
