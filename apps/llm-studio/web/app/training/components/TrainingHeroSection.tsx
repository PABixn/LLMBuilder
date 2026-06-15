import Link from "next/link";
import type { RefObject } from "react";
import {
  FiDownload,
  FiLayers,
  FiPlay,
  FiSearch,
  FiXCircle,
} from "react-icons/fi";

import type { ProjectDetail } from "../../../lib/api";
import { downloadApiArtifact } from "../../../lib/downloads";
import type { TrainingJob } from "../../../lib/training/types";
import type { TrainingJob as TokenizerTrainingJob } from "../../../lib/tokenizerLegacyApi";
import {
  formatModelConfigMeta,
  formatTokenizerMeta,
} from "../lib/display";
import { asString } from "../lib/object";
import type { AssetPickerKind, WorkflowTarget } from "../types";
import { FieldLabelText, HelpTooltip, InfoTooltip } from "../../shared/components/HelpTooltip";

type TrainingHeroSectionProps = {
  selectedProject: ProjectDetail | null;
  selectedProjectId: string | null;
  selectedTokenizer: TokenizerTrainingJob | null;
  selectedTokenizerJobId: string | null;
  preflightValid: boolean;
  preflightLoading: boolean;
  activeRun: TrainingJob | null;
  activeRunId: string | null;
  startReady: boolean;
  launching: boolean;
  activeRunCanBeStopped: boolean;
  stoppingActiveRun: boolean;
  pickerKind: AssetPickerKind | null;
  highlightedWorkflowTarget: WorkflowTarget | null;
  runName: string;
  modelSelectionRef: RefObject<HTMLDivElement | null>;
  tokenizerSelectionRef: RefObject<HTMLDivElement | null>;
  onStartTraining: () => void;
  onStopTraining: () => void;
  onOpenPicker: (kind: AssetPickerKind) => void;
  onRunNameChange: (value: string) => void;
};

export function TrainingHeroSection({
  selectedProject,
  selectedProjectId,
  selectedTokenizer,
  selectedTokenizerJobId,
  preflightValid,
  preflightLoading,
  activeRun,
  activeRunId,
  startReady,
  launching,
  activeRunCanBeStopped,
  stoppingActiveRun,
  pickerKind,
  highlightedWorkflowTarget,
  runName,
  modelSelectionRef,
  tokenizerSelectionRef,
  onStartTraining,
  onStopTraining,
  onOpenPicker,
  onRunNameChange,
}: TrainingHeroSectionProps) {
  return (
    <section className="panelCard heroCard trainingHero">
      <div className="panelHead heroHead">
        <div>
          <h1>Train a model.</h1>
          <p className="panelCopy">
            Choose a model and tokenizer, validate the run, then track progress.
          </p>
        </div>
        <div className="actionCluster trainingHeroActions">
          <HelpTooltip label="Start training explanation" align="right" width="wide" content="Starts a training job only after a model config, completed tokenizer, valid dataset, and passing preflight are available. The job uses the current settings shown on this page.">
            <button
              type="button"
              className="buttonPrimary"
              onClick={onStartTraining}
              disabled={!startReady}
            >
              <FiPlay /> {launching ? "Starting..." : "Start training"}
            </button>
          </HelpTooltip>
          <HelpTooltip label="Stop training explanation" align="right" content="Requests cancellation for the active run when the backend says it can be stopped. Completed, failed, or already stopping runs cannot be stopped again.">
            <button
              type="button"
              className="buttonDanger"
              onClick={onStopTraining}
              disabled={!activeRunCanBeStopped || stoppingActiveRun}
            >
              <FiXCircle /> {stoppingActiveRun ? "Stopping..." : "Stop training"}
            </button>
          </HelpTooltip>
          <Link className="buttonGhost" href="/">
            <FiLayers /> Open workspace
          </Link>
          {activeRunId ? (
            <button
              type="button"
              className="buttonGhost"
              onClick={() =>
                void downloadApiArtifact(
                  `/training/jobs/${activeRunId}/artifact`,
                  activeRun?.artifact_bundle_file || `training-${activeRunId}.tar.gz`
                ).catch((error: unknown) =>
                  alert(error instanceof Error ? error.message : "Could not download training bundle.")
                )
              }
            >
              <FiDownload /> Download bundle
            </button>
          ) : null}
        </div>
      </div>

      <div className="heroMetaRow" aria-label="Training launch summary">
        <div className="heroMetaPills">
          <span className={`pillBadge ${selectedProject ? "tone-good" : "tone-warn"}`}>
            {selectedProject ? "Model selected" : "Model needed"}
          </span>
          <span className={`pillBadge ${selectedTokenizer?.status === "completed" ? "tone-good" : "tone-warn"}`}>
            {selectedTokenizer?.status === "completed" ? "Tokenizer ready" : "Tokenizer needed"}
          </span>
          <span className={`pillBadge ${preflightValid ? "tone-good" : preflightLoading ? "tone-neutral" : "tone-warn"}`}>
            {preflightLoading ? "Checking run" : preflightValid ? "Run is valid" : "Run blocked"}
          </span>
        </div>
        <div className="heroMetaLine">
          <span>{selectedProject?.name ?? "No model config selected"}</span>
          <span className="heroMetaSeparator" aria-hidden>
            •
          </span>
          <span>
            {selectedTokenizer
              ? asString(selectedTokenizer.tokenizer_config.name, selectedTokenizer.id)
              : "No tokenizer selected"}
          </span>
          <span className="heroMetaSeparator" aria-hidden>
            •
          </span>
          <span>{activeRun ? `Tracking ${activeRun.name}` : "Ready for a new run"}</span>
        </div>
      </div>

      <div className="trainingHeroBody">
        <div className="trainingPairGrid">
          <div
            id="settings-model"
            ref={modelSelectionRef}
            className={`trainingAssetCard settingsCategoryAnchor ${
              highlightedWorkflowTarget === "model"
                ? "settingsCategoryAnchor-highlight"
                : ""
            }`}
          >
            <span className="trainingAssetLabel">
              Model config
              <InfoTooltip label="Model config explanation" align="left" width="wide">
                <strong>Model config</strong>
                <p>
                  The architecture JSON from Model Studio. Training uses its vocabulary size,
                  context length, embedding size, blocks, and component layout.
                </p>
              </InfoTooltip>
            </span>
            <span className="trainingAssetName">
              {selectedProject?.name ?? (selectedProjectId ? selectedProjectId : "No model selected")}
            </span>
            <span className="trainingAssetMeta">
              {selectedProject
                ? formatModelConfigMeta(selectedProject.model_config)
                : "Choose a saved model config."}
            </span>
            {selectedProject ? (
              <span className="trainingAssetMeta">
                {selectedProject.artifact_file || "Saved model config"}
              </span>
            ) : null}
            <div className="trainingAssetActions">
              <HelpTooltip label="Choose model explanation" content="Opens saved Model Studio configs from the workspace. Pick the architecture you want this training run to update." align="right">
                <button
                  type="button"
                  className="buttonGhost buttonSmall"
                  aria-haspopup="dialog"
                  aria-expanded={pickerKind === "project"}
                  onClick={() => onOpenPicker("project")}
                >
                  <FiSearch /> {selectedProject ? "Change model" : "Choose model"}
                </button>
              </HelpTooltip>
            </div>
          </div>
          <div
            id="settings-tokenizer"
            ref={tokenizerSelectionRef}
            className={`trainingAssetCard settingsCategoryAnchor ${
              highlightedWorkflowTarget === "tokenizer"
                ? "settingsCategoryAnchor-highlight"
                : ""
            }`}
          >
            <span className="trainingAssetLabel">
              Tokenizer
              <InfoTooltip label="Tokenizer explanation" align="left" width="wide">
                <strong>Tokenizer</strong>
                <p>
                  Converts text to token IDs for training. It must be completed and its vocabulary
                  size must match the selected model config.
                </p>
              </InfoTooltip>
            </span>
            <span className="trainingAssetName">
              {selectedTokenizer
                ? asString(selectedTokenizer.tokenizer_config.name, selectedTokenizer.id)
                : selectedTokenizerJobId ?? "No tokenizer selected"}
            </span>
            <span className="trainingAssetMeta">
              {selectedTokenizer
                ? formatTokenizerMeta(selectedTokenizer)
                : "Choose a completed tokenizer."}
            </span>
            {selectedTokenizer ? (
              <span className="trainingAssetMeta">
                {selectedTokenizer.artifact_file ??
                  selectedTokenizer.artifact_path ??
                  "Tokenizer path unavailable"}
              </span>
            ) : null}
            <div className="trainingAssetActions">
              <HelpTooltip label="Choose tokenizer explanation" content="Opens completed tokenizer jobs from the workspace. Training can only start with a tokenizer artifact that exists and passes compatibility checks." align="right">
                <button
                  type="button"
                  className="buttonGhost buttonSmall"
                  aria-haspopup="dialog"
                  aria-expanded={pickerKind === "tokenizer"}
                  onClick={() => onOpenPicker("tokenizer")}
                >
                  <FiSearch /> {selectedTokenizer ? "Change tokenizer" : "Choose tokenizer"}
                </button>
              </HelpTooltip>
            </div>
          </div>
        </div>

        <div className="fieldGrid compact">
          <label className="fieldLabel trainingRunNameField">
            <FieldLabelText tooltipLabel="Run name explanation" tooltip="Optional human-readable name saved with the run. If left blank, the app generates a default name from the selected model and tokenizer.">
              Run name
            </FieldLabelText>
            <input
              value={runName}
              onChange={(event) => onRunNameChange(event.target.value)}
              placeholder="Optional run name"
            />
          </label>
        </div>
      </div>
    </section>
  );
}
