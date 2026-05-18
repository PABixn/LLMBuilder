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
import { trainingArtifactDownloadUrl } from "../../../lib/training/artifacts";
import type { TrainingJob } from "../../../lib/training/types";
import type { TrainingJob as TokenizerTrainingJob } from "../../../lib/tokenizerLegacyApi";
import {
  formatModelConfigMeta,
  formatTokenizerMeta,
} from "../lib/display";
import { asString } from "../lib/object";
import type { AssetPickerKind, WorkflowTarget } from "../types";

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
          <button
            type="button"
            className="buttonPrimary"
            onClick={onStartTraining}
            disabled={!startReady}
          >
            <FiPlay /> {launching ? "Starting..." : "Start training"}
          </button>
          <button
            type="button"
            className="buttonDanger"
            onClick={onStopTraining}
            disabled={!activeRunCanBeStopped || stoppingActiveRun}
          >
            <FiXCircle /> {stoppingActiveRun ? "Stopping..." : "Stop training"}
          </button>
          <Link className="buttonGhost" href="/">
            <FiLayers /> Open workspace
          </Link>
          {activeRunId ? (
            <a className="buttonGhost" href={trainingArtifactDownloadUrl(activeRunId)}>
              <FiDownload /> Download bundle
            </a>
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
            <span className="trainingAssetLabel">Model config</span>
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
              <button
                type="button"
                className="buttonGhost buttonSmall"
                aria-haspopup="dialog"
                aria-expanded={pickerKind === "project"}
                onClick={() => onOpenPicker("project")}
              >
                <FiSearch /> {selectedProject ? "Change model" : "Choose model"}
              </button>
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
            <span className="trainingAssetLabel">Tokenizer</span>
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
              <button
                type="button"
                className="buttonGhost buttonSmall"
                aria-haspopup="dialog"
                aria-expanded={pickerKind === "tokenizer"}
                onClick={() => onOpenPicker("tokenizer")}
              >
                <FiSearch /> {selectedTokenizer ? "Change tokenizer" : "Choose tokenizer"}
              </button>
            </div>
          </div>
        </div>

        <div className="fieldGrid compact">
          <label className="fieldLabel trainingRunNameField">
            <span>Run name</span>
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
