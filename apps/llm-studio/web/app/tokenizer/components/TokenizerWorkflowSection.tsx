import type { JobState } from "../../../lib/tokenizerLegacyApi";
import type {
  BudgetUnit,
  DatasetSourceMode,
  SettingsCategory,
  TokenizerType,
} from "../types";
import { describeJobState } from "../lib/display";

type TokenizerWorkflowSectionProps = {
  tokenizerReady: boolean;
  tokenizerError: string | null;
  tokenizerType: TokenizerType;
  datasetReady: boolean;
  datasetError: string | null;
  datasetSourceMode: DatasetSourceMode;
  localTrainFilesHint: string;
  streamingDatasetCount: number;
  trainingRuntimeReady: boolean;
  trainingRuntimeError: string | null;
  budgetLimit: string;
  budgetUnit: BudgetUnit;
  activeThresholds: string | null;
  hasValidationPassed: boolean;
  isValidating: boolean;
  validationError: string | null;
  preflightReady: boolean;
  controlsDisabled: boolean;
  trainingCompleted: boolean;
  hasTrainingInProgress: boolean;
  activeJobState: JobState | null;
  isSubmitting: boolean;
  canStartTraining: boolean;
  onNavigateSettings: (category: SettingsCategory) => void;
  onValidate: () => void;
  onTrain: () => void;
};

export function TokenizerWorkflowSection({
  tokenizerReady,
  tokenizerError,
  tokenizerType,
  datasetReady,
  datasetError,
  datasetSourceMode,
  localTrainFilesHint,
  streamingDatasetCount,
  trainingRuntimeReady,
  trainingRuntimeError,
  budgetLimit,
  budgetUnit,
  activeThresholds,
  hasValidationPassed,
  isValidating,
  validationError,
  preflightReady,
  controlsDisabled,
  trainingCompleted,
  hasTrainingInProgress,
  activeJobState,
  isSubmitting,
  canStartTraining,
  onNavigateSettings,
  onValidate,
  onTrain,
}: TokenizerWorkflowSectionProps) {
  return (
    <section id="workflow" className="panelCard actionDeck">
      <div className="panelHead actionDeckHead">
        <div>
          <p className="panelEyebrow">Workflow</p>
          <h2>Train tokenizer</h2>
          <p className="panelCopy">
            Complete each step, then start training.
          </p>
        </div>
      </div>

      <div className="workflowStepGrid" role="list" aria-label="Tokenizer training steps">
        <article
          className={`workflowStepTile ${
            tokenizerReady ? "workflowStepTile-ready" : "workflowStepTile-waiting"
          }`}
          role="listitem"
        >
          <p className="workflowStepTitle">1. Tokenizer</p>
          <strong>{tokenizerReady ? "Ready" : "Needs setup"}</strong>
          <p className="fieldNote">
            {tokenizerError ??
              `${tokenizerType.toUpperCase()} tokenizer configured.`}
          </p>
          <button
            type="button"
            className="workflowStepLink workflowStepAction"
            onClick={() => onNavigateSettings("tokenizer")}
          >
            Open settings
          </button>
        </article>

        <article
          className={`workflowStepTile ${
            datasetReady ? "workflowStepTile-ready" : "workflowStepTile-waiting"
          }`}
          role="listitem"
        >
          <p className="workflowStepTitle">2. Dataset</p>
          <strong>{datasetReady ? "Ready" : "Needs setup"}</strong>
          <p className="fieldNote">
            {datasetError ??
              (datasetSourceMode === "local_file"
                ? localTrainFilesHint
                : `${streamingDatasetCount} streaming dataset${
                    streamingDatasetCount === 1 ? "" : "s"
                  } configured.`)}
          </p>
          <button
            type="button"
            className="workflowStepLink workflowStepAction"
            onClick={() => onNavigateSettings("dataset")}
          >
            Open dataset
          </button>
        </article>

        <article
          className={`workflowStepTile ${
            trainingRuntimeReady
              ? "workflowStepTile-ready"
              : "workflowStepTile-waiting"
          }`}
          role="listitem"
        >
          <p className="workflowStepTitle">3. Training budget</p>
          <strong>{trainingRuntimeReady ? "Ready" : "Needs setup"}</strong>
          <p className="fieldNote">
            {trainingRuntimeError ??
              `Budget: ${budgetLimit} ${budgetUnit}, thresholds: ${activeThresholds}.`}
          </p>
          <button
            type="button"
            className="workflowStepLink workflowStepAction"
            onClick={() => onNavigateSettings("training")}
          >
            Open budget
          </button>
        </article>

        <article
          className={`workflowStepTile ${
            hasValidationPassed
              ? "workflowStepTile-ready"
              : isValidating
                ? "workflowStepTile-inProgress"
                : "workflowStepTile-waiting"
          }`}
          role="listitem"
        >
          <p className="workflowStepTitle">4. Validate</p>
          <strong>
            {hasValidationPassed
              ? "Ready"
              : isValidating
                ? "In progress"
                : "Needs setup"}
          </strong>
          <p className="fieldNote">
            {hasValidationPassed
              ? "Configs are valid."
              : isValidating
                ? "Checking latest changes..."
                : preflightReady
                  ? validationError ?? "Waiting to validate."
                  : "Complete steps 1-3 first."}
          </p>
          <button
            type="button"
            className="secondaryButton workflowStepAction workflowStepButtonCompact"
            onClick={onValidate}
            disabled={controlsDisabled || !preflightReady || isValidating}
          >
            {isValidating ? "Validating..." : "Validate now"}
          </button>
        </article>

        <article
          className={`workflowStepTile ${
            trainingCompleted
              ? "workflowStepTile-ready"
              : hasTrainingInProgress
                ? "workflowStepTile-inProgress"
                : "workflowStepTile-waiting"
          }`}
          role="listitem"
        >
          <p className="workflowStepTitle">5. Train</p>
          <strong>
            {trainingCompleted
              ? "Trained"
              : hasTrainingInProgress
                ? "In progress"
                : "Not ready"}
          </strong>
          <p className="fieldNote">
            {trainingCompleted
              ? "Tokenizer is ready."
              : hasTrainingInProgress
                ? `Current job is ${describeJobState(activeJobState ?? "running").toLowerCase()}.`
                : hasValidationPassed
                  ? "Ready to train."
                  : isValidating
                    ? "Waiting for validation."
                    : "Validate before training."}
          </p>
          <button
            type="button"
            className="primaryButton workflowStepAction"
            onClick={onTrain}
            disabled={!canStartTraining}
          >
            {hasTrainingInProgress
              ? "Training..."
              : isSubmitting
                ? "Starting..."
                : "Start training"}
          </button>
        </article>
      </div>
    </section>
  );
}
