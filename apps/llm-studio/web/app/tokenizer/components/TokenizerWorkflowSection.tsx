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
          <p className="panelEyebrow">Top Workflow</p>
          <h2>Steps to train the tokenizer</h2>
          <p className="panelCopy">
            Complete each step in order. A step turns green only when it is ready.
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
          <p className="workflowStepTitle">Step 1 - Choose tokenizer config</p>
          <strong>{tokenizerReady ? "Ready" : "Waiting for configuration"}</strong>
          <p className="fieldNote">
            {tokenizerError ??
              `${tokenizerType.toUpperCase()} tokenizer configured.`}
          </p>
          <button
            type="button"
            className="workflowStepLink workflowStepAction"
            onClick={() => onNavigateSettings("tokenizer")}
          >
            Open tokenizer settings
          </button>
        </article>

        <article
          className={`workflowStepTile ${
            datasetReady ? "workflowStepTile-ready" : "workflowStepTile-waiting"
          }`}
          role="listitem"
        >
          <p className="workflowStepTitle">Step 2 - Choose dataset</p>
          <strong>{datasetReady ? "Ready" : "Waiting for configuration"}</strong>
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
            Open dataset settings
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
          <p className="workflowStepTitle">Step 3 - Configure training run</p>
          <strong>{trainingRuntimeReady ? "Ready" : "Waiting for configuration"}</strong>
          <p className="fieldNote">
            {trainingRuntimeError ??
              `Budget: ${budgetLimit} ${budgetUnit}, thresholds: ${activeThresholds}.`}
          </p>
          <button
            type="button"
            className="workflowStepLink workflowStepAction"
            onClick={() => onNavigateSettings("training")}
          >
            Open training settings
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
          <p className="workflowStepTitle">Step 4 - Validate configs</p>
          <strong>
            {hasValidationPassed
              ? "Ready"
              : isValidating
                ? "In progress"
                : "Waiting for configuration"}
          </strong>
          <p className="fieldNote">
            {hasValidationPassed
              ? "Validation passed for tokenizer and dataloader configs."
              : isValidating
                ? "Validating latest configuration changes automatically..."
                : preflightReady
                  ? validationError ?? "Waiting for the next validation cycle."
                  : "Complete steps 1-3 first. Validation runs automatically."}
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
          <p className="workflowStepTitle">Step 5 - Start training</p>
          <strong>
            {trainingCompleted
              ? "Ready (trained)"
              : hasTrainingInProgress
                ? "In progress"
                : "Not ready"}
          </strong>
          <p className="fieldNote">
            {trainingCompleted
              ? "Latest training job completed. Artifact is ready."
              : hasTrainingInProgress
                ? `Current job is ${describeJobState(activeJobState ?? "running").toLowerCase()}.`
                : hasValidationPassed
                  ? "Validation passed. Start training to complete this step."
                  : isValidating
                    ? "Waiting for automatic validation to finish."
                    : "Automatic validation must pass to unlock training."}
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
                : "Start Training"}
          </button>
        </article>
      </div>
    </section>
  );
}
