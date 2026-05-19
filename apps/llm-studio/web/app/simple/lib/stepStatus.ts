import type { ProjectDetail } from "../../../lib/api";
import type { TrainingJob as TokenizerJob } from "../../../lib/tokenizerLegacyApi";
import type { TrainingCheckpointEntry, TrainingJob } from "../../../lib/training/types";
import type { SimpleFlowState, SimpleStepViewModel } from "../types";
import { readTokenizerVocabSize } from "./vocabularySync";

export function isTokenizerRunning(job: TokenizerJob | null): boolean {
  return job?.status === "pending" || job?.status === "running";
}

export function isTrainingRunning(job: TrainingJob | null): boolean {
  return job?.status === "pending" || job?.status === "running";
}

export function isTokenizerCompleted(job: TokenizerJob | null): boolean {
  return job?.status === "completed" && readTokenizerVocabSize(job) !== null;
}

export function isTrainingCompletedWithCheckpoint(
  job: TrainingJob | null,
  checkpoints: TrainingCheckpointEntry[]
): boolean {
  return Boolean(job?.status === "completed" && (job.checkpoint_count > 0 || checkpoints.length > 0));
}

export function deriveSimpleStepStatuses(options: {
  flow: SimpleFlowState;
  project: ProjectDetail | null;
  projectLoading: boolean;
  projectError: string | null;
  tokenizerJob: TokenizerJob | null;
  tokenizerError: string | null;
  datasetReady: boolean;
  datasetBlocker: string | null;
  tokenizerValidationError: string | null;
  trainingRun: TrainingJob | null;
  trainingCheckpoints: TrainingCheckpointEntry[];
  preflightValid: boolean;
  preflightError: string | null;
  trainingLaunching: boolean;
  inferenceGenerating: boolean;
  generationSucceeded: boolean;
  checkpointError: string | null;
}): SimpleStepViewModel[] {
  const {
    flow,
    project,
    projectLoading,
    projectError,
    tokenizerJob,
    tokenizerError,
    datasetReady,
    datasetBlocker,
    tokenizerValidationError,
    trainingRun,
    trainingCheckpoints,
    preflightValid,
    preflightError,
    trainingLaunching,
    inferenceGenerating,
    generationSucceeded,
    checkpointError,
  } = options;

  const architectureComplete = Boolean(project?.valid);
  const tokenizerRunning = isTokenizerRunning(tokenizerJob);
  const tokenizerComplete = isTokenizerCompleted(tokenizerJob);
  const trainingRunning = isTrainingRunning(trainingRun) || trainingLaunching;
  const trainingComplete = isTrainingCompletedWithCheckpoint(trainingRun, trainingCheckpoints);

  return [
    {
      id: "architecture",
      index: 1,
      title: "Architecture",
      state: projectLoading
        ? "running"
        : projectError
          ? "failed"
          : architectureComplete
            ? "completed"
            : flow.modelName.trim() && flow.presetId
              ? "ready"
              : "blocked",
      status: architectureComplete ? "Saved" : projectLoading ? "Working" : projectError ? "Failed" : "Ready",
      blocker: projectError,
      actionLabel: architectureComplete ? "Edit architecture" : "Create architecture",
      artifactLabel: project ? project.name ?? project.artifact_file : null,
    },
    {
      id: "tokenizer",
      index: 2,
      title: "Tokenizer",
      state: tokenizerError
        ? "failed"
        : tokenizerRunning
          ? "running"
          : tokenizerComplete
            ? "completed"
            : !architectureComplete
              ? "blocked"
              : datasetReady && !tokenizerValidationError
                ? "ready"
                : "blocked",
      status: tokenizerComplete
        ? "Trained"
        : tokenizerRunning
          ? "Training"
          : !architectureComplete
            ? "Blocked"
            : "Ready",
      blocker:
        tokenizerError ??
        tokenizerValidationError ??
        (!architectureComplete ? "Create an architecture first." : datasetBlocker),
      actionLabel: tokenizerComplete ? "Review tokenizer" : "Train tokenizer",
      artifactLabel: tokenizerJob ? String(tokenizerJob.tokenizer_config.name ?? tokenizerJob.id) : null,
    },
    {
      id: "training",
      index: 3,
      title: "Training",
      state: trainingRun?.status === "failed"
        ? "failed"
        : trainingRunning
          ? "running"
          : trainingComplete
            ? "completed"
            : !tokenizerComplete || !architectureComplete
              ? "blocked"
              : preflightValid
                ? "ready"
                : "blocked",
      status: trainingComplete
        ? "Trained"
        : trainingRunning
          ? "Running"
          : preflightValid
            ? "Ready"
            : "Blocked",
      blocker:
        trainingRun?.error ??
        (!architectureComplete
          ? "Create an architecture first."
          : !tokenizerComplete
            ? "Train a tokenizer first."
            : preflightError ?? "Run preflight before training."),
      actionLabel: trainingComplete ? "Review run" : "Start training",
      artifactLabel: trainingRun ? trainingRun.name : null,
    },
    {
      id: "inference",
      index: 4,
      title: "Inference",
      state: inferenceGenerating
        ? "running"
        : generationSucceeded
          ? "completed"
          : trainingComplete && !checkpointError
            ? "ready"
            : "blocked",
      status: generationSucceeded
        ? "Generated"
        : inferenceGenerating
          ? "Generating"
          : trainingComplete
            ? "Ready"
            : "Blocked",
      blocker: checkpointError ?? (!trainingComplete ? "Finish training with a checkpoint first." : null),
      actionLabel: generationSucceeded ? "Generate again" : "Generate",
      artifactLabel: trainingComplete && trainingRun ? trainingRun.name : null,
    },
  ];
}
