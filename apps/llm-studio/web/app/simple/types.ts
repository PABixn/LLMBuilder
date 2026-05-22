import type { ProjectDetail, ModelAnalysisResponse } from "../../lib/api";
import type { TrainingJob as TokenizerJob, TokenizerPreviewResult } from "../../lib/tokenizerLegacyApi";
import type {
  GenerateTrainingCompletionResponse,
  TrainingCheckpointEntry,
  TrainingJob,
  TrainingLogsResponse,
  TrainingMetricPoint,
  TrainingPreflightResponse,
  TrainingSampleEntry,
} from "../../lib/training/types";
import type { SimpleStepId } from "../shared/lib/navigation";

export type { SimpleStepId };

export type SimpleStepState = "ready" | "blocked" | "running" | "failed" | "completed";
export type SimpleDatasetSource = "starter" | "upload" | "streaming";
export type SimpleTrainingProfile = "quick" | "balanced" | "longer";
export type SimpleExecutionKind = "local" | "runpod_pod";
export type SimpleInferenceLength = "short" | "medium" | "long";
export type SimpleInferenceCreativity = "precise" | "balanced" | "creative";
export type SimpleStreamingDatasetId = "fineweb-edu" | "tinystories" | "the-stack";

export interface SimpleLocalTrainFile {
  id: string;
  fileName: string;
  filePath: string;
  sizeBytes: number | null;
  sizeChars: number | null;
}

export interface SimpleFlowState {
  version: number;
  presetId: string;
  modelName: string;
  targetVocabSize: number;
  targetContextLength: number;
  projectId: string | null;
  tokenizerJobId: string | null;
  trainingJobId: string | null;
  datasetSource: SimpleDatasetSource;
  localTrainFiles: SimpleLocalTrainFile[];
  streamingPrimaryDatasetId: SimpleStreamingDatasetId;
  streamingAdditionalDatasetIds: SimpleStreamingDatasetId[];
  trainingProfile: SimpleTrainingProfile;
  executionKind: SimpleExecutionKind;
  checkpointValue: string;
  lastCompletedStep: SimpleStepId | null;
}

export interface SimpleStepViewModel {
  id: SimpleStepId;
  index: number;
  title: string;
  state: SimpleStepState;
  status: string;
  blocker: string | null;
  actionLabel: string;
  artifactLabel: string | null;
}

export interface SimpleModelStepState {
  project: ProjectDetail | null;
  projectError: string | null;
  analyzing: boolean;
  creating: boolean;
  analysisByPresetId: Record<string, ModelAnalysisResponse | null>;
  analysisErrorsByPresetId: Record<string, string>;
  selectedAnalysis: ModelAnalysisResponse | null;
  createArchitecture: () => Promise<void>;
  refreshProject: () => Promise<void>;
  syncProjectVocab: (vocabSize: number) => Promise<ProjectDetail | null>;
}

export interface SimpleTokenizerStepState {
  tokenizerJob: TokenizerJob | null;
  tokenizerError: string | null;
  validationError: string | null;
  validationMessage: string | null;
  validating: boolean;
  uploading: boolean;
  training: boolean;
  previewing: boolean;
  previewText: string;
  previewError: string | null;
  previewResult: TokenizerPreviewResult | null;
  tokenizerConfig: Record<string, unknown>;
  dataloaderConfig: Record<string, unknown>;
  datasetReady: boolean;
  datasetBlocker: string | null;
  setPreviewText: (value: string) => void;
  uploadFiles: (files: File[]) => Promise<void>;
  validateTokenizer: () => Promise<boolean>;
  startTokenizerTraining: () => Promise<void>;
  removeLocalFile: (fileId: string) => void;
  clearLocalFiles: () => void;
}

export interface SimpleTrainingStepState {
  trainingRun: TrainingJob | null;
  recentRuns: TrainingJob[];
  checkpoints: TrainingCheckpointEntry[];
  metrics: TrainingMetricPoint[];
  samples: TrainingSampleEntry[];
  logs: TrainingLogsResponse | null;
  trainingConfig: Record<string, unknown> | null;
  dataloaderConfig: Record<string, unknown> | null;
  preflight: TrainingPreflightResponse | null;
  preflightError: string | null;
  preflightLoading: boolean;
  launching: boolean;
  appliedFixes: string[];
  profileNote: string;
  cloudConfirmed: boolean;
  setCloudConfirmed: (confirmed: boolean) => void;
  runPreflight: () => void;
  startTraining: () => Promise<void>;
}

export interface SimpleInferenceStepState {
  selectedRun: TrainingJob | null;
  completedRuns: TrainingJob[];
  checkpoints: TrainingCheckpointEntry[];
  checkpointError: string | null;
  checkpointsLoading: boolean;
  latestCheckpoint: TrainingCheckpointEntry | null;
  lengthPreset: SimpleInferenceLength;
  creativityPreset: SimpleInferenceCreativity;
  prompt: string;
  generating: boolean;
  generationError: string | null;
  result: GenerateTrainingCompletionResponse | null;
  setLengthPreset: (preset: SimpleInferenceLength) => void;
  setCreativityPreset: (preset: SimpleInferenceCreativity) => void;
  setPrompt: (prompt: string) => void;
  generate: () => Promise<void>;
  tryAnother: () => Promise<void>;
}

export interface SimpleModeController {
  flow: SimpleFlowState;
  updateFlow: (updater: (current: SimpleFlowState) => SimpleFlowState) => void;
  resetFlow: () => void;
  activeStep: SimpleStepId;
  setActiveStep: (step: SimpleStepId) => void;
  steps: SimpleStepViewModel[];
  modelStep: SimpleModelStepState;
  tokenizerStep: SimpleTokenizerStepState;
  trainingStep: SimpleTrainingStepState;
  inferenceStep: SimpleInferenceStepState;
}
