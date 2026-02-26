import type { ValidationIssue } from "../../../../lib/api";
import type { ModelConfig } from "../../../../lib/defaults";

import type {
  BackendAnalysisState,
  BackendValidationState,
  BuilderMetrics,
  ConsecutiveBlockGroup,
  DesignSuggestion,
  Diagnostic,
} from "../../types";

export type BuildDesignSuggestionsArgs = {
  modelConfig: ModelConfig;
  metrics: BuilderMetrics;
  diagnostics: Diagnostic[];
  backendValidation: BackendValidationState;
  backendAnalysis: BackendAnalysisState;
  backendAnalysisStale: boolean;
  consecutiveBlockGroups: ConsecutiveBlockGroup[];
};

export type InternalSuggestion = Omit<DesignSuggestion, "id"> & {
  key: string;
};

export type SuggestionBucket = Map<string, InternalSuggestion>;

export type DiagnosticGroup = {
  level: Diagnostic["level"];
  source: Diagnostic["source"];
  message: string;
  count: number;
  samplePath: string;
};

export type BackendIssueGroup = {
  code: string;
  count: number;
  sample: ValidationIssue;
};
