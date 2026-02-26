import type { SuggestionCategory, SuggestionPriority } from "../types";

import { addBackendAnalysisMetricSuggestions } from "./suggestions/analysisRules";
import { addArchitectureSuggestions } from "./suggestions/architectureRules";
import {
  finalizeSuggestions,
  labelForSuggestionCategory,
  labelForSuggestionPriority,
} from "./suggestions/helpers";
import type { BuildDesignSuggestionsArgs } from "./suggestions/types";
import { addValidationAndWorkflowSuggestions } from "./suggestions/validationRules";

export function suggestionPriorityLabel(priority: SuggestionPriority): string {
  return labelForSuggestionPriority(priority);
}

export function suggestionCategoryLabel(category: SuggestionCategory): string {
  return labelForSuggestionCategory(category);
}

export function buildDesignSuggestions({
  modelConfig,
  metrics,
  diagnostics,
  backendValidation,
  backendAnalysis,
  backendAnalysisStale,
  consecutiveBlockGroups,
}: BuildDesignSuggestionsArgs) {
  const suggestions = new Map();

  addValidationAndWorkflowSuggestions({
    suggestions,
    diagnostics,
    backendValidation,
    backendAnalysis,
    backendAnalysisStale,
  });

  addArchitectureSuggestions({
    suggestions,
    modelConfig,
    metrics,
    consecutiveBlockGroups,
  });

  addBackendAnalysisMetricSuggestions({
    suggestions,
    backendAnalysis,
  });

  return finalizeSuggestions(suggestions);
}
