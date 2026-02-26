import type { Dispatch, SetStateAction } from "react";
import {
  FiAlertTriangle,
  FiCheckCircle,
  FiXCircle,
} from "react-icons/fi";

import type { ModelConfig } from "../../../../lib/defaults";

import { useDesignSuggestionActions } from "../../hooks/useDesignSuggestionActions";
import type {
  BackendAnalysisState,
  BackendValidationState,
  BuilderMetrics,
  ConsecutiveBlockGroup,
  DesignSuggestion,
  Diagnostic,
  StudioDocument,
} from "../../types";
import { formatTimeAgo } from "../../utils/format";
import {
  buildDesignSuggestions,
  suggestionPriorityLabel,
} from "../../utils/suggestions";

type DiagnosticsPanelProps = {
  diagnostics: Diagnostic[];
  localErrors: Diagnostic[];
  localWarnings: Diagnostic[];
  backendValidation: BackendValidationState;
  totalErrors: number;
  totalWarnings: number;
  modelConfig: ModelConfig;
  metrics: BuilderMetrics;
  backendAnalysis: BackendAnalysisState;
  backendAnalysisStale: boolean;
  consecutiveBlockGroups: ConsecutiveBlockGroup[];
  setDocumentState: Dispatch<SetStateAction<StudioDocument>>;
  runBackendAnalysis: () => Promise<void>;
};

function suggestionTone(
  priority: DesignSuggestion["priority"]
): "error" | "warning" | "info" | "good" {
  if (priority === "critical") {
    return "error";
  }
  if (priority === "high" || priority === "medium") {
    return "warning";
  }
  return "info";
}

function suggestionBadgeTone(
  priority: DesignSuggestion["priority"]
): "error" | "warn" | "neutral" {
  if (priority === "critical") {
    return "error";
  }
  if (priority === "high" || priority === "medium") {
    return "warn";
  }
  return "neutral";
}

export function DiagnosticsPanel({
  diagnostics,
  localErrors,
  localWarnings,
  backendValidation,
  totalErrors,
  totalWarnings,
  modelConfig,
  metrics,
  backendAnalysis,
  backendAnalysisStale,
  consecutiveBlockGroups,
  setDocumentState,
  runBackendAnalysis,
}: DiagnosticsPanelProps) {
  const diagnosticLevelRank: Record<Diagnostic["level"], number> = {
    error: 0,
    warning: 1,
    info: 2,
  };
  const diagnosticSourceRank: Record<Diagnostic["source"], number> = {
    local: 0,
    backend: 1,
  };

  const sortedDiagnostics = [...diagnostics].sort((a, b) => {
    const levelDelta = diagnosticLevelRank[a.level] - diagnosticLevelRank[b.level];
    if (levelDelta !== 0) {
      return levelDelta;
    }
    const sourceDelta = diagnosticSourceRank[a.source] - diagnosticSourceRank[b.source];
    if (sourceDelta !== 0) {
      return sourceDelta;
    }
    return a.path.localeCompare(b.path) || a.message.localeCompare(b.message);
  });

  const unifiedValidationTone =
    totalErrors > 0
      ? "error"
      : totalWarnings > 0 || backendValidation.phase === "fallback"
        ? "warn"
        : backendValidation.phase === "validating"
          ? "neutral"
          : "good";
  const unifiedValidationStateLabel =
    totalErrors > 0
      ? "Blocked"
      : totalWarnings > 0
        ? "Needs Review"
        : backendValidation.phase === "validating"
          ? "Validating"
          : backendValidation.phase === "fallback"
            ? "Local Only"
            : "Healthy";
  const backendValidationPhaseTone =
    backendValidation.phase === "fallback"
      ? "warn"
      : backendValidation.phase === "success"
        ? backendValidation.errors.length > 0
          ? "error"
          : backendValidation.warnings.length > 0
            ? "warn"
            : "good"
        : backendValidation.phase === "validating"
          ? "warn"
          : "neutral";
  const backendValidationPhaseLabel =
    backendValidation.phase === "success"
      ? backendValidation.errors.length > 0
        ? "Issues Found"
        : backendValidation.warnings.length > 0
          ? "Warnings"
          : "Passed"
      : backendValidation.phase === "fallback"
        ? "Unavailable"
        : backendValidation.phase === "validating"
          ? "Running"
          : backendValidation.phase === "skipped"
            ? "Blocked by Local Errors"
            : "Idle";

  const designSuggestions = buildDesignSuggestions({
    modelConfig,
    metrics,
    diagnostics,
    backendValidation,
    backendAnalysis,
    backendAnalysisStale,
    consecutiveBlockGroups,
  });
  const lowPrioritySuggestionCount = designSuggestions.filter(
    (suggestion) => suggestion.priority === "low"
  ).length;
  const suggestionPreviewLimit = 3;
  const visibleSuggestions = designSuggestions.slice(0, suggestionPreviewLimit);
  const hiddenSuggestions = designSuggestions.slice(suggestionPreviewLimit);

  const {
    canApplySuggestion,
    suggestionApplyDisabled,
    suggestionApplyTitle,
    applySuggestion,
  } = useDesignSuggestionActions({
    backendAnalysisPhase: backendAnalysis.phase,
    localErrorCount: localErrors.length,
    setDocumentState,
    runBackendAnalysis,
  });

  function renderDiagnosticItem(diagnostic: Diagnostic) {
    return (
      <div
        key={diagnostic.id}
        className={`diagnosticItem tone-${diagnostic.level}`}
        role="listitem"
      >
        <div className="diagnosticIcon">
          {diagnostic.level === "error" ? (
            <FiXCircle />
          ) : diagnostic.level === "warning" ? (
            <FiAlertTriangle />
          ) : (
            <FiCheckCircle />
          )}
        </div>
        <div>
          <div className="diagnosticTitle">{diagnostic.message}</div>
          <div className="diagnosticMeta">
            <code>{diagnostic.path}</code> · {diagnostic.source}
          </div>
        </div>
      </div>
    );
  }

  function renderDesignSuggestionItem(suggestion: DesignSuggestion) {
    const tone = suggestionTone(suggestion.priority);
    return (
      <article
        key={suggestion.id}
        className={`designSuggestionItem designSuggestionItemCompact tone-${tone}`}
        role="listitem"
      >
        <div className="designSuggestionCompactRow">
          <div className="designSuggestionCompactMain">
            <div className="diagnosticIcon">
              {suggestion.priority === "critical" ? (
                <FiXCircle />
              ) : suggestion.priority === "high" || suggestion.priority === "medium" ? (
                <FiAlertTriangle />
              ) : (
                <FiCheckCircle />
              )}
            </div>
            <div>
              <div className="designSuggestionTitle">{suggestion.title}</div>
              <div className="designSuggestionCompactAction">
                <strong>Tip:</strong> {suggestion.action}
              </div>
            </div>
          </div>
          <span className={`pillBadge tone-${suggestionBadgeTone(suggestion.priority)}`}>
            {suggestionPriorityLabel(suggestion.priority)}
          </span>
        </div>
        {canApplySuggestion(suggestion) ? (
          <div className="designSuggestionFooter">
            {suggestion.applyOptions.map((option) => (
              <button
                key={option.id}
                type="button"
                className="suggestionApplyButton"
                onClick={() => applySuggestion(option)}
                disabled={suggestionApplyDisabled(option)}
                title={suggestionApplyTitle(option)}
              >
                {option.label}
              </button>
            ))}
          </div>
        ) : null}
      </article>
    );
  }

  return (
    <section id="diagnostics" className="panelCard diagnosticsPanel">
      <div className="panelHead">
        <div>
          <p className="panelEyebrow">Diagnostics</p>
          <h2>Validation & Design Guidance</h2>
        </div>
      </div>

      <div
        className={`diagnosticOverviewGrid${diagnostics.length === 0 && designSuggestions.length === 0 ? " isOnlyContent" : ""}`}
        aria-label="Validation summary"
      >
        <section
          className={`diagnosticOverviewCard diagnosticUnifiedCard tone-${unifiedValidationTone}`}
          aria-label="Unified validation summary"
        >
          <div className="diagnosticOverviewTopRow">
            <div className="diagnosticOverviewLabel">Unified Validation</div>
            <div className="diagnosticOverviewState">{unifiedValidationStateLabel}</div>
          </div>
          <div className="diagnosticOverviewCounts" aria-label="Validation counts">
            <div
              className={`diagnosticOverviewCount tone-error${totalErrors === 0 ? " is-zero" : ""}`}
            >
              <span>Errors</span>
              <strong>{totalErrors}</strong>
            </div>
            <div
              className={`diagnosticOverviewCount tone-warn${totalWarnings === 0 ? " is-zero" : ""}`}
            >
              <span>Warnings</span>
              <strong>{totalWarnings}</strong>
            </div>
          </div>

          <div className="diagnosticCompactMetaRow" aria-label="Validation engine status">
            <span
              className={`pillBadge tone-${localErrors.length > 0 ? "error" : localWarnings.length > 0 ? "warn" : "good"}`}
            >
              Local:{" "}
              {localErrors.length > 0
                ? "Errors"
                : localWarnings.length > 0
                  ? "Warnings"
                  : "Passed"}
            </span>
            <span className={`pillBadge tone-${backendValidationPhaseTone}`}>
              Backend: {backendValidationPhaseLabel}
            </span>
            <span className="pillBadge tone-neutral">
              Checked: {formatTimeAgo(backendValidation.lastValidatedAt)}
            </span>
            {backendValidation.normalizedChanged ? (
              <span className="pillBadge tone-warn">Normalized</span>
            ) : null}
          </div>

          {backendValidation.phase !== "success" || backendValidation.normalizedChanged ? (
            <p className="diagnosticUnifiedMessage">{backendValidation.message}</p>
          ) : null}
        </section>
      </div>

      <div className="diagnosticSectionHead">
        <div>
          <h3 className="diagnosticSectionTitle">Top suggestions</h3>
        </div>
        {designSuggestions.length > visibleSuggestions.length ? (
          <div className="heroMetaPills">
            <span className="pillBadge tone-neutral">{designSuggestions.length} total</span>
            {lowPrioritySuggestionCount > 0 ? (
              <span className="pillBadge tone-neutral">
                {lowPrioritySuggestionCount} low-priority
              </span>
            ) : null}
          </div>
        ) : null}
      </div>

      {visibleSuggestions.length > 0 ? (
        <>
          <div className="designSuggestionList" role="list">
            {visibleSuggestions.map((suggestion) => renderDesignSuggestionItem(suggestion))}
          </div>

          {hiddenSuggestions.length > 0 ? (
            <details className="sectionDisclosure compact">
              <summary className="sectionDisclosureSummary">
                Show {hiddenSuggestions.length} more suggestion
                {hiddenSuggestions.length === 1 ? "" : "s"}
                {lowPrioritySuggestionCount > 0
                  ? ` (including ${lowPrioritySuggestionCount} low-priority)`
                  : ""}
              </summary>
              <div className="designSuggestionList designSuggestionListNested" role="list">
                {hiddenSuggestions.map((suggestion) => renderDesignSuggestionItem(suggestion))}
              </div>
            </details>
          ) : null}
        </>
      ) : (
        <div className="designSuggestionEmpty">
          Tips will appear here as you edit the model and run backend analysis.
        </div>
      )}

      {diagnostics.length > 0 ? (
        <details className="sectionDisclosure compact">
          <summary className="sectionDisclosureSummary">
            Show diagnostics ({diagnostics.length})
          </summary>
          <div className="diagnosticList diagnosticListNested" role="list">
            {sortedDiagnostics.map((diagnostic) => renderDiagnosticItem(diagnostic))}
          </div>
        </details>
      ) : null}
    </section>
  );
}
