import type { BackendAnalysisState, BackendValidationState, Diagnostic } from "../../types";
import type { SuggestionBucket } from "./types";
import {
  addSuggestion,
  backendAnalysisApplyOptions,
  backendIssueGroups,
  buildBackendIssueSuggestion,
  groupedDiagnostics,
  pluralize,
} from "./helpers";

type AddValidationAndWorkflowSuggestionsArgs = {
  suggestions: SuggestionBucket;
  diagnostics: Diagnostic[];
  backendValidation: BackendValidationState;
  backendAnalysis: BackendAnalysisState;
  backendAnalysisStale: boolean;
};

export function addValidationAndWorkflowSuggestions({
  suggestions,
  diagnostics,
  backendValidation,
  backendAnalysis,
  backendAnalysisStale,
}: AddValidationAndWorkflowSuggestionsArgs): void {
  const localDiagnostics = diagnostics.filter((item) => item.source === "local");
  const localErrors = localDiagnostics.filter((item) => item.level === "error");
  const localWarnings = localDiagnostics.filter((item) => item.level === "warning");

  if (localErrors.length > 0) {
    addSuggestion(suggestions, {
      key: "workflow:fix-local-errors-first",
      priority: "critical",
      category: "workflow",
      source: "local",
      title: "Fix local validation errors before trusting backend results",
      summary: `${pluralize(localErrors.length, "local error")} currently block backend validation and analysis from giving reliable feedback.`,
      action: "Resolve the red diagnostics first, then re-check backend validation and re-run runtime analysis.",
      path: localErrors[0]?.path ?? null,
      score: 111,
    });
  }

  for (const group of groupedDiagnostics(localDiagnostics).slice(0, 8)) {
    const message = group.message;
    if (group.level === "info") {
      continue;
    }

    if (message.includes("Block has no attention component")) {
      addSuggestion(suggestions, {
        key: "local-pattern:missing-attention",
        priority: "high",
        category: "architecture",
        source: "local",
        title: "Add attention to transformer blocks unless this is intentional",
        summary: `${pluralize(group.count, "block")} currently has no attention component. This is valid in the editor but unusual for a decoder-style transformer.`,
        action: "Add an attention component to each affected block or document the block as a specialized non-attention stage.",
        path: group.samplePath,
        score: 87 + Math.min(group.count, 5),
      });
      continue;
    }

    if (message.includes("Block has no MLP component")) {
      addSuggestion(suggestions, {
        key: "local-pattern:missing-mlp",
        priority: "high",
        category: "architecture",
        source: "local",
        title: "Add an MLP path to preserve transformer block capacity",
        summary: `${pluralize(group.count, "block")} is missing an MLP component, which often reduces representational capacity even if attention is present.`,
        action: "Add an MLP component after attention (often with a norm in between) unless you are designing an attention-only experiment.",
        path: group.samplePath,
        score: 86 + Math.min(group.count, 5),
      });
      continue;
    }

    if (message.includes("n_embd must be divisible by n_head")) {
      addSuggestion(suggestions, {
        key: "local-pattern:n-embd-divisible",
        priority: "critical",
        category: "correctness",
        source: "local",
        title: "Fix attention head divisibility",
        summary:
          "The editor detected at least one attention component where n_embd is not divisible by n_head, making head_dim invalid.",
        action:
          "Choose n_head values that divide n_embd cleanly across all attention components, or change n_embd to a compatible value.",
        path: group.samplePath,
        score: 109,
      });
      continue;
    }

    if (message.includes("n_head must be divisible by n_kv_head")) {
      addSuggestion(suggestions, {
        key: "local-pattern:gqa-divisible",
        priority: "critical",
        category: "correctness",
        source: "local",
        title: "Fix grouped-query attention ratio",
        summary:
          "At least one attention component has an invalid n_head:n_kv_head ratio and cannot be grouped evenly.",
        action: "Set n_kv_head to a divisor of n_head (for example 1, 2, 4, 8 depending on n_head).",
        path: group.samplePath,
        score: 108,
      });
      continue;
    }

    if (message.includes("n_kv_head cannot exceed n_head")) {
      addSuggestion(suggestions, {
        key: "local-pattern:n-kv-head-upper-bound",
        priority: "critical",
        category: "correctness",
        source: "local",
        title: "Reduce n_kv_head",
        summary:
          "Some attention components use more KV heads than query heads, which is invalid for standard attention and GQA.",
        action: "Set n_kv_head <= n_head for each affected attention component.",
        path: group.samplePath,
        score: 108,
      });
      continue;
    }

    if (message.includes("Rotary embeddings require an even head_dim")) {
      addSuggestion(suggestions, {
        key: "local-pattern:even-head-dim",
        priority: "high",
        category: "correctness",
        source: "local",
        title: "Use an even head dimension",
        summary:
          "The current n_embd / n_head choice produces an odd head_dim, which the local validator blocks for rotary embedding compatibility.",
        action: "Adjust n_embd or n_head so head_dim is even in every attention component.",
        path: group.samplePath,
        score: 88,
      });
      continue;
    }

    if (message.includes("MLP sequence has no activation step")) {
      addSuggestion(suggestions, {
        key: "local-pattern:mlp-no-activation",
        priority: "medium",
        category: "architecture",
        source: "local",
        title: "Add an activation step inside the MLP",
        summary:
          "An MLP without activations behaves close to a linear projection stack and usually underuses the MLP branch.",
        action: "Insert an activation (GELU or SiLU are common) between linear steps in each affected MLP.",
        path: group.samplePath,
        score: 69,
      });
      continue;
    }

    if (message.includes("MLP sequence has no linear steps")) {
      addSuggestion(suggestions, {
        key: "local-pattern:mlp-no-linear",
        priority: "medium",
        category: "architecture",
        source: "local",
        title: "Add linear projections to the MLP sequence",
        summary:
          "An MLP with no linear steps becomes a nonlinear pass-through and may not change hidden dimensionality or capacity as intended.",
        action: "Use the typical linear -> activation -> linear pattern unless you are intentionally building a custom nonlinear residual block.",
        path: group.samplePath,
        score: 68,
      });
      continue;
    }

    if (
      message.includes("When multiplier != 1, the first MLP step should be linear") ||
      message.includes("When multiplier != 1, the last MLP step should be linear") ||
      message.includes(
        "A single linear step with multiplier != 1 will leave the MLP output dimension mismatched"
      )
    ) {
      addSuggestion(suggestions, {
        key: "local-pattern:mlp-multiplier-shape",
        priority: "critical",
        category: "correctness",
        source: "local",
        title: "Preserve MLP input/output shape when multiplier != 1",
        summary:
          "The MLP sequence currently expands or contracts hidden size without projecting back correctly, which causes shape mismatches.",
        action:
          "Use at least two linear steps and keep the first/last steps linear when the MLP multiplier is not 1.",
        path: group.samplePath,
        score: 104,
      });
      continue;
    }

    addSuggestion(suggestions, {
      key: `local-generic:${group.level}:${group.message}`,
      priority: group.level === "error" ? "high" : "medium",
      category: "workflow",
      source: "local",
      title: group.level === "error" ? "Resolve local validation blocker" : "Review local validation warning",
      summary:
        group.count > 1
          ? `${pluralize(group.count, group.level)} share the same message: ${group.message}`
          : group.message,
      action:
        group.level === "error"
          ? "Fix the affected fields/components in the builder, then let backend validation re-run."
          : "Confirm the behavior is intentional or standardize the affected blocks/components for cleaner experiments.",
      path: group.samplePath,
      score: (group.level === "error" ? 82 : 63) + Math.min(group.count, 5),
    });
  }

  const backendAndAnalysisIssues = [
    ...backendValidation.errors,
    ...backendValidation.warnings,
    ...backendAnalysis.errors,
    ...backendAnalysis.warnings,
  ];
  for (const grouped of backendIssueGroups(backendAndAnalysisIssues)) {
    const suggestion = buildBackendIssueSuggestion(grouped.sample);
    addSuggestion(suggestions, {
      ...suggestion,
      summary:
        grouped.count > 1
          ? `${suggestion.summary} (${pluralize(grouped.count, "occurrence")} reported across backend checks.)`
          : suggestion.summary,
    });
  }

  if (backendValidation.phase === "fallback") {
    addSuggestion(suggestions, {
      key: "backend:fallback",
      priority: "high",
      category: "workflow",
      source: "backend",
      title: "Restore backend validation coverage",
      summary:
        "The app is currently using local checks only. Backend validation catches semantic issues and normalization differences that local checks may miss.",
      action: "Start the API backend and confirm `/validate/model` is reachable, then edit the config to trigger a fresh validation pass.",
      path: "/validate/model",
      applyOptions: [],
      score: 84,
    });
  }

  if (backendValidation.phase === "success" && backendValidation.normalizedChanged) {
    addSuggestion(suggestions, {
      key: "backend:normalized-changed",
      priority: "medium",
      category: "workflow",
      source: "backend",
      title: "Review backend normalization changes",
      summary:
        "The backend normalized your config into a different JSON shape/value set. This is often safe, but it can hide assumptions during iteration.",
      action: "Compare the JSON preview with the backend-normalized result before exporting or saving a baseline.",
      path: "/validate/model",
      applyOptions: [],
      score: 64,
    });
  }

  if (backendAnalysisStale && backendAnalysis.summary) {
    addSuggestion(suggestions, {
      key: "analysis:stale",
      priority: "medium",
      category: "workflow",
      source: "analysis",
      title: "Re-run backend analysis to refresh runtime estimates",
      summary:
        "The displayed parameter and memory numbers were computed for an older draft and may no longer match the current config.",
      action: "Run backend analysis again after structural edits (heads, blocks, MLP multiplier, context length).",
      path: "/analyze/model",
      applyOptions: backendAnalysisApplyOptions(),
      score: 66,
    });
  } else if (backendAnalysis.phase === "idle" && localErrors.length === 0) {
    addSuggestion(suggestions, {
      key: "analysis:run",
      priority: "medium",
      category: "workflow",
      source: "analysis",
      title: "Run backend analysis before finalizing the design",
      summary:
        "Runtime analysis instantiates ConfigurableGPT and reports parameter counts and KV-cache memory, which are high-impact design constraints.",
      action: "Run backend analysis after the config is locally valid, then tune block count, heads, and MLP multiplier using the reported metrics.",
      path: "/analyze/model",
      applyOptions: backendAnalysisApplyOptions(),
      score: 65,
    });
  }

  if (backendAnalysis.phase === "error" && backendAnalysis.instantiationError) {
    addSuggestion(suggestions, {
      key: "analysis:error",
      priority: "high",
      category: "workflow",
      source: "analysis",
      title: "Backend analysis failed to instantiate the model",
      summary:
        "A runtime instantiation error usually means the config is syntactically valid but still incompatible with the implementation or a specific component combination.",
      action:
        "Reduce custom differences from a known-good baseline and re-run analysis after each change to isolate the failing edit.",
      path: "/analyze/model",
      applyOptions: backendAnalysisApplyOptions(),
      score: 83,
    });
  }

  void localWarnings; // local warnings are intentionally derived but only used in UI-facing rollups here.
}
