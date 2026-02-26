import type { ValidationIssue } from "../../../../lib/api";
import type { MlpStep, NormConfig } from "../../../../lib/defaults";

import type {
  DesignSuggestion,
  Diagnostic,
  SuggestionCategory,
  SuggestionPriority,
} from "../../types";
import type {
  BackendIssueGroup,
  DiagnosticGroup,
  InternalSuggestion,
  SuggestionBucket,
} from "./types";

export const PRIORITY_ORDER: Record<SuggestionPriority, number> = {
  critical: 4,
  high: 3,
  medium: 2,
  low: 1,
};

function priorityBaseScore(priority: SuggestionPriority): number {
  if (priority === "critical") {
    return 100;
  }
  if (priority === "high") {
    return 80;
  }
  if (priority === "medium") {
    return 60;
  }
  return 40;
}

export function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

export function addSuggestion(
  bucket: SuggestionBucket,
  suggestion: Omit<InternalSuggestion, "score" | "applyOptions"> & {
    applyOptions?: InternalSuggestion["applyOptions"];
    score?: number;
  }
): void {
  const normalized: InternalSuggestion = {
    ...suggestion,
    applyOptions: suggestion.applyOptions ?? [],
    score: suggestion.score ?? priorityBaseScore(suggestion.priority),
  };
  const existing = bucket.get(normalized.key);
  if (!existing || normalized.score > existing.score) {
    bucket.set(normalized.key, normalized);
  }
}

export function groupedDiagnostics(diagnostics: Diagnostic[]): DiagnosticGroup[] {
  const grouped = new Map<string, DiagnosticGroup>();
  for (const item of diagnostics) {
    const key = `${item.level}:${item.source}:${item.message}`;
    const current = grouped.get(key);
    if (current) {
      current.count += 1;
      continue;
    }
    grouped.set(key, {
      level: item.level,
      source: item.source,
      message: item.message,
      count: 1,
      samplePath: item.path,
    });
  }
  return [...grouped.values()].sort((a, b) => {
    const severityDelta =
      (a.level === "error" ? 3 : a.level === "warning" ? 2 : 1) -
      (b.level === "error" ? 3 : b.level === "warning" ? 2 : 1);
    if (severityDelta !== 0) {
      return severityDelta * -1;
    }
    if (a.count !== b.count) {
      return b.count - a.count;
    }
    return a.message.localeCompare(b.message);
  });
}

export function backendIssueGroups(issues: ValidationIssue[]): BackendIssueGroup[] {
  const grouped = new Map<string, BackendIssueGroup>();
  for (const issue of issues) {
    const key = `${issue.code}:${issue.path}`;
    const current = grouped.get(key);
    if (current) {
      current.count += 1;
      continue;
    }
    grouped.set(key, { code: issue.code, count: 1, sample: issue });
  }
  return [...grouped.values()];
}

export function countBy<T extends string>(values: T[]): Map<T, number> {
  const counts = new Map<T, number>();
  for (const value of values) {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return counts;
}

export function topCounts<T extends string>(counts: Map<T, number>): Array<[T, number]> {
  return [...counts.entries()].sort(
    (a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0]))
  );
}

export function labelForSuggestionCategory(category: SuggestionCategory): string {
  if (category === "correctness") {
    return "Correctness";
  }
  if (category === "architecture") {
    return "Architecture";
  }
  if (category === "efficiency") {
    return "Efficiency";
  }
  if (category === "consistency") {
    return "Consistency";
  }
  return "Workflow";
}

export function labelForSuggestionPriority(priority: SuggestionPriority): string {
  if (priority === "critical") {
    return "Critical";
  }
  if (priority === "high") {
    return "High";
  }
  if (priority === "medium") {
    return "Medium";
  }
  return "Low";
}

export function backendAnalysisApplyOptions(): InternalSuggestion["applyOptions"] {
  return [
    {
      id: "run-analysis",
      label: "Apply",
      action: { kind: "run_backend_analysis" },
    },
  ];
}

export function mlpActivationApplyOptions(): InternalSuggestion["applyOptions"] {
  return [
    {
      id: "set-mlp-activation-gelu",
      label: "Apply GELU",
      action: { kind: "set_all_mlp_activations", activation: "gelu" },
    },
    {
      id: "set-mlp-activation-silu",
      label: "Apply SiLU",
      action: { kind: "set_all_mlp_activations", activation: "silu" },
    },
  ];
}

export function normFamilyApplyOptions(): InternalSuggestion["applyOptions"] {
  return [
    {
      id: "set-norm-family-layernorm",
      label: "Apply LayerNorm",
      action: { kind: "set_all_norm_family", normType: "layernorm" },
    },
    {
      id: "set-norm-family-rmsnorm",
      label: "Apply RMSNorm",
      action: { kind: "set_all_norm_family", normType: "rmsnorm", learnableGamma: true },
    },
  ];
}

export function mlpMultiplierApplyOptions(targets: number[]): InternalSuggestion["applyOptions"] {
  const seen = new Set<number>();
  return targets
    .filter((target) => Number.isFinite(target) && target > 0)
    .filter((target) => {
      const normalized = Number(target.toFixed(6));
      if (seen.has(normalized)) {
        return false;
      }
      seen.add(normalized);
      return true;
    })
    .map((target) => ({
      id: `set-mlp-multiplier-${String(target).replace(".", "_")}`,
      label: `Apply x${target}`,
      action: { kind: "set_all_mlp_multipliers", multiplier: target },
    }));
}

export function kvHeadReductionApplyOptions(): InternalSuggestion["applyOptions"] {
  return [
    {
      id: "set-kv-heads-half",
      label: "Apply GQA (1/2)",
      action: { kind: "set_all_attention_kv_heads", strategy: "half" },
    },
    {
      id: "set-kv-heads-one",
      label: "Apply MQA (1)",
      action: { kind: "set_all_attention_kv_heads", strategy: "one" },
    },
  ];
}

export function buildBackendIssueSuggestion(
  issue: ValidationIssue
): Omit<InternalSuggestion, "key" | "score" | "applyOptions"> & {
  key: string;
  applyOptions?: InternalSuggestion["applyOptions"];
  score?: number;
} {
  if (issue.code === "n_embd_not_divisible_by_n_head") {
    return {
      key: `backend-code:${issue.code}`,
      priority: "critical",
      category: "correctness",
      source: "backend",
      title: "Make embedding dimension divisible by attention heads",
      summary:
        "At least one attention component uses an n_head value that does not divide n_embd, which breaks head_dim calculation.",
      action:
        "Change n_embd or the affected n_head values so every attention block satisfies n_embd % n_head === 0.",
      path: issue.path,
      score: 108,
    };
  }
  if (issue.code === "n_kv_head_gt_n_head") {
    return {
      key: `backend-code:${issue.code}`,
      priority: "critical",
      category: "correctness",
      source: "backend",
      title: "Keep KV heads less than or equal to attention heads",
      summary:
        "Grouped-query attention requires n_kv_head <= n_head. Larger KV head counts are invalid and will block model construction.",
      action: "Reduce n_kv_head to be <= n_head for each affected attention component.",
      path: issue.path,
      score: 107,
    };
  }
  if (issue.code === "n_head_not_divisible_by_n_kv_head") {
    return {
      key: `backend-code:${issue.code}`,
      priority: "critical",
      category: "correctness",
      source: "backend",
      title: "Use a valid GQA grouping ratio",
      summary:
        "n_head must be divisible by n_kv_head so query heads can be grouped onto key/value heads without remainder.",
      action:
        "Pick n_kv_head as a divisor of n_head (for example 1, 2, 4, 8 when n_head is 8/16/32).",
      path: issue.path,
      score: 106,
    };
  }
  if (issue.code === "head_dim_not_even") {
    return {
      key: `backend-code:${issue.code}`,
      priority: "medium",
      category: "efficiency",
      source: "backend",
      title: "Prefer even head dimensions",
      summary:
        "The backend found an odd head_dim. Even head dimensions are usually safer for rotary embedding paths and kernel compatibility.",
      action:
        "Adjust n_embd or n_head so head_dim (n_embd / n_head) is even across attention components.",
      path: issue.path,
      score: 67,
    };
  }
  if (issue.code === "model_instantiation_failed") {
    return {
      key: `backend-code:${issue.code}`,
      priority: "high",
      category: "workflow",
      source: "analysis",
      title: "Backend could not instantiate ConfigurableGPT",
      summary:
        "The config passed schema validation but failed during runtime model construction, which usually indicates a semantic mismatch or unsupported combination.",
      action:
        "Review the backend analysis error details, reduce custom variations, and re-run analysis after each change.",
      path: issue.path,
      score: 85,
    };
  }

  return {
    key: `backend-code:${issue.code}`,
    priority: "medium",
    category: "workflow",
    source: "backend",
    title: "Review backend validation issue",
    summary: issue.message,
    action: "Adjust the affected fields, then wait for backend validation to re-run automatically.",
    path: issue.path,
    score: 62,
  };
}

export function collectNormTypesFromMlpStep(step: MlpStep, normTypes: NormConfig["type"][]): void {
  if ("norm" in step) {
    normTypes.push(step.norm.type);
  }
}

export function finalizeSuggestions(bucket: SuggestionBucket): DesignSuggestion[] {
  const ordered = [...bucket.values()].sort((a, b) => {
    if (a.score !== b.score) {
      return b.score - a.score;
    }
    const priorityDelta = PRIORITY_ORDER[b.priority] - PRIORITY_ORDER[a.priority];
    if (priorityDelta !== 0) {
      return priorityDelta;
    }
    return a.title.localeCompare(b.title);
  });

  return ordered.map((item) => ({
    id: item.key,
    priority: item.priority,
    category: item.category,
    source: item.source,
    title: item.title,
    summary: item.summary,
    action: item.action,
    path: item.path,
    applyOptions: item.applyOptions,
    score: item.score,
  }));
}
