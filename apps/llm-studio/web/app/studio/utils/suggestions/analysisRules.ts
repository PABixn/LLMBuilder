import { formatBytes, formatCompactCount } from "../format";
import type { BackendAnalysisState } from "../../types";
import type { SuggestionBucket } from "./types";
import { addSuggestion, kvHeadReductionApplyOptions } from "./helpers";

type AddBackendAnalysisMetricSuggestionsArgs = {
  suggestions: SuggestionBucket;
  backendAnalysis: BackendAnalysisState;
};

export function addBackendAnalysisMetricSuggestions({
  suggestions,
  backendAnalysis,
}: AddBackendAnalysisMetricSuggestionsArgs): void {
  if (!backendAnalysis.summary) {
    return;
  }

  const summary = backendAnalysis.summary;
  const kvBytes = summary.estimated_kv_cache_bytes_for_context_fp16;
  const paramBf16Bytes = summary.parameter_memory_bytes_bf16;

  if (kvBytes >= 8 * 1024 ** 3) {
    addSuggestion(suggestions, {
      key: "analysis:kv-cache-very-high",
      priority: "high",
      category: "efficiency",
      source: "analysis",
      title: "KV-cache memory is very large at the current context length",
      summary:
        `Estimated fp16 KV cache for one full context is ${formatBytes(kvBytes)}, which is likely to dominate memory for inference and debugging runs.`,
      action:
        "Lower context length, use GQA (smaller n_kv_head), or reduce n_embd / block count before scaling further.",
      path: "/analyze/model",
      applyOptions: kvHeadReductionApplyOptions(),
      score: 88,
    });
  } else if (kvBytes >= 2 * 1024 ** 3) {
    addSuggestion(suggestions, {
      key: "analysis:kv-cache-high",
      priority: "medium",
      category: "efficiency",
      source: "analysis",
      title: "KV-cache memory is a major design constraint",
      summary:
        `Estimated fp16 KV cache for a full context is ${formatBytes(kvBytes)}. Long-context experiments may be bottlenecked by cache memory before parameter memory.`,
      action:
        "Track KV cache alongside parameter count when tuning heads, n_kv_head, n_embd, and context length.",
      path: "/analyze/model",
      applyOptions: kvHeadReductionApplyOptions(),
      score: 69,
    });
  }

  if (paramBf16Bytes > 0 && kvBytes > paramBf16Bytes) {
    addSuggestion(suggestions, {
      key: "analysis:kv-dominates-params",
      priority: "medium",
      category: "efficiency",
      source: "analysis",
      title: "KV cache exceeds parameter memory",
      summary:
        `At the current context length, fp16 KV cache (${formatBytes(kvBytes)}) is larger than bf16 parameter memory (${formatBytes(paramBf16Bytes)}).`,
      action:
        "Optimize for runtime memory, not just parameter count: GQA, shorter context, or smaller hidden size can help more than trimming a few layers.",
      path: "/analyze/model",
      applyOptions: kvHeadReductionApplyOptions(),
      score: 72,
    });
  }

  if (summary.total_parameters >= 1_000_000_000) {
    addSuggestion(suggestions, {
      key: "analysis:param-count-billion-plus",
      priority: "medium",
      category: "workflow",
      source: "analysis",
      title: "Parameter count is in the billion-scale range",
      summary:
        `The current design is about ${formatCompactCount(summary.total_parameters)} parameters. Training and inference setup choices now matter as much as architecture details.`,
      action:
        "Plan batch size, optimizer memory, checkpoint cadence, and hardware targets before expanding the architecture further.",
      path: "/analyze/model",
      score: 68,
    });
  }
}
