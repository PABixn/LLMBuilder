import type { ModelConfig, NormConfig } from "../../../../lib/defaults";

import type { BuilderMetrics, ConsecutiveBlockGroup } from "../../types";
import type { SuggestionBucket } from "./types";
import {
  addSuggestion,
  collectNormTypesFromMlpStep,
  countBy,
  kvHeadReductionApplyOptions,
  mlpActivationApplyOptions,
  mlpMultiplierApplyOptions,
  normFamilyApplyOptions,
  pluralize,
  topCounts,
} from "./helpers";

type AddArchitectureSuggestionsArgs = {
  suggestions: SuggestionBucket;
  modelConfig: ModelConfig;
  metrics: BuilderMetrics;
  consecutiveBlockGroups: ConsecutiveBlockGroup[];
};

export function addArchitectureSuggestions({
  suggestions,
  modelConfig,
  metrics,
  consecutiveBlockGroups,
}: AddArchitectureSuggestionsArgs): void {
  const blockStats = modelConfig.blocks.map((block) => {
    let attention = 0;
    let mlp = 0;
    let norm = 0;
    let activation = 0;
    for (const component of block.components) {
      if ("attention" in component) {
        attention += 1;
      } else if ("mlp" in component) {
        mlp += 1;
      } else if ("norm" in component) {
        norm += 1;
      } else if ("activation" in component) {
        activation += 1;
      }
    }
    return { attention, mlp, norm, activation };
  });

  const blocksWithoutNorm = blockStats.filter((block) => block.norm === 0).length;
  if (blockStats.length > 0 && blocksWithoutNorm > 0) {
    addSuggestion(suggestions, {
      key: "arch:missing-norm",
      priority: blocksWithoutNorm === blockStats.length ? "high" : "medium",
      category: "architecture",
      source: "combined",
      title: "Add normalization around major sublayers for stability",
      summary:
        blocksWithoutNorm === blockStats.length
          ? "No blocks currently include a norm component. Most transformer variants use LayerNorm or RMSNorm for training stability."
          : `${pluralize(blocksWithoutNorm, "block")} has no norm component, which may make behavior harder to compare across the stack.`,
      action:
        "Add LayerNorm or RMSNorm before attention/MLP (or follow your target architecture pattern consistently across blocks).",
      path: blocksWithoutNorm > 0 ? "blocks" : null,
      score: blocksWithoutNorm === blockStats.length ? 84 : 62,
    });
  }

  const blocksWithMultipleAttention = blockStats.filter((block) => block.attention > 1).length;
  if (blocksWithMultipleAttention > 0) {
    addSuggestion(suggestions, {
      key: "arch:multi-attention-per-block",
      priority: "medium",
      category: "architecture",
      source: "combined",
      title: "Double-check blocks with multiple attention components",
      summary:
        `${pluralize(blocksWithMultipleAttention, "block")} contains more than one attention component. This can be intentional, but it changes compute and memory quickly.`,
      action:
        "Confirm the ordering and purpose of each attention component, then use backend analysis to compare parameter and KV-cache growth against a baseline.",
      path: "blocks",
      score: 64,
    });
  }

  const blocksWithMultipleMlp = blockStats.filter((block) => block.mlp > 1).length;
  if (blocksWithMultipleMlp > 0) {
    addSuggestion(suggestions, {
      key: "arch:multi-mlp-per-block",
      priority: "medium",
      category: "architecture",
      source: "combined",
      title: "Double-check blocks with multiple MLP components",
      summary:
        `${pluralize(blocksWithMultipleMlp, "block")} contains multiple MLP components. This can increase capacity, but it also changes compute distribution and residual behavior.`,
      action:
        "Keep a baseline with one MLP per block so you can measure the impact of the extra MLP path cleanly.",
      path: "blocks",
      score: 63,
    });
  }

  const attentionComponents: Array<{ n_head: number; n_kv_head: number; path: string }> = [];
  const mlpMultipliers: number[] = [];
  const normTypes: NormConfig["type"][] = [];
  const mlpActivationTypes: string[] = [];
  let topLevelActivationComponents = 0;

  modelConfig.blocks.forEach((block, blockIndex) => {
    block.components.forEach((component, componentIndex) => {
      const componentPath = `blocks[${blockIndex}].components[${componentIndex}]`;
      if ("attention" in component) {
        attentionComponents.push({
          n_head: component.attention.n_head,
          n_kv_head: component.attention.n_kv_head,
          path: `${componentPath}.attention`,
        });
        return;
      }
      if ("mlp" in component) {
        mlpMultipliers.push(component.mlp.multiplier);
        for (const step of component.mlp.sequence) {
          if ("activation" in step) {
            mlpActivationTypes.push(step.activation.type);
          }
          collectNormTypesFromMlpStep(step, normTypes);
        }
        return;
      }
      if ("norm" in component) {
        normTypes.push(component.norm.type);
        return;
      }
      if ("activation" in component) {
        topLevelActivationComponents += 1;
      }
    });
  });

  if (topLevelActivationComponents > 0) {
    addSuggestion(suggestions, {
      key: "arch:top-level-activations",
      priority: "low",
      category: "architecture",
      source: "combined",
      title: "Top-level activation components are unusual in transformer blocks",
      summary:
        `${pluralize(topLevelActivationComponents, "standalone activation component")} sits directly in the block component list (outside MLP steps).`,
      action:
        "Keep this only if it is part of a deliberate architecture experiment; otherwise place activations inside MLP sequences.",
      path: "blocks",
      score: 48,
    });
  }

  const validHeadDims = attentionComponents
    .filter(
      (item) =>
        Number.isInteger(item.n_head) &&
        item.n_head > 0 &&
        Number.isInteger(modelConfig.n_embd) &&
        modelConfig.n_embd > 0 &&
        modelConfig.n_embd % item.n_head === 0
    )
    .map((item) => modelConfig.n_embd / item.n_head);
  const uniqueHeadDims = [...new Set(validHeadDims)];
  if (uniqueHeadDims.length > 1) {
    addSuggestion(suggestions, {
      key: "attn:head-dim-consistency",
      priority: "low",
      category: "consistency",
      source: "combined",
      title: "Consider standardizing head dimensions across attention components",
      summary:
        `The current design uses multiple head dimensions (${uniqueHeadDims.sort((a, b) => a - b).join(", ")}). That can be intentional, but it complicates ablations and runtime comparisons.`,
      action:
        "Start from one head_dim across blocks, then introduce targeted deviations only where you want to test a hypothesis.",
      path: "blocks",
      score: 50,
    });
  }

  const allAttentionFullKV =
    attentionComponents.length > 0 &&
    attentionComponents.every((item) => item.n_head > 0 && item.n_kv_head === item.n_head);
  const largestNHead = attentionComponents.reduce(
    (max, item) => (Number.isFinite(item.n_head) ? Math.max(max, item.n_head) : max),
    0
  );
  if (allAttentionFullKV && modelConfig.context_length >= 4096 && largestNHead >= 8) {
    addSuggestion(suggestions, {
      key: "attn:consider-gqa",
      priority: "medium",
      category: "efficiency",
      source: "combined",
      title: "Consider GQA to reduce KV-cache memory",
      summary:
        "All attention layers currently use n_kv_head = n_head (full multi-head KV). For long contexts this can make KV cache memory dominate runtime footprint.",
      action:
        "Experiment with grouped-query attention by lowering n_kv_head (while keeping it a divisor of n_head) and compare quality vs KV-cache savings.",
      path: attentionComponents[0]?.path ?? "blocks",
      applyOptions: kvHeadReductionApplyOptions(),
      score: 71,
    });
  }

  if (mlpMultipliers.length > 0) {
    const minMultiplier = Math.min(...mlpMultipliers);
    const maxMultiplier = Math.max(...mlpMultipliers);
    if (minMultiplier < 2) {
      addSuggestion(suggestions, {
        key: "mlp:small-multiplier",
        priority: "medium",
        category: "architecture",
        source: "combined",
        title: "Review small MLP multipliers",
        summary:
          `At least one MLP uses a multiplier below 2 (min ${minMultiplier}). This can be valid for tiny models, but it usually reduces feed-forward capacity.`,
        action:
          "Benchmark against a stronger baseline (for example x3-x4) before committing to a small multiplier for all blocks.",
        path: "blocks",
        applyOptions: mlpMultiplierApplyOptions([3, 4]),
        score: 64,
      });
    }
    if (maxMultiplier > 8) {
      addSuggestion(suggestions, {
        key: "mlp:large-multiplier",
        priority: "medium",
        category: "efficiency",
        source: "combined",
        title: "Review large MLP multipliers",
        summary:
          `At least one MLP uses a high multiplier (max ${maxMultiplier}), which can dominate parameter count and compute relative to attention.`,
        action:
          "Use backend analysis to compare parameter growth and consider reducing multiplier or applying it only to selected layers.",
        path: "blocks",
        applyOptions: mlpMultiplierApplyOptions([4, 6]),
        score: 63,
      });
    }
    const uniqueMultipliers = [...new Set(mlpMultipliers.map((value) => Number(value.toFixed(6))))];
    if (uniqueMultipliers.length > 2 && modelConfig.blocks.length >= 4) {
      addSuggestion(suggestions, {
        key: "mlp:multiplier-consistency",
        priority: "low",
        category: "consistency",
        source: "combined",
        title: "Standardize MLP multipliers for cleaner experiments",
        summary:
          `The model uses ${uniqueMultipliers.length} different MLP multipliers. This can make it harder to attribute training behavior to one architectural change.`,
        action:
          "Start with one multiplier across the stack, then vary only a small subset of layers for targeted experiments.",
        path: "blocks",
        applyOptions: mlpMultiplierApplyOptions([4, 3]),
        score: 49,
      });
    }
  }

  if (mlpActivationTypes.length > 0) {
    const activationCounts = topCounts(countBy(mlpActivationTypes));
    const activationTypeCount = activationCounts.length;
    const [topActivationType, topActivationCount] = activationCounts[0] ?? [null, 0];
    if (activationTypeCount === 1 && topActivationType === "relu" && metrics.blockCount >= 2) {
      addSuggestion(suggestions, {
        key: "mlp:activation-relu-only",
        priority: "low",
        category: "architecture",
        source: "combined",
        title: "Consider GELU or SiLU for transformer MLP activations",
        summary:
          "ReLU is valid, but many transformer baselines use GELU or SiLU for smoother MLP behavior and stronger language modeling performance.",
        action:
          "Try a GELU or SiLU variant on a copied baseline and compare training stability and validation loss.",
        path: "blocks",
        applyOptions: mlpActivationApplyOptions(),
        score: 47,
      });
    } else if (activationTypeCount > 2) {
      addSuggestion(suggestions, {
        key: "mlp:activation-mixed",
        priority: "low",
        category: "consistency",
        source: "combined",
        title: "Reduce activation variety unless you are running an ablation",
        summary:
          `The current design mixes ${activationTypeCount} MLP activation types, with ${String(topActivationType)} used most (${topActivationCount}).`,
        action:
          "Pick one default activation and introduce exceptions only where you want a controlled comparison.",
        path: "blocks",
        applyOptions: mlpActivationApplyOptions(),
        score: 46,
      });
    }
  }

  if (normTypes.length > 0) {
    const normTypeCount = topCounts(countBy(normTypes)).length;
    if (normTypeCount > 1) {
      addSuggestion(suggestions, {
        key: "norm:mixed-types",
        priority: "low",
        category: "consistency",
        source: "combined",
        title: "Mixing LayerNorm and RMSNorm increases experiment complexity",
        summary:
          "The current config mixes normalization families. This can be intentional, but it makes training behavior harder to attribute to a single change.",
        action:
          "Choose one normalization family for the baseline, then introduce a separate variant to compare.",
        path: "blocks",
        applyOptions: normFamilyApplyOptions(),
        score: 48,
      });
    }
  }

  if (modelConfig.blocks.length >= 6) {
    const repeatedCoverage =
      modelConfig.blocks.length === 0
        ? 0
        : consecutiveBlockGroups
            .filter((group) => group.count > 1)
            .reduce((sum, group) => sum + group.count, 0) / modelConfig.blocks.length;
    const singletonGroups = consecutiveBlockGroups.filter((group) => group.count === 1).length;

    if (repeatedCoverage < 0.5 && singletonGroups >= 3) {
      addSuggestion(suggestions, {
        key: "blocks:baseline-repeat-pattern",
        priority: "low",
        category: "consistency",
        source: "combined",
        title: "Use a repeated block template for the baseline",
        summary:
          "Most blocks are structurally unique. That can be powerful, but it makes debugging and ablation results harder to interpret early on.",
        action:
          "Keep one repeated block pattern for the baseline stack, then modify specific layers as controlled experiments.",
        path: "blocks",
        score: 45,
      });
    } else if (repeatedCoverage >= 0.5 && singletonGroups > 0) {
      addSuggestion(suggestions, {
        key: "blocks:inspect-outliers",
        priority: "low",
        category: "consistency",
        source: "combined",
        title: "Audit block outliers against the repeated stack",
        summary:
          "The model mostly uses repeated blocks with a few structural outliers. This is a good pattern for experiments, but the outliers carry most of the behavior change.",
        action:
          "Verify the outlier blocks are intentional and documented so future comparisons stay reproducible.",
        path: "blocks",
        score: 44,
      });
    }
  }
}
