import type { Dispatch, SetStateAction } from "react";

import type { SuggestionApplyOption, StudioDocument } from "../types";

type UseDesignSuggestionActionsArgs = {
  backendAnalysisPhase: "idle" | "running" | "success" | "error";
  localErrorCount: number;
  setDocumentState: Dispatch<SetStateAction<StudioDocument>>;
  runBackendAnalysis: () => Promise<void>;
};

function selectKvHeadForStrategy(nHead: number, strategy: "half" | "one"): number {
  if (!Number.isInteger(nHead) || nHead < 1) {
    return nHead;
  }
  if (strategy === "one") {
    return 1;
  }
  if (nHead <= 1) {
    return 1;
  }
  const target = Math.max(1, Math.floor(nHead / 2));
  for (let candidate = target; candidate >= 1; candidate -= 1) {
    if (nHead % candidate === 0) {
      return candidate;
    }
  }
  return 1;
}

export function useDesignSuggestionActions({
  backendAnalysisPhase,
  localErrorCount,
  setDocumentState,
  runBackendAnalysis,
}: UseDesignSuggestionActionsArgs) {
  function canApplySuggestion(
    suggestion: { applyOptions: SuggestionApplyOption[] }
  ): boolean {
    return suggestion.applyOptions.length > 0;
  }

  function suggestionApplyDisabled(option: SuggestionApplyOption): boolean {
    if (option.action.kind === "run_backend_analysis") {
      return backendAnalysisPhase === "running" || localErrorCount > 0;
    }
    return false;
  }

  function suggestionApplyTitle(option: SuggestionApplyOption): string {
    switch (option.action.kind) {
      case "run_backend_analysis":
        if (localErrorCount > 0) {
          return "Fix local errors before running backend analysis.";
        }
        if (backendAnalysisPhase === "running") {
          return "Backend analysis is already running.";
        }
        return "Run backend analysis now.";
      case "set_all_mlp_activations":
        return `Set all MLP activation steps to ${option.action.activation.toUpperCase()}.`;
      case "set_all_norm_family":
        return option.action.normType === "layernorm"
          ? "Convert all norm layers to LayerNorm."
          : "Convert all norm layers to RMSNorm.";
      case "set_all_mlp_multipliers":
        return `Set all MLP multipliers to x${option.action.multiplier}.`;
      case "set_all_attention_kv_heads":
        return option.action.strategy === "half"
          ? "Reduce KV heads toward half of n_head while keeping valid divisibility."
          : "Set all KV heads to 1 (MQA).";
      default:
        return "Apply suggestion";
    }
  }

  function applySuggestion(option: SuggestionApplyOption): void {
    switch (option.action.kind) {
      case "run_backend_analysis":
        if (backendAnalysisPhase === "running" || localErrorCount > 0) {
          return;
        }
        void runBackendAnalysis();
        return;
      case "set_all_mlp_activations": {
        const targetActivation = option.action.activation;
        setDocumentState((current) => {
          let changed = false;
          const nextBlocks = current.blocks.map((block) => {
            let blockChanged = false;
            const nextComponents = block.components.map((component) => {
              if (component.kind !== "mlp") {
                return component;
              }

              let componentChanged = false;
              const nextSequence = component.mlp.sequence.map((step) => {
                if (step.kind !== "activation") {
                  return step;
                }
                if (step.activation.type === targetActivation) {
                  return step;
                }
                componentChanged = true;
                return {
                  ...step,
                  activation: {
                    ...step.activation,
                    type: targetActivation,
                  },
                };
              });

              if (!componentChanged) {
                return component;
              }

              blockChanged = true;
              return {
                ...component,
                mlp: {
                  ...component.mlp,
                  sequence: nextSequence,
                },
              };
            });

            if (!blockChanged) {
              return block;
            }

            changed = true;
            return {
              ...block,
              components: nextComponents,
            };
          });

          return changed ? { ...current, blocks: nextBlocks } : current;
        });
        return;
      }
      case "set_all_norm_family": {
        const { normType } = option.action;
        const learnableGamma = option.action.learnableGamma ?? true;
        setDocumentState((current) => {
          let changed = false;
          const nextBlocks = current.blocks.map((block) => {
            let blockChanged = false;
            const nextComponents = block.components.map((component) => {
              if (component.kind === "norm") {
                const currentNorm = component.norm;
                const nextNorm =
                  normType === "layernorm"
                    ? { type: "layernorm" as const }
                    : { type: "rmsnorm" as const, learnable_gamma: learnableGamma };
                const same =
                  currentNorm.type === nextNorm.type &&
                  (currentNorm.type !== "rmsnorm" ||
                    ("learnable_gamma" in currentNorm &&
                      currentNorm.learnable_gamma ===
                        ("learnable_gamma" in nextNorm
                          ? nextNorm.learnable_gamma
                          : undefined)));
                if (same) {
                  return component;
                }
                blockChanged = true;
                return {
                  ...component,
                  norm: nextNorm,
                };
              }

              if (component.kind !== "mlp") {
                return component;
              }

              let componentChanged = false;
              const nextSequence = component.mlp.sequence.map((step) => {
                if (step.kind !== "norm") {
                  return step;
                }
                const currentNorm = step.norm;
                const nextNorm =
                  normType === "layernorm"
                    ? { type: "layernorm" as const }
                    : { type: "rmsnorm" as const, learnable_gamma: learnableGamma };
                const same =
                  currentNorm.type === nextNorm.type &&
                  (currentNorm.type !== "rmsnorm" ||
                    ("learnable_gamma" in currentNorm &&
                      currentNorm.learnable_gamma ===
                        ("learnable_gamma" in nextNorm
                          ? nextNorm.learnable_gamma
                          : undefined)));
                if (same) {
                  return step;
                }
                componentChanged = true;
                return {
                  ...step,
                  norm: nextNorm,
                };
              });

              if (!componentChanged) {
                return component;
              }

              blockChanged = true;
              return {
                ...component,
                mlp: {
                  ...component.mlp,
                  sequence: nextSequence,
                },
              };
            });

            if (!blockChanged) {
              return block;
            }
            changed = true;
            return {
              ...block,
              components: nextComponents,
            };
          });

          return changed ? { ...current, blocks: nextBlocks } : current;
        });
        return;
      }
      case "set_all_mlp_multipliers": {
        const targetMultiplier = option.action.multiplier;
        setDocumentState((current) => {
          let changed = false;
          const nextBlocks = current.blocks.map((block) => {
            let blockChanged = false;
            const nextComponents = block.components.map((component) => {
              if (component.kind !== "mlp" || component.mlp.multiplier === targetMultiplier) {
                return component;
              }
              blockChanged = true;
              return {
                ...component,
                mlp: {
                  ...component.mlp,
                  multiplier: targetMultiplier,
                },
              };
            });
            if (!blockChanged) {
              return block;
            }
            changed = true;
            return {
              ...block,
              components: nextComponents,
            };
          });
          return changed ? { ...current, blocks: nextBlocks } : current;
        });
        return;
      }
      case "set_all_attention_kv_heads": {
        const strategy = option.action.strategy;
        setDocumentState((current) => {
          let changed = false;
          const nextBlocks = current.blocks.map((block) => {
            let blockChanged = false;
            const nextComponents = block.components.map((component) => {
              if (component.kind !== "attention") {
                return component;
              }
              const nextKv = selectKvHeadForStrategy(component.attention.n_head, strategy);
              if (component.attention.n_kv_head === nextKv) {
                return component;
              }
              blockChanged = true;
              return {
                ...component,
                attention: {
                  ...component.attention,
                  n_kv_head: nextKv,
                },
              };
            });
            if (!blockChanged) {
              return block;
            }
            changed = true;
            return {
              ...block,
              components: nextComponents,
            };
          });
          return changed ? { ...current, blocks: nextBlocks } : current;
        });
        return;
      }
      default:
        return;
    }
  }

  return {
    canApplySuggestion,
    suggestionApplyDisabled,
    suggestionApplyTitle,
    applySuggestion,
  };
}
